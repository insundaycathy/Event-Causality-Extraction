import pandas as pd
from tqdm import tqdm
import re
import pickle
import os
import openai
import random
import spacy
import string
import pyinflect
from nltk.stem.wordnet import WordNetLemmatizer
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import json
import argparse
from sentence_transformers.util import cos_sim
eval_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

nlp = spacy.load("en_core_web_sm")

def get_args(description='COPES event causality extraction'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--save_dict", default='output.csv', type=str, help="output_path for compute csv file")
    parser.add_argument("--model", default='gpt-3.5-turbo', type=str, help="model name")
    parser.add_argument("--input_data", default='test_set_no_answers.csv', type=str, help="the csv file that stores the COPES dataset")
    parser.add_argument("--prompt", default='event_graph.txt', type=str, help="the path to the prompt file")
    args = parser.parse_args()

    with open('args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args

def get_cause_effect_id(relation,story_list):
    causality = re.findall(r'\[.+\]', relation)[-1].replace('[','').replace(']','')
    causality = causality.replace('caused','causes')
    cause, effect = causality.split('causes')
    cause_id = get_id(cause,story_list)
    effect_id = get_id(effect,story_list)
    return cause_id,effect_id
    

def get_id (cause,story_list):
    cause_id =int(cause.replace('Event','').replace(' ',''))

    cause_id = story_list[cause_id]
    
    return cause_id

def get_sent_1n2(relation):
    try:
        state = re.findall(r'\(.+\)', relation)[-1].replace('(','').replace(')','').split(':')[-1]
    except:
        state = relation.split('>')[0].split('in Event')[0]
    sentence_1 = relation.split('>')[0].split('in Event')[0]
    sentence_2 = relation.split('>')[-1].split('in Event')[0]
    return sentence_1,state, sentence_2

def OpenAI_api(msg):
    key_list = []
    openai.api_key = random.choice(key_list)

    response = openai.ChatCompletion.create( 
    model = args.model,
    messages = msg,
    max_tokens =400,  
    temperature =0
    )
    output_str = response["choices"][0]["message"]["content"].strip()
    return output_str

# use a rule based system to transfrom LLM output to GLUCOSE output format
def relation2output(relation,story_list):
        pattern = r'\([^()]*\)'
        relation = relation.split(':')[-1]
        try:
            state = re.findall(pattern, relation)[0]
        except:
            return -1,None,None,None, None
        state = state.replace('Node','Event').replace('(','').replace(')','')
        cause,effect = state.split('->')
        if 'Event' in cause and 'Event' in effect:
            cause_id = get_id(cause,story_list)
            effect_id = get_id(effect,story_list)
            output = cause_id + '>Causes/Enables>' + effect_id
            return 1, cause_id,effect_id,output,output
        else:
            return -1,None,None,None, None

    

        
def load_eval_csv(path,test_dict, eval_dict):
    test_csv = pd.read_csv(path)
    with open(path.replace('.csv','.pkl'), 'rb') as file:
        eval = pickle.load(file)
    test_dict.append(test_csv)
    eval_dict.append(eval)
    return test_dict,eval_dict

def get_index(numbers):

    # Create a list of (number, index) tuples
    indexed_numbers = [(num, index) for index, num in enumerate(numbers)]

    # Sort the list of tuples by the numbers in descending order
    sorted_indices = sorted(indexed_numbers, key=lambda x: x[0], reverse=True)

    # Extract the sorted indices
    sorted_indices = [index for _, index in sorted_indices]

    return sorted_indices



def causality_extraction():

    test_csv = pd.read_csv(args.input_data)
    save_dict = args.save_dict.replace('.csv','.pkl')

    with open(args.prompt, 'r') as f : # load prompt
        prompt = f.read()

    if os.path.exists(save_dict):
        with open(save_dict,'rb') as f:
            story = pickle.load(f)
    else:
        story = dict()
        for i in tqdm(range(len(test_csv))):
            storyline=test_csv.loc[i, 'story'].split('****')
            selected_sentence= test_csv.loc[i, 'selected_sentence']
            tmp = dict()
            tmp['story'] = storyline
            tmp['selected_sentence'] = set()
            tmp['selected_sentence'].add(selected_sentence)
            tmp['graph'] = ''
            story[storyline[0][:20]] =tmp
        with open(save_dict,'wb') as f:
            pickle.dump(story,f)
    k = 0
    for id in story:
        k = k+1
        if len(story[id]['graph']) == 0:
            storyline = ''
            for i in range(len(story[id]['story'])):
                storyline = storyline+ 'Node '+str(i)+':'+story[id]['story'][i]+'\n'

            input_str = prompt.replace('<storyline>', storyline)
            msg = [ {"role": "user", "content": input_str}]
            output_str  = OpenAI_api(msg) #original sentence 
            
            msg.append({"role": "system", "content": output_str})
            story[id]['graph'] = output_str
            graph = output_str
        else:
            graph = story[id]['graph']
            

        with open(save_dict,'wb') as f:
            pickle.dump(story,f)
        
        graph = graph.split('\n')

        tmp = []
        for line in graph:
            if 'Node' not in line.split(':')[0]:
                tmp.append(line)
        graph = tmp


        story[id]['sents_dim']=dict()
        for i in range(5):
            story[id]['sents_dim'][story[id]['story'][i]]=dict()
            for j in range(10):
                story[id]['sents_dim'][story[id]['story'][i]][str(j+1)+'_specificNL']= 'escaped'

        for relation in graph:
            
            dim, cause_id, effect_id,dim_big,dim_small = relation2output(relation,story[id]['story'])

            if dim == 1:
                if cause_id!=None:
                    story[id]['sents_dim'][cause_id]['6_specificNL'] = dim_big
                story[id]['sents_dim'][effect_id]['1_specificNL'] = dim_small
            elif dim == -1:
                continue    

        with open(save_dict,'wb') as f:
            pickle.dump(story,f)        

    experiment_output = {}
    for i in tqdm(range(len(test_csv))):

        storyline=test_csv.loc[i, 'story'].split('****')
        selected_sentence= test_csv.loc[i, 'selected_sentence']
        experimentid = test_csv.loc[i, 'unique_id']
        try:
            experiment_output[experimentid]=story[storyline[0][:20]]['sents_dim'][selected_sentence]
        except:
            continue
    
    answers = pd.DataFrame.from_dict(experiment_output, orient='index').reset_index().rename(columns={'index': 'unique_id'})
    test_csv.set_index('unique_id', inplace=True)
    test_csv.update(answers.set_index('unique_id'))
    test_csv.reset_index() 
    test_csv.to_csv(args.save_dict, index=False)
           

args = get_args()
causality_extraction()
