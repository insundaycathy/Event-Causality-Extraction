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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

import transformers


def get_args(description='COPES event causality extraction'):
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument("--save_dict", default='output.pkl', type=str, help="output_path for compute event causal graphs")
    parser.add_argument("--model", default='gpt-3.5-turbo', type=str, help="model name")
    parser.add_argument("--input_data", default='COPES.json', type=str, help="the json file that stores the COPES dataset")
    parser.add_argument("--data_split", default='split_idx_42.json', type=str, help="the json file that stores the data split of the COPES dataset")
    parser.add_argument("--prompt", default='event_graph.txt', type=str, help="the path to the prompt file")
    args = parser.parse_args()

    with open('args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return args




def OpenAI_api(msg):
    key_list = [] # fill in ur own keys.
    openai.api_key = random.choice(key_list)

    response = openai.ChatCompletion.create( 
    model = args.model,
    messages = msg,
    max_tokens =400,  
    temperature =0
    )
    output_str = response["choices"][0]["message"]["content"].strip()
    return output_str




def find_causes(graph):
    graph = graph.split('\n')
    causes = []
    for line in graph:
        if 'Edge' in line:
            line = line.split(':')[-1]
            cause_effect =re.findall(r'\d+', line)
            if len(cause_effect) ==2:
                cause = int(cause_effect[0])
                effect = int(cause_effect[1])
                if effect == 4:
                    causes.append(cause)
    return causes


def unpack_label(data_list):
    ground_truth_list = []
    for i in range(len(data_list)):
        pos_set = set(data_list[i])
        cur_label = [int(i in pos_set) for i in range(4)]
        ground_truth_list.extend(cur_label)
    return ground_truth_list


def get_f1(causal_label_list, ground_truth_list):
    shared_positive_list = causal_label_list * ground_truth_list
    shared_count = np.sum(shared_positive_list)
    p_count = np.sum(causal_label_list)
    r_count = np.sum(ground_truth_list)

    precision = shared_count / p_count if p_count != 0 else 0
    recall = shared_count / r_count if r_count != 0 else 0
    f1 = 2 * recall * precision / (recall + precision)

    return precision, recall, f1


def get_macro_f1(causal_label_list, ground_truth_list):
    p1, r1, f1 = get_f1(causal_label_list, ground_truth_list)

    reversed_causal_label_list = 1 - causal_label_list
    reversed_ground_truth_list = 1 - ground_truth_list
    p0, r0, f0 = get_f1(reversed_causal_label_list, reversed_ground_truth_list)

    p = (p0 + p1) / 2
    r = (r0 + r1) / 2
    f = (f0 + f1) / 2
    return p, r, f, f1

def get_acc(causal_label_list, ground_truth_list):
    shared_positive_list = causal_label_list == ground_truth_list
    shared_count = np.sum(shared_positive_list)
    return shared_count / len(shared_positive_list)


args = get_args()
save_dict = args.save_dict
with open(args.data_split,'r') as file:
    index = json.load(file)

if 'gpt' not in args.model:
    model = AutoModelForCausalLM.from_pretrained(args.model,  trust_remote_code=True,  device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


prompt = open(args.prompt,'r').read()


data = []
with open(args.input_data, 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace and load the JSON from each line
        json_object = json.loads(line.strip())
        data.append(json_object)

if os.path.exists(save_dict):
    with open(save_dict,'rb') as f:
        story_graph = pickle.load(f)
else:
    story_graph = dict()


pred_list = []
true_list = [] 
for i in range(len(data)):
    if i in index['valid']:
        continue

    story = data[i]['story']
    storyline = ''
    for j in range(len(story)):
        storyline = storyline+'Node: '+story[j]+'\n'
    
    if i in story_graph:
        graph = story_graph[i]
    else:
        input_str = prompt.replace('<storyline>', storyline)
        if 'gpt' in args.model:
            msg = [ {"role": "user", "content": input_str}]
            output_str  = OpenAI_api(msg) #original sentence
            msg.append({"role": "system", "content": output_str})
        else:
            sequences = pipeline(
                    input_str,
                    max_length=1536,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id, # Setting `pad_token_id` to `eos_token_id`:11 for open-end generation.
                    return_full_text=False
                )

            output_str = sequences[0]['generated_text']

        graph = output_str
        story_graph[i] = output_str
        with open(save_dict,'wb') as f:
            pickle.dump(story_graph,f)
    
    pred_causes = find_causes(graph)

    true_causes = data[i]['cause_idx']
    pred_list.append(pred_causes)
    true_list.append(true_causes)

pred_list = unpack_label(pred_list)
true_list = unpack_label(true_list)

causal_label_list = np.array(pred_list)
ground_truth_list = np.array(true_list)
res = {}
res["acc"] = get_acc(causal_label_list, ground_truth_list)
res["ma-p"], res["ma-r"], res["ma-f1"], res["f1"] = get_macro_f1(causal_label_list, ground_truth_list)

print(res)


    
    