import os
import openai
import sys
import json
import numpy as np
import pandas as pd 
import random
import re
from tqdm import tqdm
import time
import argparse

    
import tiktoken
import tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

import openai
from openai import OpenAI


import json
import os
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
) 

# These are the basic functions to call ChatGPT
def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


@retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(6), before=before_retry_fn)
def completion_with_backoff(**kwargs):
    key_list=["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"]
    client = OpenAI(api_key = random.choice(key_list))
    return client.chat.completions.create(**kwargs)

def get_completion(prompt, args):
    if args.gpt_model=='gpt3.5':
        model ='gpt-3.5-turbo-0613'
    elif args.gpt_model=='gpt4':
        model ='gpt-4'

    messages = [{"role": "user", "content": prompt}]
    response = completion_with_backoff(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        max_tokens=args.max_tokens,
        seed = int(args.seed)
    )
    return response.choices[0].message.content.strip()



def main(args):

    with open('prompts/EN_generate/'+args.OpenMEVA_dataset+'.txt', 'r') as f :
        Demo = f.read()
    
    story_dict = np.load(args.data_path, allow_pickle=True).item()

    for prompt_id, experiments in tqdm(story_dict.items()):  # est 4mins
        models_output = experiments['gen']
        for gen_model_id, model_output in models_output.items(): # we have 5 models generation story for every story_prompt
            model_output['LM_output'] = {}
            if args.OpenMEVA_dataset == 'ROC':
                input_str = Demo.replace("<S1>", model_output['sentence1'])
                input_str = input_str.replace("<S2>", model_output['sentence2'])
                input_str = input_str.replace("<S3>", model_output['sentence3'])

                if 'sentence4' in model_output:
                    input_str = input_str.replace("<S4>", model_output['sentence4'])
                else:
                    input_str = input_str.replace("Node 3: <S4>", '')           
                
                if 'sentence5' in model_output:
                    input_str = input_str.replace("<S5>", model_output['sentence5'])   
                else:
                    input_str = input_str.replace("Node 4: <S5>", '')   
            

            elif args.OpenMEVA_dataset == 'WP':
                i=1
                Nodes_string=''
                sentence_i = 'sentence'+str(i)
                while sentence_i in model_output.keys():
                    Nodes_string += 'Node ' + str(i-1)+': ' + model_output[sentence_i]+'\n'
                    i+=1
                    sentence_i = 'sentence'+str(i)

                input_str = Demo.replace("<Nodes>", Nodes_string)


            msg = [ {"role": "user", "content": input_str}
                    ]
            try:
                output_str = get_completion(msg, args)
            except:
                output_str = get_completion(msg, args)


            model_output['LM_output']= output_str
            np.save(args.EN_data_path, story_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some files.')

    # Add the arguments
    parser.add_argument('--OpenMEVA_dataset', '-d', type=str, required=True, help='ROC or WP')
    parser.add_argument('--data_path', '-dp', type=str, default='./datasets/OpenMeva_ROC.npy', help='The path to the OpenMEVA story file')
    parser.add_argument('--EN_data_path', '-EN', type=str, default='EN_OpenMeva_ROC.npy', help='The path to store event graph output file')
    parser.add_argument('--seed', '-s', type=int, default=2, help='set seed for ChatGPT')
    parser.add_argument('--gpt_model', '-g', type=str, default='gpt3.5', help='choose the ChatGPT model, can be "gpt3.5" or "gpt4"')
    parser.add_argument('--max_token', '-m', type=int, default=256, help='the max token for ChatGPT completion, suggest 256')

    args = parser.parse_args()
    main()
