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
    key_list=["sk-"]
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


def caputer_numbers(output_str):
    pattern_decimal  =r'\b\d+\.\d\b'
    matches = re.findall(pattern_decimal, output_str)
    if len(matches)>0:
        numbers = [float(match) for match in matches]
    else:
        number_strings = {
            "zero": 0,
            "none": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5
        }

        pattern = r"\b(|zero|none|one|two|three|four|five|\d+)\b"
        matches = re.findall(pattern, output_str)
        numbers = []
        for match in matches:
            if match.isdigit():
                numbers.append(int(match))
            elif match in number_strings:
                numbers.append(number_strings[match])
        # print(output_str)
        if numbers:
            score = numbers[0]
        else:
            score = 1 # if no number is found, we assume the score is 1
    return score


########### gpt model and prompt ##########
def main(args):
    file_path = 'score_'+args.OpenMEVA_dataset+'_'+args.prompt_type+'_seed_'+args.seed+'.npy'

    with open('./prompts/'+args.prompt_type+'.txt', 'r') as f :
        Demo = f.read()

    story_dict = np.load(args.data_path, allow_pickle=True).item()
    # 

    if args.prompt_type == 'EN':
        EN_dict = np.load(args.EN_data_path, allow_pickle=True).item()
        
    for prompt_id, experiments in tqdm(story_dict.items()):  # est 4mins
        models_output = experiments['gen']
        story_prompt = experiments['prompt']
        for gen_model_id, model_output in models_output.items(): # we have 5 models generation story for every story_prompt
            if 'LLM_output' in model_output:
                output_str = model_output['LLM_output']
                score = caputer_numbers(output_str.lower())
                model_output['LLM_score']={args.gpt_model: score[-1]}
                np.save(file_path, story_dict)
            
            if args.OpenMEVA_dataset=='WP':
                generated = model_output['storyline']
            elif args.OpenMEVA_dataset=='ROC':
                generated = model_output['text']

            

            prompt = Demo.replace("<generated Story>", generated)
            prompt = prompt.replace("<S1>", story_prompt)
            if args.prompt_type == 'EN':
                EN_output =EN_dict[prompt_id]['gen'][gen_model_id]['LM_output']
                prompt = prompt.replace("<event graph>", EN_output)

            output_str = get_completion(prompt, args)
            model_output['LLM_output']=output_str
            time.sleep(3)

            score = caputer_numbers(output_str.lower())
            model_output['LLM_score']={args.gpt_model: score}
            

        np.save(file_path, story_dict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some files.')

    # Add the arguments
    parser.add_argument('--prompt_type', '-p', type=str, required=True, help='The prompt type, choose from "causal", "orig", etc. see the prompt text files for the complete list')
    parser.add_argument('--OpenMEVA_dataset', '-d', type=str, required=True, help='ROC or WP')
    parser.add_argument('--data_path', '-dp', type=str, default='./datasets/OpenMeva_ROC.npy', help='The path to the OpenMEVA file')
    parser.add_argument('--EN_data_path', '-EN', type=str, default='EN_OpenMeva_ROC.npy', help='The path to event graph output file')
    parser.add_argument('--seed', '-s', type=int, help='set seed for ChatGPT')
    parser.add_argument('--gpt_model', '-g', type=str, default='gpt3.5', help='choose the ChatGPT model, can be "gpt3.5" or "gpt4"')
    parser.add_argument('--max_token', '-m', type=int, default=10, help='the max token for ChatGPT completion, suggest 10')

    args = parser.parse_args()
    main()
