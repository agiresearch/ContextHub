import json
import argparse
from utils import set_seed
from prompt import TEXT_UNDERSTANDING_PROMPTS, TEXT_UNDERSTANDING_PARSE_PROMPTS
from tqdm import tqdm
import re
import random
import time
import os
from model import call_data_generation_model, call_evaluation_model, call_anthropic_api, load_evaluation_model

def parse_function(result):
    assert '<answer>' in result
    start = result.index('<answer>')
    end = result.index('</answer>')
    answer = result[start + len('<answer>'):end]
    return answer

def parse_result(parse_prompt, result):
    prompt = parse_prompt + '\n' + result
    try:
        answer = re.findall(r'<answer>(.*?)</answer>', result, re.DOTALL)[-1]
    except:
        while True:
            try:
                parsed_result = call_anthropic_api(prompt)
                answer = parse_function(parsed_result)
                break
            except KeyboardInterrupt:
                raise
            except:
                time.sleep(0.1)

    return answer


def run(args, llm, evaluate_prompt, parse_prompt, data, dataset):
    prompts = []
    # one correct logic template
    # 4 incorrect logic template
    correct_abstract_logic = data['question']
    wrong_candidate_abstract_logic = random.sample([data['question'] for data in dataset if data['question'] != correct_abstract_logic], k=4)
    # contextualized result
    for domain in data.keys():
        if type(data[domain]) != list or domain == 'question':
            continue
        # create data
        for i in range(len(data[domain])):
            text = data[domain][i]['<nl>']
            for logic in [correct_abstract_logic] + wrong_candidate_abstract_logic:
                one_prompt = evaluate_prompt.format(logic, text)
                prompts.append(one_prompt)
    # run model
    results = call_evaluation_model(model=args.model, text_prompt=prompts, llm=llm)
    new_results = []
    for r in tqdm(results, desc="Parsing generated answers"):
        new_results.append(parse_result(parse_prompt, r))
    # collect result
    data = collect_answer(data, new_results)

    return data

def collect_answer(data, results):
    j = 0
    for domain in data.keys():
        if type(data[domain]) != list or domain == 'question':
            continue
        for i in range(len(data[domain])):
            data[domain][i]['parse'] = results[j:j+5]
            j += 5
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tulu-13", help="The model to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--level", type=int, default=1, help="The difficulty level")
    parser.add_argument("--dataset", type=str, default="abductive_logic", help="The dataset")
    parser.add_argument("--proficiency", type=str, default=None, help="The proficiency level")
    args = parser.parse_args()

    set_seed(args.seed)

    evaluate_prompt = TEXT_UNDERSTANDING_PROMPTS
    parse_prompt = TEXT_UNDERSTANDING_PARSE_PROMPTS

    data_path = f'data/data_level{args.level}/'

    result_folder = f'result_parsing/data_level{args.level}/{args.model}/'
    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if args.proficiency is not None:
        data_path += f'{args.proficiency}_'
    data_path += f'{args.dataset}.json'

    if args.proficiency is not None:
        result_folder = result_folder + f'{args.proficiency}_'
    result_path = result_folder + f'{args.dataset}_{args.seed}.json'

    print('=='*20)
    print(result_path)
    print('=='*20)

    with open(data_path, 'r') as f:
        dataset = json.load(f)

    llm = load_evaluation_model(args.model)

    results = []
    idx = 0
    for data in dataset:
        # idx += 1
        # if idx > 2:
        #     break
        result = run(args, llm, evaluate_prompt, parse_prompt, data, dataset)

        results.append(result)

        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)