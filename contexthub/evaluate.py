import json
from model import call_data_generation_model, call_evaluation_model, call_anthropic_api, load_evaluation_model, call_gpt35_api
import argparse
import re
import os
import time
import random
from utils import set_seed
from prompt import EVALUATE_PROMPTS, PARSE_PROMPTS
from tqdm import tqdm
import openai

def parse_function(result):
    if '<answer>' not in result:
        return 'no answer'
    assert '<answer>' in result
    start = result.index('<answer>')
    end = result.index('</answer>')
    answer = result[start + len('<answer>'):end]
    return answer

def parse_result(parse_prompt, question, result):
    answer = 'no answer'
    prompt = parse_prompt.format(question, result)
    try:
        answer = re.findall(r'<answer>(.*?)</answer>', result, re.DOTALL)[-1]
    except:
        i = 0
        while i <= 5:
            i += 1
            try:
                #parsed_result = call_anthropic_api(prompt)
                parsed_result = call_gpt35_api(prompt)
                answer = parse_function(parsed_result)
                return {'answer':answer, 'reasoning':result}
            except KeyboardInterrupt:
                raise
            except openai.BadRequestError as e:
                print(f"BadRequestError: {e}")
                return {'answer':'no answer', 'reasoning':result}
            except OSError as e:
                print(f"OSError: {e}")
                return {'answer':'no answer', 'reasoning':result}
            # except HTTPStatusError as e:
            #     return {'answer':'no answer', 'reasoning':result}
            except:
                time.sleep(0.1)
    return {'answer':answer, 'reasoning':result}

def collect_answer(data, results):
    j = 0
    for domain in data.keys():
        # if domain != 'abstract':
        #     data[domain] = partial_result[domain]
        # else:
        #     for subcat, subdata in data[domain].items():
        #         data[domain][subcat]['result'] = results[j]
        #         j += 1
        if type(data[domain]) is list:
            for i in range(len(data[domain])):
                data[domain][i]['result'] = results[j]
                j += 1
        if type(data[domain]) is dict:
            for subcat, subdata in data[domain].items():
                data[domain][subcat]['result'] = results[j]
                j += 1
    return data

def run(args, llm, evaluate_prompt, parse_prompt, data):
    prompts = []
    questions = []
    # contextualized result
    for domain in data.keys():
        # if domain == 'abstract':
        #     for subcat, subdata in data[domain].items():
        #         text = subdata['<nl>']
        #         one_prompt = evaluate_prompt.format(text)
        #         prompts.append(one_prompt)
        #         questions.append(text.split('\n')[-1])
        if type(data[domain]) is list:
            for i in range(len(data[domain])):
                text = data[domain][i]['<nl>']
                one_prompt = evaluate_prompt.format(text)
                prompts.append(one_prompt)
                questions.append(text.split('\n')[-1])
        # create data
        if type(data[domain]) is dict:
            for subcat, subdata in data[domain].items():
                text = subdata['<nl>']
                one_prompt = evaluate_prompt.format(text)
                prompts.append(one_prompt)
                questions.append(text.split('\n')[-1])
    # run model
    if 'gpt' not in args.model:
        results = call_evaluation_model(model=args.model, text_prompt=prompts, llm=llm)
    else:
        results = []
        for prompt in tqdm(prompts, desc="Generating answers"):
            r = call_data_generation_model(args.model,prompt)
            results.append(r)
    new_results = []
    for q, r in tqdm(zip(questions, results), desc="Parsing generated answers"):
        new_results.append(parse_result(parse_prompt, q, r))
    # collect result
    data = collect_answer(data, new_results)

    return data

def parse_only(parse_prompt, data):
    results = []
    questions = []
    # contextualized result
    for domain in data.keys():
        if type(data[domain]) is list:
            for i in range(len(data[domain])):
                text = data[domain][i]['<nl>']
                results.append(data[domain][i]['result']['reasoning'])
                questions.append(text.split('\n')[-1])
        # create data
        if type(data[domain]) is dict:
            for subcat, subdata in data[domain].items():
                text = subdata['<nl>']
                results.append(data[domain][subcat]['result']['reasoning'])
                questions.append(text.split('\n')[-1])
    # parse result from reasoning
    new_results = []
    for q, r in tqdm(zip(questions, results), desc="Parsing generated answers"):
        new_results.append(parse_result(parse_prompt, q, r))
    # collect result
    data = collect_answer(data, new_results)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tulu-13", help="The model to evaluate.")
    parser.add_argument("--trained_model_name", type=str, default=None, help="The model to evaluate: such as, pretrained_models/whole_model/qwen-0.5_abstract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--level", type=int, default=1, help="The difficulty level")
    parser.add_argument("--dataset", type=str, default="abductive_logic", help="The dataset")
    parser.add_argument("--proficiency", type=str, default=None, help="The proficiency level")
    parser.add_argument("--with_rules", action="store_true", help="with basic rules in prompt")
    parser.add_argument('--parse_only', action='store_true', help='parse only')
    args = parser.parse_args()

    set_seed(args.seed)
    
    # if args.with_rules:
    #     evaluate_prompt = EVALUATE_PROMPTS_WITH_BASIC_RULES[args.dataset]
    # else:
    evaluate_prompt = EVALUATE_PROMPTS[args.dataset]

    parse_prompt = PARSE_PROMPTS[args.dataset]

    data_path = f'data/data_level{args.level}/'

    if args.trained_model_name is not None:
        if 'gpt' not in args.model:
            data_strategy = args.trained_model_name.split('/')[-1].split('_')[1]
            result_folder = f'result/finetuned/data_level{args.level}/{data_strategy}-{args.model}/'
        else:
            result_folder = f'result/finetuned/data_level{args.level}/{args.model}/'
    else:
        result_folder = f'result/data_level{args.level}/{args.model}/'
    
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

    if args.parse_only:
        with open(result_path, 'r') as f:
            dataset = json.load(f)

        results = []
        idx = 0
        for data in tqdm(dataset):
            # idx += 1
            # if idx < 16:
            #     continue
            result = parse_only(parse_prompt, data)

            results.append(result)

            with open(result_path, 'w') as f:
                json.dump(results, f, indent=4)

    else:
        with open(data_path, 'r') as f:
            dataset = json.load(f)

        if 'gpt' not in args.model:
            llm = load_evaluation_model(model=args.model,model_name=args.trained_model_name)
        else:
            llm = None

        results = []
        idx = 0
        for data in dataset:
            result = run(args, llm, evaluate_prompt, parse_prompt, data)

            results.append(result)

            with open(result_path, 'w') as f:
                json.dump(results, f, indent=4)
