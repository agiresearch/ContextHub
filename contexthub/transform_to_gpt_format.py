import json 
import argparse 
import os
from prompt import EVALUATE_PROMPTS
import random
import jsonlines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument('--data_category', type=str, default='abstract', help='The data category')
    parser.add_argument('--partial_sample', type=int, default=1, help='The number of partial samples from 1 to 5')
    args = parser.parse_args()

    all_data = []
    for level in range(1, 5):
        for dataset in ['abductive_logic', 'deductive_logic']:
            input_file = f'data/data_level{level}/{dataset}_traincot.json'
            with open(input_file, 'r') as f:
                data = json.load(f)
            all_data.append((data, dataset))
    if args.data_category == 'abstract':
        output_file = f"data/gptdata/{args.data_category}.jsonl"
        gptdata = []
        for data, dataset in all_data:
            for one_template in data:
                for category, contextualized_data in one_template.items():
                    if category == 'abstract':
                        for subcat, subdata in contextualized_data.items():
                            text = subdata['<nl>']
                            text = EVALUATE_PROMPTS[dataset].format(text)
                            cot = subdata['gold_cot']
                            gptdata.append({"messages": [{"role": "system", "content": "You are a rational assistant that carefully answer the question."},{"role": "user", "content": text},{"role": "assistant", "content": cot}]})
        with jsonlines.open(output_file, 'w') as f:
            for d in gptdata:
                f.write(d)
            f.close()
    elif args.data_category == 'contextualized':
        output_file = f"data/gptdata/{args.data_category}.jsonl"
        gpt_data = []
        for data, dataset in all_data:
            for one_template in data:
                for category, contextualized_data in one_template.items():
                    if category in ['question', 'answer', 'abstract']:
                        continue 
                    else:
                        for subcat, subdata in contextualized_data.items():
                            text = subdata['<nl>']
                            text = EVALUATE_PROMPTS[dataset].format(text)
                            cot = subdata['gold_cot']
                            gpt_data.append({"messages": [{"role": "system", "content": "You are a rational assistant that carefully answer the question."},{"role": "user", "content": text},{"role": "assistant", "content": cot}]})
        gpt_data = random.sample(gpt_data, int(len(gpt_data)/11))
        with jsonlines.open(output_file, 'w') as f:
            for d in gpt_data:
                f.write(d)
            f.close()             
    elif args.data_category == 'partial':
        output_file = f"data/gptdata/{args.data_category}{args.partial_sample}.json"
        gpt_data = []
        for data, dataset in all_data:
            for one_template in data:
                for category, contextualized_data in one_template.items():
                    if category in ['question', 'answer']:
                        continue 
                    else:
                        i = 0
                        for subcat, subdata in contextualized_data.items():
                            text = subdata['<nl>']
                            text = EVALUATE_PROMPTS[dataset].format(text)
                            cot = subdata['gold_cot']
                            gpt_data.append({"messages": 
                                            [{"role": "system", "content": "You are a rational assistant that carefully answer the question."},
                                            {"role": "user", "content": text},
                                            {"role": "assistant", "content": cot}]})
                            i += 1
                            if i >= args.partial_sample:
                                break
        with jsonlines.open(output_file, 'w') as f:
            for d in gpt_data:
                f.write(d)
            f.close()  
    elif args.data_category == 'mixed_2':
        mixture = ['culture', 'math']
        output_file = f"data/gptdata/culture_math.jsonl"
        gptdata = []
        for data, dataset in all_data:
            for one_template in data:
                for category, contextualized_data in one_template.items():
                    if category in ['culture and arts', 'mathematics and logic']:
                        for subcat, subdata in contextualized_data.items():
                            text = subdata['<nl>']
                            text = EVALUATE_PROMPTS[dataset].format(text)
                            cot = subdata['gold_cot']
                            gptdata.append({"messages": [{"role": "system", "content": "You are a rational assistant that carefully answer the question."},{"role": "user", "content": text},{"role": "assistant", "content": cot}]})
        gptdata = random.sample(gptdata, int(len(gptdata)/2))
        with jsonlines.open(output_file, 'w') as f:
            for d in gptdata:
                f.write(d)
            f.close()
    else:
        if args.data_category == 'culture':
            data_category = 'culture and arts'
        elif args.data_category ==  'geography':
            data_category = 'geography and places'
        elif args.data_category == 'human':
            data_category = 'human activities'
        elif args.data_category == 'math':
            data_category = 'mathematics and logic'
        elif args.data_category == 'science':
            data_category = 'natural and physical sciences'
        elif args.data_category == 'people':
            data_category = 'people and self'
        elif args.data_category == 'philosophy':
            data_category = 'philosophy and thinking'
        elif args.data_category == 'religion':
            data_category = 'religion and belief systems'
        elif args.data_category == 'society':
            data_category = 'society and social sciences'
        elif args.data_category == 'technology':
            data_category = 'technology and applied sciences'
        elif args.data_category == 'health':
            data_category = 'health and fitness'
        output_file = f"data/gptdata/{args.data_category}.jsonl"
        gptdata = []
        for data, dataset in all_data:
            for one_template in data:
                for category, contextualized_data in one_template.items():
                    if category == data_category:
                        for subcat, subdata in contextualized_data.items():
                            text = subdata['<nl>']
                            text = EVALUATE_PROMPTS[dataset].format(text)
                            cot = subdata['gold_cot']
                            gptdata.append({"messages": [{"role": "system", "content": "You are a rational assistant that carefully answer the question."},{"role": "user", "content": text},{"role": "assistant", "content": cot}]})
        with jsonlines.open(output_file, 'w') as f:
            for d in gptdata:
                f.write(d)
            f.close()
    

