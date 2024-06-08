import json 
import argparse
import os
import time

def clean_number(n):
    n = n.replace('$', '').replace(',', '')
    return n

def check_correctness(output, answer):
    try: 
        output = clean_number(output)
        if output == answer:
            #print('correct', (output, answer))
            return True
        output = float(output)
        if output == answer:
            #print('correct', (output, answer))
            return True
    except:
        if output in ['true', 'false', 'n/a']:
            if output == answer or bool(output) == answer:
                #print('correct', (output, answer))
                return True
    #print('wrong', (output, answer))
    return False

def present_result(dataset):
    for did, data in enumerate(dataset):
        answer = data['answer']
        print('************************')
        print('CORRECT ANSWER:', answer)
        for domain in data:
            if type(data[domain]) is dict:
                for subcat, subdata in data[domain].items():
                    a = data[domain][subcat]['result']['answer']
                    if not check_correctness(a.lower(), str(answer).lower()):
                        print(domain)
                        print(data[domain][subcat]['<nl>'])
                        print(data[domain][subcat]['result'])
                        print('-------------')
                        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen-0.5')
    parser.add_argument('--dataset', type=str, default='deductive_logic')
    parser.add_argument('--proficiency', type=str, default=None)
    parser.add_argument('--group', action='store_true')
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()

    with open('result/data_level{}/{}/{}_42.json'.format(args.level, args.model, args.dataset), 'r') as f:
        data = json.load(f)

    present_result(data)
   