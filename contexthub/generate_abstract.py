import json 
import argparse 
import string
import re
import random

if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data', help='Path to the input json file')
    parser.add_argument('--dataset', type=str, default='deductive_logic')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--number_of_instantiation', type=int, default=5)
    args = parser.parse_args()

    # load data
    data_path = f'{args.data}/data_level{args.level}/{args.dataset}.json'
    print('='*50)
    print(f'Loading data from {data_path}')
    print('='*50)
    with open(data_path, 'r') as f:
        data = json.load(f)

    # printatble
    chars = list(string.ascii_lowercase)
    number_of_chars = len(chars)

    # add abstract domain
    for template_id, template in enumerate(data):
        question = template['question'][0]['<nl>']
        symbols = list(set(re.findall(r'aa\w', question)))
        data[template_id]['abstract'] = {}
        for i in range(args.number_of_instantiation):
            data[template_id]['abstract'][i] = {}
            if i != 0:
                symbol_replacement = {}
                for symbol in symbols:
                    length = random.randint(2, 5)
                    symbol_replacement[symbol] = ''.join([chars[random.randint(0, number_of_chars-1)] for _ in range(length)])
                replace_question = question
                for k,v in symbol_replacement.items():
                    data[template_id]['abstract'][i]['<'+k+'>'] = v
                    replace_question = replace_question.replace(k, v)
                data[template_id]['abstract'][i]['<nl>'] = replace_question
            else:
                symbol_replacement = {}
                for symbol in symbols:
                    symbol_replacement[symbol] = symbol
                for k,v in symbol_replacement.items():
                    data[template_id]['abstract'][i]['<'+k+'>'] = v
                data[template_id]['abstract'][i]['<nl>'] = question

    # save data
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)
