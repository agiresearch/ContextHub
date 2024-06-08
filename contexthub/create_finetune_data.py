import json 
import argparse
import re
import time

def load_data(args):
    with open(f'data/data_level{args.level}/{args.dataset}.json', 'r') as f:
        dataset = json.load(f)

    with open(f'data/data_level{args.level}/{args.dataset}_cot.json', 'r') as f:
        cots = json.load(f)

    paired_data_cot = {}
    for dataset_id, logic_template in enumerate(dataset):
        for cot_id, cot in enumerate(cots):
            if logic_template['question'][0]['<nl>'] == cot['descriptions']:
                paired_data_cot[dataset_id] = cot_id

    return dataset, cots, paired_data_cot


def create_finetune_data(datas, cots, paired_data_cot):
    def naturalize(text):
        text = text.replace('OR', 'or')
        text = text.replace('AND', 'and')
        text = text.replace('NOT', 'not')
        text = text.replace('->', 'logically implies')
        text = text.replace('=', 'whose corresponding truth value is')
        return text
    for dataset_id, logic_template in enumerate(datas):
        abstract_cot = cots[paired_data_cot[dataset_id]]['inferences'].replace('<<<', '<answer>').replace('>>>', '</answer>')
        symbols_in_abstract_cot = list(set(re.findall(r'aa\w', abstract_cot)))
        for domain in logic_template.keys():
            if domain == 'question' or domain == 'answer':
                continue
            for subcat, subdata in logic_template[domain].items():
                replace_abstract_cot = abstract_cot
                for symbol in symbols_in_abstract_cot:
                    symbol_key = '<'+symbol+'>'
                    to_replace = subdata[symbol_key]
                    if to_replace.endswith('.'):
                        to_replace = to_replace[:-1]
                    if len(to_replace.split()) > 1:
                        to_replace = '"'+to_replace+'"'
                    replace_abstract_cot = replace_abstract_cot.replace(symbol, to_replace)
                replace_abstract_cot = naturalize(replace_abstract_cot)
                datas[dataset_id][domain][subcat]['gold_cot'] = replace_abstract_cot

    return datas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tulu-13", help="The model to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--level", type=int, default=2, help="The difficulty level")
    parser.add_argument("--dataset", type=str, default="deductive_logic", help="The dataset")
    parser.add_argument("--proficiency", type=str, default=None, help="The proficiency level")
    parser.add_argument("--with_rules", action="store_true", help="with basic rules in prompt")
    parser.add_argument('--parse_only', action='store_true', help='parse only')
    args = parser.parse_args()


    dataset, cots, paired_data_cot = load_data(args)

    data_with_cot = create_finetune_data(dataset, cots, paired_data_cot)

    with open(f'data/data_level{args.level}/{args.dataset}_traincot.json', 'w') as f:
        json.dump(data_with_cot, f, indent=4)