import json, os, sys
import argparse 
from model import load_tokenizer
import matplotlib.pyplot as plt

def check_correctness(output, answer):
    if str(output).lower() == str(answer).lower():
        return 1
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation length')
    parser.add_argument('--data_dir', type=str, default='result', help='Data directory')
    parser.add_argument('--level', type=int, default=1, help='Level')
    parser.add_argument('--model', type=str, default='qwen-0.5', help='Model')
    parser.add_argument('--dataset', type=str, default='deductive_logic', help='deductive_logic, abductive_logic')
    args = parser.parse_args()

    file_name = args.data_dir + f'/data_level{args.level}/{args.model}/{args.dataset}_42.json'
    tokenizer = load_tokenizer(args)

    with open(file_name, 'r') as f:
        data = json.load(f)

    questions = {}
    for template_id, one_template in enumerate(data):
        correct_answer = one_template['answer']
        for k,v in one_template.items():
            if k in ['answer', 'question']:
                continue
            else:
                for subcat, subdata in one_template[k].items():
                    text = subdata['<nl>']
                    text_length = len(tokenizer.tokenize(text))
                    answer = subdata['result']['answer']
                    result = check_correctness(answer, correct_answer)
                    questions[text_length] = result

    sorted_questions = sorted(questions.items(), key=lambda x: x[0])

    xs = []
    ys = []
    for set_length in [d[0] for d in sorted_questions]:
        set_questions = [q for q in sorted_questions if q[0] <= set_length]
        acc = sum([d[1] for d in set_questions])/len(set_questions)
        ys.append(acc)
        xs.append(set_length)

    plt.plot(xs, ys)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    dataset = args.dataset.replace('_', ' ')
    model = args.model.capitalize()
    plt.title(f'{model} on {dataset} at level {args.level}', fontsize=15)
    plt.show()
    plt.savefig(f'figs/correlation_length_{args.model}_{args.dataset}_{args.level}.png')
    


    