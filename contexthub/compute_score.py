import json 
import numpy as np
import argparse
import statistics
from scipy.stats import ks_2samp
import os
from model import load_tokenizer
from utils import Logger

def compute_PRF(result):
    true_TP = len([a for a in result if check_correctness(a[0], a[1]) and a[0] == 'true'])
    true_FP = len([a for a in result if a[0] == 'true' and a[1] != 'true'])
    true_TN = len([a for a in result if check_correctness(a[0], a[1]) and a[1] != 'true'])
    true_FN = len([a for a in result if a[0] != 'true' and a[1] == 'true'])
    if (true_TP + true_FP) != 0:
        true_precision = true_TP/(true_TP + true_FP)
    else:
        true_precision = None
    if (true_TP + true_FN) != 0:
        true_recall = true_TP/(true_TP + true_FN)
    else:
        true_recall = None
    true_f1 = None
    if true_recall is not None:
        if true_precision is not None:
            if true_precision + true_recall != 0:
                true_f1 = 2*true_precision*true_recall/(true_precision + true_recall)
            else:
                true_f1 = 0
        else:
            true_f1 = 0
    
    false_TP = len([a for a in result if check_correctness(a[0], a[1]) and a[0] == 'false'])
    false_FP = len([a for a in result if a[0] == 'false' and a[1] != 'false'])
    false_TN = len([a for a in result if check_correctness(a[0], a[1]) and a[1] != 'false'])
    false_FN = len([a for a in result if a[0] != 'false' and a[1] == 'false'])

    if (false_TP + false_FP) != 0:
        false_precision = false_TP/(false_TP + false_FP)
    else:
        false_precision = None
    if (false_TP + false_FN) != 0:
        false_recall = false_TP/(false_TP + false_FN)
    else:
        false_recall = None
    false_f1 = None
    if false_recall is not None:
        if false_precision is not None:
            if false_precision + false_recall != 0:
                false_f1 = 2*false_precision*false_recall/(false_precision + false_recall)
            else:
                false_f1 = 0
        else:
            false_f1 = 0

    na_TP = len([a for a in result if check_correctness(a[0], a[1]) and a[0] == 'n/a'])
    na_FP = len([a for a in result if a[0] == 'n/a' and a[1] != 'n/a'])
    na_TN = len([a for a in result if check_correctness(a[0], a[1]) and a[1] != 'n/a'])
    na_FN = len([a for a in result if a[0] != 'n/a' and a[1] == 'n/a'])

    if (na_TP + na_FP) != 0:
        na_precision = na_TP/(na_TP + na_FP)
    else:
        na_precision = None
    if (na_TP + na_FN) != 0:
        na_recall = na_TP/(na_TP + na_FN)
    else:
        na_recall = None
    na_f1 = None
    if na_recall is not None:
        if na_precision is not None:
            if na_precision + na_recall != 0:
                na_f1 = 2*na_precision*na_recall/(na_precision + na_recall)
            else:
                na_f1 = 0
        else:
            na_f1 = 0

    average_precision = np.mean([a for a in [true_precision, false_precision, na_precision] if a is not None])
    average_recall = np.mean([a for a in [true_recall, false_recall, na_recall] if a is not None])
    average_f1 = np.mean([a for a in [true_f1, false_f1, na_f1] if a is not None])
    return average_precision, average_recall, average_f1, {
        'true': [true_precision, true_recall, true_f1], 
        'false': [false_precision, false_recall, false_f1], 
        'n/a': [na_precision, na_recall, na_f1]
        }

def compute_statistics(args, correct, distribution, domain_sentences):
    standard_deviation = statistics.stdev([v[0]/v[1] for k,v in correct.items() if k!= 'question'])
    print('standard deviation: ', standard_deviation)

    max_accuracy = np.max([v[0]/v[1] for k,v in correct.items() if k!= 'question'])
    min_accuracy = np.min([v[0]/v[1] for k,v in correct.items() if k!= 'question'])

    max_domain = [k for k,v in correct.items() if v[0]/v[1] == max_accuracy and k != 'question']
    min_domain = [k for k,v in correct.items() if v[0]/v[1] == min_accuracy and k != 'question']
    
    max_distribution = distribution[max_domain[0]]
    min_distribution = distribution[min_domain[0]]
    print('*******Domain difference significance:*******')
    print(ks_2samp(max_distribution, min_distribution))

    if args.print_length:
        tokenizer = load_tokenizer(args)
        for i in range(len(max_domain)):
            for j in range(len(min_domain)):
                max_domain_sentences = domain_sentences[max_domain[i]]
                min_domain_sentences = domain_sentences[min_domain[j]]
                max_domain_sentences_length = [len(tokenizer(max_domain_sentence)['input_ids']) for max_domain_sentence in max_domain_sentences]
                min_domain_sentences_length = [len(tokenizer(min_domain_sentence)['input_ids']) for min_domain_sentence in min_domain_sentences]
                print(f'-----Length statistics on domains with significant performance difference, between {i} and {j}:-----')
                print('mean length of max domain: ', np.mean(max_domain_sentences_length))
                print('mean length of min domain: ', np.mean(min_domain_sentences_length))
                print(ks_2samp(max_domain_sentences_length, min_domain_sentences_length))

        print('\n')
        print('\n')
        print('*******Performance statistics on domains with significant length difference*******')
        domain_sentences_length = {}
        for k,v in domain_sentences.items():
            domain_sentences_length[k] = [len(tokenizer(domain_sentence)['input_ids']) for domain_sentence in v]
        for domain in domain_sentences_length.keys():
            for another_domain in domain_sentences_length.keys():
                if domain != another_domain:
                    sta = ks_2samp(domain_sentences_length[domain], domain_sentences_length[another_domain])
                    if sta.pvalue < 0.05:
                        print(f'---between {domain} and {another_domain}:---')
                        print(ks_2samp(domain_sentences_length[domain], domain_sentences_length[another_domain]))
                        domain_distribution = distribution[domain]
                        another_domain_distribution = distribution[another_domain]
                        print(ks_2samp(domain_distribution, another_domain_distribution))


def present_result(correct):
    for k,v in correct.items():
        print('Domain: ', k)
        print('Correct out of total: ', correct[k][0]/correct[k][1])
        print('*'*10)
    print('=====================')
    print('abstract standard logic accuracy: ', correct['question'][0]/correct['question'][1])
    print('total contextualized accuracy: ', np.sum([v[0] for k,v in correct.items() if k!= 'question'])/np.sum([v[1] for k,v in correct.items() if k!= 'question']))
    max_accuracy = np.max([v[0]/v[1] for k,v in correct.items() if k!= 'question'])
    min_accuracy = np.min([v[0]/v[1] for k,v in correct.items() if k!= 'question'])
    max_domain = ', '.join([k for k,v in correct.items() if v[0]/v[1] == max_accuracy and k != 'question'])
    min_domain = ', '.join([k for k,v in correct.items() if v[0]/v[1] == min_accuracy and k != 'question'])
    print(f'max contextualized accuracy in domain {max_domain}: {max_accuracy}')
    print(f'min contextualized accuracy in domain {min_domain}: {min_accuracy}')
    better_domains = [k for k,v in correct.items() if v[0]/v[1] >= correct['question'][0]/correct['question'][1] and k != 'question']
    worse_domains = [k for k,v in correct.items() if v[0]/v[1] < correct['question'][0]/correct['question'][1] and k != 'question']
    print(f'better domains: {better_domains}')
    print(f'worse domains: {worse_domains}')


def present_PRF(actual_result_collection, logger):
    print('=====================')
    f1s = 0
    for k,v in actual_result_collection.items():
        logger.log(f'Domain: {k}')
        average_precision, average_recall, average_f1, PRF = compute_PRF(v)
        logger.log(f'average precision: {average_precision}')
        logger.log(f'average recall: {average_recall}')
        logger.log(f'faverage f1: {average_f1}')
        if k not in ['question', 'abstract']:
            f1s += average_f1
        #print(PRF)
        logger.log('*'*10)
    logger.log('Total data f1: ' + str(f1s/(len(actual_result_collection)-2)))
    

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


def present_scores(args, dataset, logger):
    correct = {domain:[0,0] for domain in list(dataset[0].keys()) if domain != 'answer'}
    distribution = {domain:[] for domain in list(dataset[0].keys()) if domain != 'answer'}
    actual_result_collection = {domain:[] for domain in list(dataset[0].keys()) if domain != 'answer'}
    general_correct_sentences = []
    general_incorrect_sentences = []
    domain_sentences = {}
    for did, data in enumerate(dataset):
        answer = data['answer']
        for domain in data:
            if type(data[domain]) is list:
                for i in range(len(data[domain])):
                    a = data[domain][i]['result']['answer']
                    if check_correctness(a.lower(), str(answer).lower()):
                        correct[domain][0] += 1
                        distribution[domain].append(1)
                    else:
                        distribution[domain].append(0)
                    actual_result_collection[domain].append((a.lower(), str(answer).lower()))
                    correct[domain][1] += 1
            if type(data[domain]) is dict:
                if domain not in domain_sentences:
                    domain_sentences[domain] = []
                for subcat, subdata in data[domain].items():
                    a = data[domain][subcat]['result']['answer']
                    actual_result_collection[domain].append((a.lower(), str(answer).lower()))
                    #print(a)
                    if check_correctness(a.lower(), str(answer).lower()):
                        correct[domain][0] += 1
                        distribution[domain].append(1)
                        general_correct_sentences.append(data[domain][subcat]['<nl>'])
                    else:
                        distribution[domain].append(0)
                        general_incorrect_sentences.append(data[domain][subcat]['<nl>'])
                    correct[domain][1] += 1
                    domain_sentences[domain].append(data[domain][subcat]['<nl>'])

    with open(f'result/distribution/{args.model}_level{args.level}_{args.dataset}.json', 'w') as f:
        json.dump(distribution, f, indent=4)
    #present_result(correct)
    #compute_statistics(args, correct, distribution, domain_sentences)
    present_PRF(actual_result_collection, logger)
    if args.print_length:
        length_analysis(args, general_correct_sentences, general_incorrect_sentences)
    
def length_analysis(args, correct_sentences, incorrect_sentences):
    tokenizer = load_tokenizer(args)
    correct_lengths = [len(tokenizer(correct_sentence)['input_ids']) for correct_sentence in correct_sentences]
    incorrect_lengths = [len(tokenizer(incorrect_sentence)['input_ids']) for incorrect_sentence in incorrect_sentences]
    print('\n')
    print('\n')
    print("*******General length analysis*******")
    print('correct lengths: ', np.mean(correct_lengths), np.std(correct_lengths))
    print('incorrect lengths: ', np.mean(incorrect_lengths), np.std(incorrect_lengths))
    print(ks_2samp(correct_lengths, incorrect_lengths)) 

# multiple random seeds
def compute_group_statistics(group_correct, group_distribution):
    print(group_correct)
    domain_group_result = {}
    for k,v in group_correct[0].items():
        domain_group_result[k] = [one_correct[k][0]/one_correct[k][1] for one_correct in group_correct]
        print('Domain: ', k)
        print('Mean on performance: ', statistics.mean(domain_group_result[k]))
        print('Variance on performance: ', statistics.variance(domain_group_result[k]))
        print('*'*10)
    print('=====================')

    standard_logic_accuracy = np.mean(domain_group_result['question'])
    max_accuracy = np.max([np.mean(v) for k,v in domain_group_result.items() if k!= 'question'])
    min_accuracy = np.min([np.mean(v) for k,v in domain_group_result.items() if k!= 'question'])
    max_domain = [k for k,v in domain_group_result.items() if np.mean(v) == max_accuracy and k != 'question']
    min_domain = [k for k,v in domain_group_result.items() if np.mean(v) == min_accuracy and k != 'question']
    print(f'non-contextualized standard logic accuracy: {standard_logic_accuracy}')
    print(f'max contextualized accuracy in domain {max_domain}: {max_accuracy}')
    print(f'min contextualized accuracy in domain {min_domain}: {min_accuracy}')
    better_domains = [k for k,v in domain_group_result.items() if np.mean(v) > standard_logic_accuracy]
    worse_domains = [k for k,v in domain_group_result.items() if np.mean(v) < standard_logic_accuracy]
    print(f'better domains: {better_domains}')
    print(f'worse domains: {worse_domains}')
    print('Variance on performance: ', ks_2samp(domain_group_result[max_domain[0]], domain_group_result[min_domain[0]]))

    domain_group_distribution = {}
    for k,v in group_distribution[0].items():
        domain_group_distribution[k] = [np.mean([group_distribution[i][k][j] for i in range(len(group_distribution))]) for j in range(len(group_distribution[0][k]))]
    print('Variance on distribution: ', ks_2samp(domain_group_distribution[max_domain[0]], domain_group_distribution[min_domain[0]]))

def present_group_scores(all_data):
    group_correct = []
    group_distribution = []
    for dataset in all_data:
        correct = {domain:[0,0] for domain in list(dataset[0].keys()) if domain != 'answer'}
        distribution = {domain:[] for domain in list(dataset[0].keys()) if domain != 'answer'}
        for did, data in enumerate(dataset):
            answer = data['answer']
            for domain in data:
                if type(data[domain]) is list:
                    for i in range(len(data[domain])):
                        a = data[domain][i]['result']['answer']
                        if check_correctness(a.lower(), str(answer).lower()):
                            correct[domain][0] += 1
                            distribution[domain].append(1)
                        else:
                            distribution[domain].append(0)
                        correct[domain][1] += 1
                if type(data[domain]) is dict:
                    for subcat, subdata in data[domain].items():
                        a = data[domain][subcat]['result']['answer']
                        if check_correctness(a.lower(), str(answer).lower()):
                            correct[domain][0] += 1
                            distribution[domain].append(1)
                        else:
                            distribution[domain].append(0)
                        correct[domain][1] += 1
        group_correct.append(correct)
        group_distribution.append(distribution)

    compute_group_statistics(group_correct, group_distribution)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen-0.5')
    parser.add_argument("--trained_model_name", type=str, default=None, help="The model to evaluate: such as, pretrained_models/whole_model/qwen-0.5_abstract")
    parser.add_argument('--dataset', type=str, default='deductive_logic')
    parser.add_argument('--proficiency', type=str, default=None)
    parser.add_argument('--group', action='store_true')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--print_length', action='store_true')
    args = parser.parse_args()

    if args.trained_model_name is not None:
        data_strategy = args.trained_model_name.split('/')[-1].split('_')[1]
        logger = Logger(f'result_log/finetuned/{args.dataset}_{args.level}_{args.model}_{data_strategy}.log', on=False)
    else:
        logger = Logger(f'result_log/{args.dataset}_{args.level}_{args.model}.log', on=False)

    if not args.group:
        if args.trained_model_name is None:
            with open('result/data_level{}/{}/{}_42.json'.format(args.level, args.model, args.dataset), 'r') as f:
                data = json.load(f)
        else:
            if 'gpt' not in args.model:
                data_strategy = args.trained_model_name.split('/')[-1].split('_')[1]
                with open('result/finetuned/data_level{}/{}-{}/{}_42.json'.format(args.level, data_strategy, args.model, args.dataset), 'r') as f:
                    data = json.load(f)
            else:
                with open('result/finetuned/data_level{}/{}/{}_42.json'.format(args.level, args.model, args.dataset), 'r') as f:
                    data = json.load(f)
        present_scores(args, data, logger)
    else:
        allfiles = ['result/data_level{}/{}/'.format(args.level, args.model) + f for f in os.listdir('result/data_level1/{}/'.format(args.model))]
        if args.proficiency:
            data_prefix = 'result/data_level{}/{}/{}_{}'.format(args.level, args.model, args.proficiency, args.dataset)
            data_dirs = [f for f in allfiles if f.startswith(data_prefix)]
        else:
            data_prefix = 'result/data_level{}/{}/{}'.format(args.level, args.model, args.dataset)
            data_dirs = [f for f in allfiles if f.startswith(data_prefix)]
        all_data = []
        for dr in data_dirs:
                with open(dr, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
            
        present_group_scores(all_data)
