# pip install datasets
from datasets import load_dataset
import argparse
import openai
import time
import re
import json

from prompt import PARSE_PROMPTS
from model import call_gpt35_api, call_evaluation_model, load_evaluation_model


def generate_evaluation_prompt(dataset_name):
    prompts = {
        "folio": "Given the natural language premises and their first order logic (FOL) representations, determine if the natural language conclusion and its FOL representation are true, false, or uncertain.\n\nNatural language premises: {premises}\n\nFOL premises: {premises-FOL}\n\nNatural language conclusion: {conclusion}\n\nFOL conclusion: {conclusion-FOL}\n\nPlease think step by step and respond with '<answer>true</answer>', '<answer>false</answer>', or '<answer>uncertain</answer>'.",

        "ruletaker": "Using the rules and facts provided, determine whether the statement is entailed or not.\n\nRules & Facts: {context}\n\nStatement: {question}\n\nPlease think step by step and Respond with '<answer>entailment</answer>' or '<answer>not entailment</answer>'.",

        "logicnli": "Decide if the logical hypothesis is contradict/self-contradict/neutral/entailment based on the premises.\n\nPremises: {premise}\n\nHypothesis: {hypothesis}\n\nPlease think step by step and respond with '<answer>contradiction</answer>', '<answer>self-contradiction</answer>', '<answer>neutral</answer>', or '<answer>entailment</answer>'.",

        "sst2": "Analyze the sentiment of this sentence: {sentence}\n\nPlease think step by step and respond with '<answer>positive</answer>' or '<answer>negative</answer>'.",

        "cola": "Assess the following sentence and determine if it is grammatically correct:\n\nSentence: {sentence}\n\nPlease think step by step and respond with '<answer>Acceptable</answer>' or '<answer>Unacceptable</answer>'.",

        "qqp": "Determine if the given pair of statements can be considered the same.\n\nQuestion1: {question1}\nQuestion2: {question2}\n\nPlease think step by step and respond with '<answer>equivalent</answer>' or '<answer>not_equivalent</answer>'.",

        "wnli": "Does the relationship between the given two sentences represent entailment or not_entailment?\n\nSentence1: {sentence1}\nSentence2: {sentence2}\n\nPlease think step by step and respond with '<answer>entailment</answer>' or '<answer>not_entailment</answer>'."
    }
    
    if dataset_name in prompts:
        return prompts[dataset_name]
    else:
        raise ValueError(f"No prompt available for: {dataset_name}")


def load_dataset_by_name(dataset_name):
    dataset = []
    if dataset_name == "folio":
        # return load_dataset("tasksource/folio")["validation"]
        data = load_dataset("tasksource/folio")["validation"]
        for d in data:
            dataset.append(d)

    elif dataset_name == "ruletaker":
        # return load_dataset("tasksource/ruletaker")["test"]
        data = load_dataset("tasksource/ruletaker")["test"]
        for d in data:
            dataset.append(d)

    elif dataset_name == "logicnli":
        # return load_dataset("tasksource/LogicNLI")["test"]
        data = load_dataset("tasksource/LogicNLI")["test"]
        for d in data:
            dataset.append(d)

    elif dataset_name == "sst2":
        # return load_dataset("glue", 'sst2')["test"]
        data = load_dataset("glue", 'sst2')["test"]
        for d in data:
            if d["label"] == -1:
                dataset.append({"sentence": d["sentence"], "label": "Negative"})
            elif d["label"] == 1:
                dataset.append({"sentence": d["sentence"], "label": "Positive"})
            else:
                raise ValueError(f"Unknown label: {d['label']}")
    elif dataset_name == "cola":
        # return load_dataset("glue", 'cola')["test"]
        data = load_dataset("glue", 'cola')["test"]
        for d in data:
            if d["label"] == -1:
                dataset.append({"sentence": d["sentence"], "label": "Unacceptable"})
            elif d["label"] == 1:
                dataset.append({"sentence": d["sentence"], "label": "Acceptable"})
            else:
                raise ValueError(f"Unknown label: {d['label']}")
    elif dataset_name == "qqp":
        # return load_dataset("glue", 'qqp')["test"]
        data = load_dataset("glue", 'qqp')["test"]
        for d in data:
            if d["label"] == -1:
                dataset.append({"question1": d["question1"], "question2": d["question2"], "label": "not_equivalent"})
            elif d["label"] == 1:
                dataset.append({"question1": d["question1"], "question2": d["question2"], "label": "equivalent"})
            else:
                raise ValueError(f"Unknown label: {d['label']}")
    elif dataset_name == "wnli":
        # return load_dataset("glue", 'wnli')["test"]
        data = load_dataset("glue", 'wnli')["test"]
        for d in data:
            if d["label"] == -1:
                dataset.append({"sentence1": d["sentence1"], "sentence2": d["sentence2"], "label": "not_entailment"})
            elif d["label"] == 1:
                dataset.append({"sentence1": d["sentence1"], "sentence2": d["sentence2"], "label": "entailment"})
            else:
                raise ValueError(f"Unknown label: {d['label']}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def parse_function(result):
    if '<answer>' not in result:
        return 'no answer'
    assert '<answer>' in result
    start = result.index('<answer>')
    end = result.index('</answer>')
    answer = result[start + len('<answer>'):end]
    return answer


def parse_result(parse_prompt, question, result):
    answer = "no answer"
    if '<no_answer>' in result:
        return {'answer':answer, 'reasoning':result}
    prompt = parse_prompt.format(question, result)
    try:
        answer = re.findall(r'<answer>(.*?)</answer>', result, re.DOTALL)[-1]
    except:
        retry_cnt = 5
        while retry_cnt > 0:
            retry_cnt -= 1
            try:
                #parsed_result = call_anthropic_api(prompt)
                parsed_result = call_gpt35_api(prompt)
                print("in parse_result function, parsed_result is")
                print(parsed_result)
                answer = parse_function(parsed_result)
                print("in parse_result function, answer is")
                print(answer)
                break
            except KeyboardInterrupt:
                raise
            except openai.BadRequestError as e:
                print(f"BadRequestError: {e}")
                return {'answer':'no answer', 'reasoning':result}
            except:
                time.sleep(0.1)
    return {'answer':answer, 'reasoning':result}

def run(dataset_name, llm):
    prompt = generate_evaluation_prompt(dataset_name)
    dataset = load_dataset_by_name(dataset_name)
    parse_prompt = PARSE_PROMPTS[dataset_name]
    
    if len(dataset) > 1000:
        dataset = dataset[:1000]

    input_texts = []
    results = []
    for data in dataset:
        gt = data['label']
        input_text = prompt.format(**data)
        input_texts.append(input_text)

        results.append({
            "input": input_text,
            "gt": gt,
        })

    preds = call_evaluation_model(model=args.model, text_prompt=input_texts, llm=llm)

    for i, p in enumerate(preds):
        parsed_pred = parse_result(parse_prompt, input_texts[i], p)
        results[i]["prediction"] = parsed_pred["answer"]
        results[i]["reasoning"] = parsed_pred["reasoning"]

    correct = 0
    total = 0
    for d in results:
        if d["answer"].lower() == d['gt'].lower():
            correct += 1
    
        total += 1


    return results, correct/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-4", help="The model to evaluate.")
    parser.add_argument("--dataset", type=str, default="wnli", help="The dataset")
    parser.add_argument("--trained_model_name", type=str, default=None, help="The model to evaluate: such as, pretrained_models/whole_model/qwen-0.5_abstract")
    args = parser.parse_args()

    dataset = load_dataset_by_name(args.dataset)
    # from openai import OpenAI
    # import os
    # # os.environ["OPENAI_API_KEY"] = "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"
    # os.environ["OPENAI_API_KEY"] = "sk-proj-ipNOpLetNR6UrjhI1xG2T3BlbkFJXgGV2DOEOOfQHhVXVDMJ"
    # client = OpenAI()

    # contexutualized_model_id = 'ft:gpt-3.5-turbo-0125:rutgers-university::9SgzFY26'
    # abstract_model_id = 'ft:gpt-3.5-turbo-0125:rutgers-university::9Swm3OAg'
    # partial1_model_id = 'ft:gpt-3.5-turbo-0125:rutgers-university::9SwvJmQg'
    # partial2_model_id = 'ft:gpt-3.5-turbo-0125:rutgers-university::9Sxm1qKh'
    # partial3_model_id = 'ft:gpt-3.5-turbo-0125:rutgers-university::9SzKkDED'
    # partial4_model_id = 'ft:gpt-3.5-turbo-0125:rutgers-university::9SzpHo7I'
    # partial5_model_id = 'ft:gpt-3.5-turbo-0125:rutgers-university::9T0B0SCA'

    llm = load_evaluation_model(model=args.model,model_name=args.trained_model_name)
        
    # for dataset_name in ["ruletaker", "logicnli", "sst2", "cola", "qqp", "wnli"]:
    for dataset_name in ["ruletaker", "logicnli", "wnli"]:
        if args.trained_model_name:
            model_id = args.trained_model_name.split('/')[-1]
        else:
            model_id = args.model

        print('='*50)
        print("results_ft/{}_{}_accuracy.json".format(model_id, dataset_name))
        print('='*50)

        results, accuracy = run(dataset_name, llm)

        print('Model: {}, Dataset: {}, Accuracy: {:.2f}'.format(model_id, dataset_name, accuracy))

        with open("results_ft/{}_{}_{:.2f}.json".format(model_id, dataset_name, accuracy), "w") as f:
            json.dump(results, f, indent=4)

    # # for dataset_name in ["folio"]:

    #     for model_id in ["gpt-3.5-turbo-0125", contexutualized_model_id, abstract_model_id, partial1_model_id, partial2_model_id, partial3_model_id, partial4_model_id, partial5_model_id]:
    #     # for model_id in []:

    #         results = []
    #         correct = 0
    #         total = 0

    #         prompt = generate_evaluation_prompt(dataset_name)
    #         dataset = load_dataset_by_name(dataset_name)
    #         parse_prompt = PARSE_PROMPTS[dataset_name]
            
    #         if len(dataset) > 1000:
    #             dataset = dataset[:1000]

    #         print(len(dataset))
    #         print(dataset_name)
    #         # continue
    #         for data in dataset:
    #             input_text = prompt.format(**data)
    #             print("========================================")
    #             print("input_text")
    #             print(input_text)

    #             completion = client.chat.completions.create(
    #                 model=model_id,
    #                 messages=[
    #                     {"role": "system", "content": "You are a rational assistant that carefully answer the question."},
    #                     {"role": "user", "content": input_text}
    #                 ]
    #             )
                
    #             pred = completion.choices[0].message.content
    #             print("========================================")
    #             print("pred")
    #             print(pred)
                
    #             gt = data["label"]

    #             parsed_pred = parse_result(parse_prompt, input_text, pred)
    #             print("========================================")
    #             print("parsed_pred")
    #             print(parsed_pred)

    #             results.append({
    #                 "input": input_text,
    #                 "prediction": parsed_pred["answer"],
    #                 "gt": gt,
    #                 "reasoning": parsed_pred["reasoning"]
    #             })

    #             if parsed_pred["answer"].lower() == gt.lower():
    #                 correct += 1
                
    #             total += 1
            
    #         if not os.path.exists("results_ft_models"):
    #             os.makedirs("results_ft_models")

    #         with open("results_ft_models/{}_{}_{:.2f}.json".format(model_id, dataset_name, correct / total), "w") as f:
    #             json.dump(results, f)
            
    #         print(f"Model: {model_id}, Dataset: {dataset_name}, Accuracy: {correct / total}")
