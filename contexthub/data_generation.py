import json
from model import call_data_generation_model
import argparse
import re
import time
from prompt import RAW_DATA_PROMPTS, CONTEXTUALIZE_PROMPTS, RECOMBINE_PROMPT, LOGIC_FOLLOWING_CHECKING, COMMON_SENSE_CHECKING, SENSIBILITY_CHECKING, TAUTOLOGY_CHECKING
from category import CATEGORIES
import random


def parse_contextualized(text):
    # assert '<aaa>' in text and '</aaa>' in text
    natural_language_translation = {}

    # find per variable corresponding text
    # find per variable starts
    per_variable_starts = re.findall("<...>", text)
    if "</nl>" in per_variable_starts:
        per_variable_starts.remove("</nl>")
    start_positions = [text.index(s) for s in per_variable_starts]
    # find per variable ends
    per_variable_ends = re.findall("</...>", text)
    end_positions = [text.index(s) for s in per_variable_ends]
    assert len(per_variable_starts) == len(per_variable_ends)
    # extracted the corresponding texts
    variable_corresponding_texts = [text[start+5:end] for start, end in zip(start_positions, end_positions)]
    assert len(variable_corresponding_texts) == len(per_variable_starts)
    for i, variable in enumerate(per_variable_starts):
        natural_language_translation[variable] = variable_corresponding_texts[i]

    if '<nl>' in text and '</nl>' in text:
        # find natural language translation
        start = text.index("<nl>") + 4
        end = text.index("</nl>")
        natural_language_translation['<nl>'] = text[start:end]

    return natural_language_translation

def parse_contextualized_nl(text):
    assert '<nl>' in text and '</nl>' in text
    start = text.index("<nl>") + 4
    end = text.index("</nl>")
    return text[start:end]

def contextualize(model, prompt, logic, domain, sub_domain, explanation):
    while True:
        try:
            input_text = prompt.format(sub_domain, domain, explanation, logic, sub_domain, domain)
            # exit()
            contextualized = call_data_generation_model(model, input_text)
            # print(contextualized)
            # print("end call")
            result = parse_contextualized(contextualized)
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(e)
            time.sleep(1)
    if '<nl>' in result:
        return result 
    else:
        variable_text = ''
        for k,v in result.items():
            ending_k = '</' + k[1:]
            variable_text += f"{k}{v}{ending_k}\n"
        input_text = RECOMBINE_PROMPT.format(variable_text, logic)
        while True:
            try:
                recombine_nl = call_data_generation_model(model, input_text)
                nl = parse_contextualized_nl(recombine_nl)
                result['<nl>'] = nl
                return result 
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e)
                time.sleep(1)
    

##### check whether a data generated have good quality
def quality_control(logic, contextualized_logic):
    def parse_answer(result):
        print(result)
        assert '<answer>' in result and '</answer>' in result
        start = result.index("<answer>") + len('<answer>')
        end = result.index("</answer>")
        answer = result[start:end]
        assert answer in ['Yes', 'No']
        return answer
    def logic_following(logic, contextualized_logic):
        prompt = LOGIC_FOLLOWING_CHECKING.format(logic, contextualized_logic)
        while True:
            try: 
                result = call_data_generation_model("claude", prompt)
                answer = parse_answer(result)
                return answer
            except KeyboardInterrupt:
                raise
            except:
                time.sleep(0.1)
    def common_sense(contextualized_logic):
        prompt = COMMON_SENSE_CHECKING.format(contextualized_logic)
        while True:
            try:
                result = call_data_generation_model("claude", prompt)
                answer = parse_answer(result)
                return answer
            except KeyboardInterrupt:
                raise
            except:
                time.sleep(0.1)
    def sensible(contextualized_logic):
        prompt = SENSIBILITY_CHECKING.format(contextualized_logic)
        while True:
            try:
                result = call_data_generation_model("claude", prompt)
                answer = parse_answer(result)
                return answer
            except KeyboardInterrupt:
                raise
            except:
                time.sleep(0.1)
    def tautology(contextualized_logic):
        prompt = TAUTOLOGY_CHECKING.format(contextualized_logic)
        while True:
            try:
                result = call_data_generation_model("claude", prompt)
                answer = parse_answer(result)
                return answer
            except KeyboardInterrupt:
                raise
            except:
                time.sleep(0.1)
    #logic_following_answer = logic_following(logic, contextualized_logic)
    common_sense_answer = common_sense(contextualized_logic)
    sensible_answer = sensible(contextualized_logic)
    tautology_answer = tautology(contextualized_logic)
    if common_sense_answer == 'No' and sensible_answer == 'Yes' and tautology_answer == 'No':
        return True
    return False

##### check whether a data is repeated
def check_repetition(new_data, data):
    """
    Check whether the new data is repeated in the old data.
    """
    for old_data in data:
        if set(new_data.values()) & set(old_data.values()):
            return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="claude", help="The model to use for translation.")
    parser.add_argument("--dataset", type=str, default="abductive_logic", help="The dataset to be contextualized.")
    parser.add_argument("--number_of_instantiation", type=int, default=5, help="The number of contextualized logic problems to generate for each abstract logical question.")
    parser.add_argument('--data_level',type=int, default='3')
    args = parser.parse_args()

    raw_data_prompt = RAW_DATA_PROMPTS[args.dataset]
    contextualize_prompt = CONTEXTUALIZE_PROMPTS[args.dataset]

    with open(f'data/{args.dataset}_level{args.data_level}.json', 'r') as f:
        raw_data = json.load(f)
    contextualized_data = []
    for raw_d in raw_data:
        question = raw_data_prompt.format(**raw_d)
        answer = raw_d['answers']

        data = {'question':[{'<nl>': question}], 'answer':answer}
        # domains = ["everyday"]

        # domains = ["everyday", "legal", "financial", "medical", "psychological", "astronomy", "botanical", "chemical", "real estate", "united nations"]
        domains = CATEGORIES.keys()
        domains = list(domains)
        for domain in domains:
            data[domain] = {}
            sub_domains = list(CATEGORIES[domain].keys())
            
            while len(list(data[domain].keys())) < args.number_of_instantiation:
                
                sub_domain = random.choice(sub_domains)
                explanation = CATEGORIES[domain][sub_domain]

                print("==========================")
                print(f"Domain: {domain}")
                print(f"Sub domain: {sub_domain}")
                print(data[domain])
            
                print("==========================")
                print(f"Number of Instantiation: {len(data[domain])}")
                result = contextualize(args.model, contextualize_prompt, question, domain, sub_domain, explanation)
                quality_controlled = quality_control(question, result['<nl>'])
                # if not check_repetition(result, data[domain]) and quality_controlled:
                if quality_controlled:
                    print(result)
                    data[domain][sub_domain] = result
        
        contextualized_data.append(data)

        with open(f'data/data_level{args.data_level}/{args.dataset}_new_categories_w_explanation.json', 'w') as f:
            json.dump(contextualized_data, f, indent=4)

    


