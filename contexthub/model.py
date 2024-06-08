import anthropic
import google.generativeai as genai
import time
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, Phi3ForCausalLM
import torch


from huggingface_hub import login
login(token = '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')

####### data generation close-source models #######


def call_anthropic_api(message):
    api_client = anthropic.Anthropic(
        api_key=claude_key,
    )
    system_prompt = "You are a rational assistant that carefully answer the question."
    message = api_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=3000,
        temperature=1,
        system=system_prompt,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    return message.content[0].text

def call_gemini_api(message):
    retries = 60  # Maximum number of retries
    while retries > 0:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(message)
            return response.text
        except Exception as e:
            retries -= 1
            time.sleep(0.1)

def call_gpt4o_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = "gpt-4o",
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_gpt4_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = "gpt-4-turbo", 
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content


def call_contextualized_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9SgzFY26',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_abstract_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9Swm3OAg',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_culture_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9U39iPwd',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_math_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9U3NgfeC',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_health_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9U3Juq9u',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_society_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = "ft:gpt-3.5-turbo-0125:robustlearn:society:9UtPMnTt",
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_technology_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = "ft:gpt-3.5-turbo-0125:robustlearn:tech:9UtGDNih",
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_human_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:robustlearn:human:9UqrKt91',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_people_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:robustlearn:people:9Ur2jbnT',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_geography_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = "ft:gpt-3.5-turbo-0125:robustlearn:geo:9Ur2R0tB",
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_philosophy_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:robustlearn:phi:9Us6b7Q0',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_religion_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:robustlearn:religon:9UsJt7xY',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_science_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:robustlearn:sci:9Us7xPp7',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content


def call_culturemath_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9U4hNB6I',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content


def call_partial1_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9SwvJmQg',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_partial2_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9Sxm1qKh',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def call_partial3_gpt35_api(message):
    openai_client = OpenAI(api_key=open_ai_key)
    response = openai_client.chat.completions.create(
        model = 'ft:gpt-3.5-turbo-0125:rutgers-university::9SzKkDED',
        messages=[
            {"role": "user", "content": message},
        ],
        temperature=0.0,
        max_tokens=2000
    )
    return response.choices[0].message.content


def call_data_generation_model(model, message):
    if model == 'gemini':
        result = call_gemini_api(message)
    elif model == 'claude':
        result = call_anthropic_api(message)
    elif model == 'gpt-4o':
        result = call_gpt4o_api(message)
    elif model == 'gpt-4':
        result = call_gpt4_api(message)
    elif model == 'gpt-35':
        result = call_gpt35_api(message)
    elif model == 'contextualized-gpt-35':
        result = call_contextualized_gpt35_api(message)
    elif model == 'abstract-gpt-35':
        result = call_abstract_gpt35_api(message)
    elif model == 'partial1-gpt-35':
        result = call_partial1_gpt35_api(message)
    elif model == 'partial2-gpt-35':
        result = call_partial2_gpt35_api(message)
    elif model == 'partial3-gpt-35':
        result = call_partial3_gpt35_api(message)
    elif model == 'culture-gpt-35':
        result = call_culture_gpt35_api(message)
    elif model == 'math-gpt-35':
        result = call_math_gpt35_api(message)
    elif model == 'philosophy-gpt-35':
        result = call_philosophy_gpt35_api(message)
    elif model == 'religion-gpt-35':
        result = call_religion_gpt35_api(message)
    elif model == 'science-gpt-35':
        result = call_science_gpt35_api(message)
    elif model == 'people-gpt-35':
        result = call_people_gpt35_api(message)
    elif model == 'human-gpt-35':
        result = call_human_gpt35_api(message)
    elif model == 'geography-gpt-35':
        result = call_geography_gpt35_api(message)
    elif model == 'technology-gpt-35':
        result = call_technology_gpt35_api(message)
    elif model == 'health-gpt-35':
        result = call_health_gpt35_api(message)
    elif model == 'society-gpt-35':
        result = call_society_gpt35_api(message)
    elif model == 'mixed2-gpt-35':
        result = call_culturemath_gpt35_api(message)

    return result

####### model loading for domain specific model #######
####### medical #######
def load_BioMistral_7(text_prompt, model_name=None):
    if model_name is None:
        model_name = 'BioMistral/BioMistral-7B'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 32768,
    )
    return llm

def load_SaulLM_7(text_prompt, model_name=None):
    if model_name is None:
        model_name = 'Equall/Saul-Instruct-v1'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 32768,
    )
    return llm

def load_mistral(text_prompt, model_name=None):
    if model_name is None:
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm

def load_finma_7(text_prompt, model_name=None):
    if model_name is None:
        model_name = 'ChanceFocus/finma-7b-nlp'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        dtype=torch.float16,
        max_num_batched_tokens=4 * 2048,
    )
    return llm

####### model loading with scaling #######
####### tulu #######
def load_tulu_7(text_prompt, model_name=None):
    if model_name is None:
        model_name = 'allenai/tulu-2-7b'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm

def load_tulu_13(text_prompt, model_name=None):
    if model_name is None:
        model_name = 'allenai/tulu-2-13b'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm

def load_tulu_70(model_name=None):
    if model_name is None:
        model_name = 'allenai/tulu-2-70b'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


####### qwen-1.5 #######
def load_qwen_05(model_name=None):
    if model_name is None:
        model_name = 'Qwen/Qwen1.5-0.5B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm

def load_qwen_18(model_name=None):
    if model_name is None:
        model_name = 'Qwen/Qwen1.5-1.8B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm

def load_qwen_4(model_name=None):
    if model_name is None:
        model_name = 'Qwen/Qwen1.5-4B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm

def load_qwen_7(model_name=None):
    if model_name is None:
        model_name = 'Qwen/Qwen1.5-7B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


def load_qwen_14(model_name=None):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    if model_name is None:
        model_name = 'Qwen/Qwen1.5-14B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


def load_qwen_32(model_name=None):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    if model_name is None:
        model_name = 'Qwen/Qwen1.5-32B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm

def load_qwen_72(model_name=None):
    if model_name is None:
        model_name = 'Qwen/Qwen1.5-72B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


def load_qwen_110(model_name=None):
    if model_name is None:
        model_name = 'Qwen/Qwen1.5-110B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm

def load_yi_6(model_name=None):
    if model_name is None:
        model_name = '01-ai/Yi-1.5-6B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


def load_yi_9(model_name=None):
    if model_name is None:
        model_name = '01-ai/Yi-1.5-9B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


def load_yi_34(model_name=None):
    if model_name is None:
        model_name = '01-ai/Yi-1.5-34B-Chat'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


def load_llama_7(model_name=None):
    if model_name is None:
        model_name = 'meta-llama/Llama-2-7b-chat-hf'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


def load_llama_13(model_name=None):
    if model_name is None:
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm


def load_llama_70(model_name=None):
    if model_name is None:
        model_name = 'meta-llama/Llama-2-70b-chat-hf'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    llm = LLM(
        model=model_name,
        download_dir=download_dir,
        trust_remote_code=True,
        tensor_parallel_size=4, 
        max_num_seqs=4,
        max_num_batched_tokens=4 * 8192,
    )
    return llm



def load_phi3_mini(model_name=None):
    if model_name is None:
        model_name = 'microsoft/phi-3-mini-4k-instruct'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    # llm = LLM(
    #     model=model_name,
    #     download_dir=download_dir,
    #     trust_remote_code=True,
    #     tensor_parallel_size=4, 
    #     max_num_seqs=4,
    #     max_num_batched_tokens=4 * 8192,
    # )

    llm = Phi3ForCausalLM.from_pretrained(model_name, cache_dir=download_dir, device_map='auto', trust_remote_code=True)
    return llm


def load_phi3_small(model_name=None):
    if model_name is None:
        model_name = 'microsoft/Phi-3-small-4k-instruct'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    # llm = LLM(
    #     model=model_name,
    #     download_dir=download_dir,
    #     trust_remote_code=True,
    #     tensor_parallel_size=4, 
    #     max_num_seqs=4,
    #     max_num_batched_tokens=4 * 8192,
    # )
    llm = Phi3ForCausalLM.from_pretrained(model_name, cache_dir=download_dir, device_map='auto')
    return llm

def load_phi3_medium(model_name=None):
    if model_name is None:
        model_name = 'microsoft/Phi-3-medium-4k-instruct'
        download_dir = 'pretrained_models'
    else:
        download_dir = model_name
    # llm = LLM(
    #     model=model_name,
    #     download_dir=download_dir,
    #     trust_remote_code=True,
    #     tensor_parallel_size=4, 
    #     max_num_seqs=4,
    #     max_num_batched_tokens=4 * 8192,
    # )
    llm = Phi3ForCausalLM.from_pretrained(model_name, cache_dir=download_dir, device_map='auto')
    return llm



def load_evaluation_model(model, model_name=None):
    if model == 'tulu-7':
        llm = load_tulu_7(model_name=model_name)
    elif model == 'tulu-13':
        llm = load_tulu_13(model_name=model_name)
    elif model == 'tulu-70':
        llm = load_tulu_70(model_name=model_name)
    elif model == 'llama-7':
        llm = load_llama_7(model_name=model_name)
    elif model == 'llama-13':
        llm = load_llama_13(model_name=model_name)
    elif model == 'llama-70':
        llm = load_llama_70(model_name=model_name)
    elif model == 'yi-6':
        llm = load_yi_6(model_name=model_name)
    elif model == 'yi-9':
        llm = load_yi_9(model_name=model_name)
    elif model == 'yi-34':
        llm = load_yi_34(model_name=model_name)
    elif model == 'qwen-0.5':
        llm = load_qwen_05(model_name=model_name)
    elif model == 'qwen-1.8':
        llm = load_qwen_18(model_name=model_name)
    elif model == 'qwen-4':
        llm = load_qwen_4(model_name=model_name)
    elif model == 'qwen-7':
        llm = load_qwen_7(model_name=model_name)
    elif model == 'qwen-14':
        llm = load_qwen_14(model_name=model_name)
    elif model == 'qwen-32':
        llm = load_qwen_32(model_name=model_name)
    elif model == 'qwen-72':
        llm = load_qwen_72(model_name=model_name)
    elif model == 'qwen-110':
        llm = load_qwen_110(model_name=model_name)
    elif model == 'phi3-mini':
        llm = load_phi3_mini(model_name=model_name)
    elif model == 'phi3-small':
        llm = load_phi3_small(model_name=model_name)
    elif model == 'phi3-medium':
        llm = load_phi3_medium(model_name=model_name)
    elif model == 'biomistral':
        llm = load_BioMistral_7(model_name=model_name)
    elif model == 'saullm':
        llm = load_SaulLM_7(model_name=model_name)
    elif model == 'mistral':
        llm = load_mistral(model_name=model_name)
    elif model == 'finma':
        llm = load_finma_7(model_name=model_name)

    return llm

####### model evaluation with scaling #######
####### domain specific related #######
def run_BioMistral_7(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<s>[INST] '+t+' [/INST]' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

def run_SaulLM_7(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<s>[INST] '+t+' [/INST]' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

def run_mistral(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<s>[INST] '+t+' [/INST]' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

def run_finma_7(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['Human: \n'+t+'\n\nAssistant: \n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

####### tulu #######
def run_tulu_7(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|user|>\n'+t+'<|assistant|>\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

def run_tulu_13(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|user|>\n'+t+'<|assistant|>\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

def run_tulu_70(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 


####### qwen-1.5 #######
def run_qwen_05(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

def run_qwen_18(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

def run_qwen_4(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 

def run_qwen_7(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 


def run_qwen_14(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 


def run_qwen_32(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 


def run_qwen_72(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 


def run_qwen_110(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions 


def run_yi_6(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions


def run_yi_9(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions

def run_yi_34(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+t+'<|im_end|>\n<|im_start|>assistant\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions


def run_llama_7(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<s>[INST] <<SYS>>\nYou are a helpful assistant.<</SYS>>\n\n'+t+' [/INST]\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions


def run_llama_13(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<s>[INST] <<SYS>>\nYou are a helpful assistant.<</SYS>>\n\n'+t+' [/INST]\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions


def run_llama_70(llm, text_prompt):
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    text_prompt = ['<s>[INST] <<SYS>>\nYou are a helpful assistant.<</SYS>>\n\n'+t+' [/INST]\n' for t in text_prompt]
    predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    for RequestOutput in predictions:
        output = RequestOutput.outputs[0].text
        all_predictions.append(output)
    return all_predictions


def run_phi3_mini(llm, text_prompt):
    # sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    # text_prompt = ['<s><|user|>\n'+t+'\n<|assistant|>\n' for t in text_prompt]
    # predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    # for RequestOutput in predictions:
    #     output = RequestOutput.outputs[0].text
    #     all_predictions.append(output)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct", cache_dir='pretrained_models', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    batch_size = 10
    for i in range(0, len(text_prompt), batch_size):
        one_text_prompt = text_prompt[i:i+batch_size]
        inputs = tokenizer(one_text_prompt, return_tensors="pt", padding=True)
        generate_ids = llm.generate(inputs.input_ids, max_length=500)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        all_predictions.extend(output)
    # for one_text_prompt in text_prompt:
    #     inputs = tokenizer(one_text_prompt, return_tensors="pt")
    #     generate_ids = llm.generate(inputs.input_ids, max_length=100)
    #     output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     all_predictions.append(output)
    return all_predictions 


def run_phi3_small(llm, text_prompt):
    # sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    # text_prompt = ['<s><|user|>\n'+t+'\n<|assistant|>\n' for t in text_prompt]
    # predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    # for RequestOutput in predictions:
    #     output = RequestOutput.outputs[0].text
    #     all_predictions.append(output)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-small-4k-instruct", cache_dir='pretrained_models', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    batch_size = 5
    for i in range(0, len(text_prompt), batch_size):
        one_text_prompt = text_prompt[i:i+batch_size]
        inputs = tokenizer(one_text_prompt, return_tensors="pt", padding=True)
        generate_ids = llm.generate(inputs.input_ids, max_length=500)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        all_predictions.extend(output)
    return all_predictions 

def run_phi3_medium(llm, text_prompt):
    # sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    # text_prompt = ['<s><|user|>\n'+t+'\n<|assistant|>\n' for t in text_prompt]
    # predictions = llm.generate(text_prompt, sampling_params)
    all_predictions = []
    # for RequestOutput in predictions:
    #     output = RequestOutput.outputs[0].text
    #     all_predictions.append(output)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-medium-4k-instruct", cache_dir='pretrained_models', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    batch_size = 2
    for i in range(0, len(text_prompt), batch_size):
        one_text_prompt = text_prompt[i:i+batch_size]
        inputs = tokenizer(one_text_prompt, return_tensors="pt", padding=True)
        generate_ids = llm.generate(inputs.input_ids, max_length=500)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        all_predictions.extend(output)

    return all_predictions 


def call_evaluation_model(model, text_prompt, llm):
    if model == 'tulu-7':
        result = run_tulu_7(llm, text_prompt)
    elif model == 'tulu-13':
        result = run_tulu_13(llm, text_prompt)
    elif model == 'tulu-70':
        result = run_tulu_70(llm, text_prompt)
    elif model == 'llama-7':
        result = run_llama_7(llm, text_prompt)
    elif model == 'llama-13':
        result = run_llama_13(llm, text_prompt)
    elif model == 'llama-70':
        result = run_llama_70(llm, text_prompt)
    elif model == 'yi-6':
        result = run_yi_6(llm, text_prompt)
    elif model == 'yi-9':
        result = run_yi_9(llm, text_prompt)
    elif model == 'yi-34':
        result = run_yi_34(llm, text_prompt)
    elif model == 'qwen-0.5':
        result = run_qwen_05(llm, text_prompt)
    elif model == 'qwen-1.8':
        result = run_qwen_18(llm, text_prompt)
    elif model == 'qwen-4':
        result = run_qwen_4(llm, text_prompt)
    elif model == 'qwen-7':
        result = run_qwen_7(llm, text_prompt)
    elif model == 'qwen-14':
        result = run_qwen_14(llm, text_prompt)
    elif model == 'qwen-32':
        result = run_qwen_32(llm, text_prompt)
    elif model == 'qwen-72':
        result = run_qwen_72(llm, text_prompt)
    elif model == 'qwen-110':
        result = run_qwen_110(llm, text_prompt)
    elif model == 'phi3-mini':
        result = run_phi3_mini(llm, text_prompt)
    elif model == 'phi3-medium':
        result = run_phi3_medium(llm, text_prompt)
    elif model == 'phi3-small':
        result = run_phi3_small(llm, text_prompt)
    elif model == 'biomistral':
        result = run_BioMistral_7(llm, text_prompt)
    elif model == 'saullm':
        result = run_SaulLM_7(llm, text_prompt)
    elif model == 'mistral':
        result = run_mistral(llm, text_prompt)
    elif model == 'finma':
        result = run_finma_7(llm, text_prompt)

    return result


def load_tokenizer(args):
    if args.model == 'phi3-mini':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'qwen-0.5':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'qwen-1.8':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'qwen-4':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'qwen-7':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'qwen-14':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'qwen-32':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-32B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'qwen-72':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'qwen-110':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-110B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'tulu-7':
        tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-7b", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'tulu-13':
        tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-13b", cache_dir='pretrained_models', trust_remote_code=True) 
    elif args.model == 'yi-6':
        tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-6B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'yi-9':
        tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-9B-Chat", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'yi-34':
        tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-34B-Chat", cache_dir='pretrained_models', trust_remote_code=True) 
    elif args.model == 'llama-7':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'llama-13':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", cache_dir='pretrained_models', trust_remote_code=True)
    elif args.model == 'llama-70':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf", cache_dir='pretrained_models', trust_remote_code=True)
    return tokenizer

