# Disentangling Logic: The Role of Context in Large Language Model Reasoning Capabilities


![data](https://github.com/agiresearch/ContextHub/assets/28013619/99e82d1f-d44e-4a46-9706-59c6360dea79)



## Abstract
This study intends to disentangle pure logic reasoning and text understanding by investigating the contrast across abstract and contextualized logical problems from a comprehensive set of domains. We explore whether LLMs demonstrate genuine reasoning capabilities across various domains when the underlying logical structure remains constant. We focus on two main questions:

(1) Can abstract logical problems alone accurately benchmark LLMs' reasoning ability in real-world scenarios, disentangled from contextual support in practical settings? 

(2) Does fine-tuning LLMs on abstract logic problems generalize to contextualized logic problems and vice versa? 

To investigate these questions, we focus on standard propositional logic, specifically propositional deductive and abductive logic reasoning. In particular, we construct instantiated datasets for deductive and abductive reasoning with $4$ levels of difficulty, encompassing $12$ distinct domains based on the categorization of Wikipedia. Our experiments aim to provide insights into disentangling context in logical reasoning and the true reasoning capabilities of LLMs and their generalization potential. 

## Dataset Construction

![context_2](https://github.com/agiresearch/ContextHub/assets/28013619/b5403b7f-0ce7-428e-bc68-aecfa96b7949)

1. Create formal logic template from DyVal

2. Instantiate variables to meaningful sentences

3. Compile sentences into passage based on logic template


### Dataset
You can find the instantiated logic problems in the below directory, where {level_number} can be 1, 2, 3, 4, and {logic_type} can be "deductive_logic" and "abductive_logic".
```
contexthub/data/data_level{level_number}/{logic_type}.json
```

Data with groundtruth chain-of-thought reasoning can be found in the below directory:
```
contexthub/data/data_level{level_number}/{logic_type}_traincot.json
```

## Evaluation
Use the below command to run the evaluation, where {model} can be qwen-0.5...qwen-110, yi-6...yi-34, llama-7...llama-72.
```
python evaluate.py --model {model} --dataset {logic_type} --level {level_number} --seed 42
```
You can also find other commands in:
```
run_evaluate.sh
```

## Compute result
```
python compute_score.py --dataset {logic_type} --level {level_number} --model {model}
```
You can also find other commands in:
```
run_compute_score.sh
```
