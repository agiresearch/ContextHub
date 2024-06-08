#!/bin/bash

# python evaluate.py --model contextualized-gpt-35 --dataset deductive_logic --level 1 --seed 42
# python compute_score.py --model contextualized-gpt-35 --dataset deductive_logic --level 1
python evaluate.py --model contextualized-gpt-35 --dataset deductive_logic --level 2 --seed 42
python compute_score.py --model contextualized-gpt-35 --dataset deductive_logic --level 2
ulimit -n 8192
python evaluate.py --model contextualized-gpt-35 --dataset deductive_logic --level 3 --seed 42
python compute_score.py --model contextualized-gpt-35 --dataset deductive_logic --level 3
ulimit -n 8192
python evaluate.py --model contextualized-gpt-35 --dataset deductive_logic --level 4 --seed 42
python compute_score.py --model contextualized-gpt-35 --dataset deductive_logic --level 4
ulimit -n 8192
# python evaluate.py --model contextualized-gpt-35 --dataset abductive_logic --level 1 --seed 42
# python compute_score.py --model contextualized-gpt-35 --dataset abductive_logic --level 1
python evaluate.py --model contextualized-gpt-35 --dataset abductive_logic --level 2 --seed 42
python compute_score.py --model contextualized-gpt-35 --dataset abductive_logic --level 2
ulimit -n 8192
python evaluate.py --model contextualized-gpt-35 --dataset abductive_logic --level 3 --seed 42
python compute_score.py --model contextualized-gpt-35 --dataset abductive_logic --level 3
ulimit -n 8192
python evaluate.py --model contextualized-gpt-35 --dataset abductive_logic --level 4 --seed 42
python compute_score.py --model contextualized-gpt-35 --dataset abductive_logic --level 4
ulimit -n 8192
python evaluate.py --model contextualized-gpt-35 --dataset abductive_logic --level 5 --seed 42
python compute_score.py --model contextualized-gpt-35 --dataset abductive_logic --level 5

# python evaluate.py --model gpt-35 --dataset deductive_logic --level 1 --seed 42
# python evaluate.py --model gpt-35 --dataset abductive_logic --level 1 --seed 42

# ulimit -n 8192
# python evaluate.py --model gpt-35 --dataset deductive_logic --level 2 --seed 42
# ulimit -n 8192
# python evaluate.py --model gpt-35 --dataset abductive_logic --level 2 --seed 42
# ulimit -n 8192
# python evaluate.py --model gpt-35 --dataset deductive_logic --level 3 --seed 42
# ulimit -n 8192
# python evaluate.py --model gpt-35 --dataset abductive_logic --level 3 --seed 42
# ulimit -n 8192
# python evaluate.py --model gpt-35 --dataset deductive_logic --level 4 --seed 42
# ulimit -n 8192
# python evaluate.py --model gpt-35 --dataset abductive_logic --level 4 --seed 42

# python evaluate.py --model yi-6 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model yi-9 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model yi-34 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model llama-7 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model llama-13 --dataset deductive_logic --level 1 --seed 42 --parse_only

# python evaluate.py --model yi-6 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model yi-9 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model yi-34 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model llama-7 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model llama-13 --dataset abductive_logic --level 1 --seed 42 --parse_only

# python evaluate.py --model yi-6 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model yi-9 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model yi-34 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model llama-7 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model llama-13 --dataset deductive_logic --level 2 --seed 42 --parse_only

# python evaluate.py --model yi-6 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model yi-9 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model yi-34 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model llama-7 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model llama-13 --dataset abductive_logic --level 2 --seed 42 --parse_only

# python evaluate.py --model yi-6 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model yi-9 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model yi-34 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model llama-7 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model llama-13 --dataset deductive_logic --level 3 --seed 42 --parse_only

# python evaluate.py --model yi-6 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model yi-9 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model yi-34 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model llama-7 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model llama-13 --dataset abductive_logic --level 3 --seed 42 --parse_only

# python evaluate.py --model yi-6 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model yi-9 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model yi-34 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model llama-7 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model llama-13 --dataset deductive_logic --level 4 --seed 42 --parse_only

# python evaluate.py --model yi-6 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model yi-9 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model yi-34 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model llama-7 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model llama-13 --dataset abductive_logic --level 4 --seed 42 --parse_only



# python evaluate.py --model qwen-0.5 --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model qwen-1.8 --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model qwen-4 --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model qwen-7 --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model qwen-14 --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model tulu-7 --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model tulu-13 --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model mistral --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model biomistral --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model saullm --dataset abductive_logic --seed 42 --proficiency B1
# python evaluate.py --model finma --dataset abductive_logic --seed 42 --proficiency B1

# python evaluate.py --model qwen-0.5 --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model qwen-1.8 --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model qwen-4 --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model qwen-7 --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model qwen-14 --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model tulu-7 --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model tulu-13 --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model mistral --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model biomistral --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model saullm --dataset abductive_logic --seed 10 --proficiency B1
# python evaluate.py --model finma --dataset abductive_logic --seed 10 --proficiency B1

# python evaluate.py --model qwen-0.5 --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model qwen-1.8 --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model qwen-4 --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model qwen-7 --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model qwen-14 --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model tulu-7 --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model tulu-13 --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model mistral --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model biomistral --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model saullm --dataset abductive_logic --seed 98 --proficiency B1
# python evaluate.py --model finma --dataset abductive_logic --seed 98 --proficiency B1

# python evaluate.py --model qwen-0.5 --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model qwen-1.8 --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model qwen-4 --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model qwen-7 --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model qwen-14 --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model tulu-7 --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model tulu-13 --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model mistral --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model biomistral --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model saullm --dataset abductive_logic --seed 76 --proficiency B1
# python evaluate.py --model finma --dataset abductive_logic --seed 76 --proficiency B1

# python evaluate.py --model qwen-0.5 --proficiency B1 --dataset deductive_logic --seed 42
# python evaluate.py --model qwen-1.8 --proficiency B1 --dataset deductive_logic --seed 42
# python evaluate.py --model qwen-4 --proficiency B1 --dataset deductive_logic --seed 42
# python evaluate.py --model qwen-7 --proficiency B1 --dataset deductive_logic --seed 42
# python evaluate.py --model qwen-14 --proficiency B1 --dataset deductive_logic --seed 42
# python evaluate.py --model tulu-7 --proficiency B1 --dataset deductive_logic --seed 42
# python evaluate.py --model tulu-13 --proficiency B1 --dataset deductive_logic --seed 42
# python evaluate.py --model mistral --dataset deductive_logic --seed 42 --proficiency B1
# python evaluate.py --model biomistral --dataset deductive_logic --seed 42 --proficiency B1
# python evaluate.py --model saullm --dataset deductive_logic --seed 42 --proficiency B1
# python evaluate.py --model finma --dataset deductive_logic --seed 42 --proficiency B1

# python evaluate.py --model qwen-0.5 --proficiency B1 --dataset deductive_logic --seed 10
# python evaluate.py --model qwen-1.8 --proficiency B1 --dataset deductive_logic --seed 10
# python evaluate.py --model qwen-4 --proficiency B1 --dataset deductive_logic --seed 10
# python evaluate.py --model qwen-7 --proficiency B1 --dataset deductive_logic --seed 10
# python evaluate.py --model qwen-14 --proficiency B1 --dataset deductive_logic --seed 10
# python evaluate.py --model tulu-7 --proficiency B1 --dataset deductive_logic --seed 10
# python evaluate.py --model tulu-13 --proficiency B1 --dataset deductive_logic --seed 10
# python evaluate.py --model mistral --dataset deductive_logic --seed 10 --proficiency B1
# python evaluate.py --model biomistral --dataset deductive_logic --seed 10 --proficiency B1
# python evaluate.py --model saullm --dataset deductive_logic --seed 10 --proficiency B1
# python evaluate.py --model finma --dataset deductive_logic --seed 10 --proficiency B1

# python evaluate.py --model qwen-0.5 --proficiency B1 --dataset deductive_logic --seed 98
# python evaluate.py --model qwen-1.8 --proficiency B1 --dataset deductive_logic --seed 98
# python evaluate.py --model qwen-4 --proficiency B1 --dataset deductive_logic --seed 98
# python evaluate.py --model qwen-7 --proficiency B1 --dataset deductive_logic --seed 98
# python evaluate.py --model qwen-14 --proficiency B1 --dataset deductive_logic --seed 98
# python evaluate.py --model tulu-7 --proficiency B1 --dataset deductive_logic --seed 98
# python evaluate.py --model tulu-13 --proficiency B1 --dataset deductive_logic --seed 98
# python evaluate.py --model mistral --dataset deductive_logic --seed 98 --proficiency B1
# python evaluate.py --model biomistral --dataset deductive_logic --seed 98 --proficiency B1
# python evaluate.py --model saullm --dataset deductive_logic --seed 98 --proficiency B1
# python evaluate.py --model finma --dataset deductive_logic --seed 98 --proficiency B1

# python evaluate.py --model qwen-0.5 --proficiency B1 --dataset deductive_logic --seed 76
# python evaluate.py --model qwen-1.8 --proficiency B1 --dataset deductive_logic --seed 76
# python evaluate.py --model qwen-4 --proficiency B1 --dataset deductive_logic --seed 76
# python evaluate.py --model qwen-7 --proficiency B1 --dataset deductive_logic --seed 76
# python evaluate.py --model qwen-14 --proficiency B1 --dataset deductive_logic --seed 76
# python evaluate.py --model tulu-7 --proficiency B1 --dataset deductive_logic --seed 76
# python evaluate.py --model tulu-13 --proficiency B1 --dataset deductive_logic --seed 76
# python evaluate.py --model mistral --dataset deductive_logic --seed 76 --proficiency B1
# python evaluate.py --model biomistral --dataset deductive_logic --seed 76 --proficiency B1
# python evaluate.py --model saullm --dataset deductive_logic --seed 76 --proficiency B1
# python evaluate.py --model finma --dataset deductive_logic --seed 76 --proficiency B1

# python evaluate.py --model qwen-32 --dataset deductive_logic --level 1 --seed 42
# python evaluate.py --model qwen-32 --dataset abductive_logic --level 1 --seed 42
# python evaluate.py --model qwen-32 --dataset deductive_logic --level 2 --seed 42
# python evaluate.py --model qwen-32 --dataset abductive_logic --level 2 --seed 42

# python evaluate.py --model qwen-0.5 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-32 --dataset abductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset abductive_logic --level 2 --seed 42
# python evaluate.py --model tulu-13 --dataset abductive_logic --level 2 --seed 42
# python evaluate.py --model phi3-mini --dataset abductive_logic --level 2 --seed 42

# python evaluate.py --model qwen-0.5 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model qwen-32 --dataset deductive_logic --level 1 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset deductive_logic --level 2 --seed 42
# python evaluate.py --model tulu-13 --dataset deductive_logic --level 2 --seed 42
# python evaluate.py --model phi3-mini --dataset deductive_logic --level 2 --seed 42

# python evaluate.py --model qwen-0.5 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-32 --dataset abductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset abductive_logic --level 2 --seed 42
# python evaluate.py --model tulu-13 --dataset abductive_logic --level 2 --seed 42
# python evaluate.py --model phi3-mini --dataset abductive_logic --level 2 --seed 42

# python evaluate.py --model qwen-0.5 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-32 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset deductive_logic --level 2 --seed 42
# python evaluate.py --model tulu-13 --dataset deductive_logic --level 2 --seed 42
# python evaluate.py --model phi3-mini --dataset deductive_logic --level 2 --seed 42

# python evaluate.py --model qwen-0.5 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-32 --dataset deductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset deductive_logic --level 3 --seed 42
# python evaluate.py --model tulu-13 --dataset deductive_logic --level 3 --seed 42
# python evaluate.py --model phi3-mini --dataset deductive_logic --level 3 --seed 42

# python evaluate.py --model qwen-0.5 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model qwen-32 --dataset abductive_logic --level 3 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset abductive_logic --level 3 --seed 42
# python evaluate.py --model tulu-13 --dataset abductive_logic --level 3 --seed 42
# python evaluate.py --model phi3-mini --dataset abductive_logic --level 3 --seed 42

# python evaluate.py --model qwen-0.5 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-32 --dataset deductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset deductive_logic --level 4 --seed 42
# python evaluate.py --model tulu-13 --dataset deductive_logic --level 4 --seed 42
# python evaluate.py --model phi3-mini --dataset deductive_logic --level 4 --seed 42

# python evaluate.py --model qwen-0.5 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model qwen-32 --dataset abductive_logic --level 4 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset abductive_logic --level 4 --seed 42
# python evaluate.py --model tulu-13 --dataset abductive_logic --level 4 --seed 42
# python evaluate.py --model phi3-mini --dataset abductive_logic --level 4 --seed 42

# python evaluate.py --model qwen-0.5 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-1.8 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-4 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-7 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model qwen-14 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model tulu-7 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model tulu-13 --dataset deductive_logic --level 2 --seed 42 --parse_only
# python evaluate.py --model phi3-mini --dataset deductive_logic --level 2 --seed 42 --parse_only

# python evaluate.py --model qwen-0.5 --proficiency B1 --dataset deductive_logic
# python evaluate.py --model qwen-1.8 --proficiency B1 --dataset deductive_logic
# python evaluate.py --model qwen-4 --proficiency B1 --dataset deductive_logic
# python evaluate.py --model qwen-7 --proficiency B1 --dataset deductive_logic
# python evaluate.py --model qwen-14 --proficiency B1 --dataset deductive_logic
# python evaluate.py --model tulu-7 --proficiency B1 --dataset deductive_logic
# python evaluate.py --model tulu-13 --proficiency B1 --dataset deductive_logic


# python evaluate.py --model qwen-0.5 --dataset arithmetic
# python evaluate.py --model qwen-1.8 --dataset arithmetic
# python evaluate.py --model qwen-4 --dataset arithmetic
# python evaluate.py --model qwen-7 --dataset arithmetic
# python evaluate.py --model qwen-14 --dataset arithmetic
# python evaluate.py --model tulu-7 --dataset arithmetic
# python evaluate.py --model tulu-13 --dataset arithmetic


# python evaluate.py --model qwen-0.5 --proficiency B1 --dataset arithmetic
# python evaluate.py --model qwen-1.8 --proficiency B1 --dataset arithmetic
# python evaluate.py --model qwen-4 --proficiency B1 --dataset arithmetic
# python evaluate.py --model qwen-7 --proficiency B1 --dataset arithmetic
# python evaluate.py --model qwen-14 --proficiency B1 --dataset arithmetic
# python evaluate.py --model tulu-7 --proficiency B1 --dataset arithmetic
# python evaluate.py --model tulu-13 --proficiency B1 --dataset arithmetic