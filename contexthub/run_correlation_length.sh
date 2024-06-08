#!/bin/bash


# deductive logic
python correlation_length.py --model qwen-0.5 --dataset deductive_logic --level 1
python correlation_length.py --model qwen-0.5 --dataset deductive_logic --level 2
python correlation_length.py --model qwen-0.5 --dataset deductive_logic --level 3
python correlation_length.py --model qwen-0.5 --dataset deductive_logic --level 4


python correlation_length.py --model qwen-0.5 --dataset abductive_logic --level 1
python correlation_length.py --model qwen-0.5 --dataset abductive_logic --level 2
python correlation_length.py --model qwen-0.5 --dataset abductive_logic --level 3
python correlation_length.py --model qwen-0.5 --dataset abductive_logic --level 4



# deductive logic
python correlation_length.py --model qwen-7 --dataset deductive_logic --level 1
python correlation_length.py --model qwen-7 --dataset deductive_logic --level 2
python correlation_length.py --model qwen-7 --dataset deductive_logic --level 3
python correlation_length.py --model qwen-7 --dataset deductive_logic --level 4

python correlation_length.py --model qwen-7 --dataset abductive_logic --level 1
python correlation_length.py --model qwen-7 --dataset abductive_logic --level 2
python correlation_length.py --model qwen-7 --dataset abductive_logic --level 3
python correlation_length.py --model qwen-7 --dataset abductive_logic --level 4


# deductive logic
python correlation_length.py --model qwen-32 --dataset deductive_logic --level 1
python correlation_length.py --model qwen-32 --dataset deductive_logic --level 2
python correlation_length.py --model qwen-32 --dataset deductive_logic --level 3
python correlation_length.py --model qwen-32 --dataset deductive_logic --level 4

python correlation_length.py --model qwen-32 --dataset abductive_logic --level 1
python correlation_length.py --model qwen-32 --dataset abductive_logic --level 2
python correlation_length.py --model qwen-32 --dataset abductive_logic --level 3
python correlation_length.py --model qwen-32 --dataset abductive_logic --level 4


# deductive logic
python correlation_length.py --model qwen-110 --dataset deductive_logic --level 1
python correlation_length.py --model qwen-110 --dataset deductive_logic --level 2
python correlation_length.py --model qwen-110 --dataset deductive_logic --level 3
python correlation_length.py --model qwen-110 --dataset deductive_logic --level 4

python correlation_length.py --model qwen-110 --dataset abductive_logic --level 1
python correlation_length.py --model qwen-110 --dataset abductive_logic --level 2
python correlation_length.py --model qwen-110 --dataset abductive_logic --level 3
python correlation_length.py --model qwen-110 --dataset abductive_logic --level 4