#!/bin/bash

# python generate_abstract.py --dataset deductive_logic --level 1
# python generate_abstract.py --dataset deductive_logic --level 2
# python generate_abstract.py --dataset deductive_logic --level 3

# python generate_abstract.py --dataset abductive_logic --level 1
# python generate_abstract.py --dataset abductive_logic --level 2

python generate_abstract.py --dataset deductive_logic --level 4
python generate_abstract.py --dataset abductive_logic --level 4