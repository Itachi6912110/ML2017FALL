#!/bin/bash

python3 parse_data.py $1 $2
python3 embedding.py
python3 train.py $1