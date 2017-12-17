#!/bin/bash

python3 predict.py model/model_mf13.h5 result_mf13.csv $1 1 0
python3 predict.py model/model_cf8.h5 result_cf8.csv $1 1 1
python3 predict.py model/model_cf4.h5 result_cf4.csv $1 0 1
python3 ensemble.py $2