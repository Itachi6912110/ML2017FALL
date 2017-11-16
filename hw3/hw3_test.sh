#!/bin/bash

tar zxvf test_models_1.tar.gz
wget -O restruct_mdl7.h5  https://www.dropbox.com/s/qiojb75ovt4g5fp/restruct_mdl7.h5?dl=0
python3 compare_model.py $1 $2
