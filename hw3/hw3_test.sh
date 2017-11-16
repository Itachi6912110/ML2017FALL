#!/bin/bash

tar zxvf test_models_1.tar.gz
wget -O restruct_mdl7.h5  https://www.dropbox.com/s/qiojb75ovt4g5fp/restruct_mdl7.h5?dl=0
wget -O model_sample_3.h5 https://www.dropbox.com/s/rnr3opxmyilnloa/model_sample_3.h5?dl=0
wget -O model_sample_4.h5 https://www.dropbox.com/s/l67xd3qhb8mkhn3/model_sample_4.h5?dl=0
python3 compare_model.py $1 $2
