#!/bin/bash

wget -O gensim_pre3.h5 https://www.dropbox.com/s/obwkerxl47is7te/gensim_pre3.h5?dl=0
wget -O embed_model_128 https://www.dropbox.com/s/5e9bslxj7g64kqm/embed_model_128?dl=0
python3 test.py $1 $2