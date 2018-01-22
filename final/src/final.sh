#!/bin/bash
wget --content-disposition -P ../model/ https://www.dropbox.com/s/815qok011f0t0jc/FastQA?dl=1
wget --content-disposition -P ../model/ https://www.dropbox.com/s/4k07vytrc22sbtm/w2v?dl=1
python3 parse_test.py $1
python3 test.py $2 