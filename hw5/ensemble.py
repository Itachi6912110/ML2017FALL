from keras.models import load_model
from CFModel import CFModel, DeepModel, MFModel
import numpy as np
import pandas as pd
import pickle
import h5py
import csv
import sys

def read_data(filename):
	data = pd.read_csv(filename,sep=',').values[:]
	return data[:,1]

rfile1 = 'result_cf8.csv'
rfile2 = 'result_mf13.csv'
rfile3 = 'result_cf4.csv'

r1 = read_data(rfile1)
r2 = read_data(rfile2)
r3 = read_data(rfile3)

result = (r1+r2+r3)/3.0

output_file=open(sys.argv[1],'w')
writer=csv.writer(output_file)

writer.writerow(["TestDataId","Rating"])

for i in range(result.size):
	writer.writerow([i+1,result[i]])

output_file.close()
