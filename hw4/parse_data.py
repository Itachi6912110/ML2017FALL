import sys

#################################################
#                  Parameters                   #
#################################################

corpus_file = sys.argv[1]
corpus_file_nolabel = sys.argv[2]
#corpus_file_test = '/home/louiefu/Desktop/ML_HW4/testing_data.txt'
out_file = 'corpus.txt'

#################################################
#                 Parsing Data                  #
#################################################
#parse label data
f = open(corpus_file, 'r')
out = open(out_file, 'w')
for line in f:
	line = line[10:len(line)]
	out.write(line+' ')

print("finish parsing 1 ...")

f.close()

#parse nolabel data
f = open(corpus_file_nolabel, 'r')
for line in f:
	line = line[:len(line)]
	out.write(line+' ')
out.write('\n')
print("finish parsing 2 ...")

f.close()

"""
f = open(corpus_file_test, 'r')
count = 0
first_line = True
for line in f:
	if first_line:
		first_line = False
		continue
	line = line[1+len(str(count)):len(line)]
	count += 1
	out.write(line+' ')
out.write('\n')
print("finish parsing 3 ...")

f.close()
out.close()
"""