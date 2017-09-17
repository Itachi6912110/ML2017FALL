import sys

f = open(sys.argv[1],'r')

words = f.read()[:-1].split(" ")
str_list = []
f.close()

for w in words :
	if w not in str_list:
		str_list.append(w)

f = open('Q1.txt','w')
index = 0
for w in str_list :
	f.write(w+" "+str(index)+" "+str(words.count(w))+"\n")
	index = index + 1

f.close()
