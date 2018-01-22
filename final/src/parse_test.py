import sys
import json
import jieba
import pickle
import numpy as np

with open(sys.argv[1]) as f:
	data = json.load(f)

context_list = []
question_list = []
id_list = []
#answer_list = []
#answer_text = []

i = 0
for topic in data["data"]:
	title = topic["title"]
	for p in topic["paragraphs"]:
		context = p["context"]
		qas = p["qas"]
		context_list.append(context)
		for qa in qas:
			question_id = qa["id"]
			question = qa["question"]
			#answers = qa["answers"]
			'''
			for ans in answers:
				start = ans["answer_start"]
				end = start + len(ans["text"])
				answer_list.append([start,end])
				answer_text.append(ans["text"])
			'''
			question_list.append([i,question])
			id_list.append(question_id)
		i += 1

for i in range(len(context_list)):
	temp = list(context_list[i])
	for j in range(len(context_list[i])):
		if temp[j] == '\n':
			if temp[j-1] != '。':
				temp[j] = '。'
			else:
				temp[j] = '!'
	modify= ''.join(temp)
	context_list[i] = modify
	
context_list = [sentence.split('。') for sentence in context_list]

jieba.set_dictionary("dict.txt.big")

context_cut = []
for i in range(len(context_list)):
	paragraph = []
	for j in range(len(context_list[i])):
		cut = jieba.cut(context_list[i][j],cut_all=False)
		sentence = ' '.join(cut)
		paragraph.append(sentence)
	context_cut.append(paragraph)

question_cut = []
for i in range(len(question_list)):
	cut = jieba.cut(question_list[i][1],cut_all=False)
	question_cut.append(' '.join(cut))

'''
answer_cut = []
for i in range(len(answer_text)):
	cut = jieba.cut(answer_text[i],cut_all=False)
	answer_cut.append(' '.join(cut))
'''
test_P_list = []
test_Q_list = []
len_counter = []
#answers = np.zeros((len(answer_list),2)).astype(int)
seletor = np.zeros(len(question_list)).astype(int)
k = 0
for question in question_list:
	paragraph = context_list[question[0]]
	appear = np.zeros(len(paragraph)).astype(int)
	counter = np.zeros(len(paragraph)).astype(int)
	
	for i in range(len(paragraph)):
		for j in range(len(question[1])):
			if question[1][j] in paragraph[i]:
				appear[i] = appear[i] + 1
		if i != 0: counter[i] = np.sum(counter[i-1]) + len(paragraph[i-1]) + 1
	index = np.argmax(appear)
	seletor[k] = index
	'''
	try:
		answers[k][0] = context_cut[question[0]][index].split().index(answer_cut[k].split()[0])
	except ValueError:
		answers[k][0] = 0
	try:
		answers[k][1] = context_cut[question[0]][index].split()[answers[k][0]:].index(answer_cut[k].split()[-1])+answers[k][0]
	except ValueError:
		answers[k][1] = 0
	'''
	test_P_list.append(context_cut[question[0]][index])
	test_Q_list.append(question_cut[k])
	len_counter.append(counter)
	k = k + 1

#print(list(answers))

with open("../preprocess/test_P.pkl", 'wb') as f:
    pickle.dump(test_P_list, f, pickle.HIGHEST_PROTOCOL)

with open("../preprocess/test_Q.pkl", 'wb') as f:
    pickle.dump(test_Q_list, f, pickle.HIGHEST_PROTOCOL)
'''
with open("preprocess/qas2cox.pkl", 'wb') as f:
    pickle.dump(question_list, f, pickle.HIGHEST_PROTOCOL)
'''
with open("../preprocess/test_id.pkl", 'wb') as f:
    pickle.dump(id_list, f, pickle.HIGHEST_PROTOCOL)

with open("../preprocess/test_context_len.pkl", 'wb') as f:
    pickle.dump(len_counter, f, pickle.HIGHEST_PROTOCOL)

#np.save("data/test_ans",answers)

np.save("../preprocess/test_selector",seletor)