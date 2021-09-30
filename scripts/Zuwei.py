
#!python -m pip install --user -U nltk
#!python -m pip install -U gensim

import json
import os
from functools import reduce
import random
import nltk
import gensim
from nltk.data import find

nltk.download('word2vec_sample')
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)


with open('train_metadata.json') as f:
    train_text = json.load(f)



Q_templet = []

diff_tag = ['smaller', 'greater']

for i,train_sample in enumerate(train_text):
#train_sample = train_text[2]
    if len(train_sample['table']) == 2:
        label_dim = 0 #choice flag, string*0 ==> the string is not included
        metadata_1_idx = 0
        metadata_2_idx = random.randrange(1, (sum(x is not None for x in train_sample['table'][0]))) #random choose beside first name
        metadata_1 = train_sample['table'][0][metadata_1_idx]
        metadata_2 = train_sample['table'][0][metadata_2_idx]
        value_ticks = train_sample['table'][1] #all values in the graph
        metadata_1_value = value_ticks[metadata_1_idx]
        metadata_2_value = value_ticks[metadata_2_idx]
        diff_flag = random.randrange(2) #0 is smaller, 1 is greater
        rand_multi = random.randrange(2,10) #random number for the '<n> times greater/smaller' sentence
        rand_diff = random.randrange(max(value_ticks)) #random number for the '<n> greater/smaller' sentence
        rand_question = random.randrange(max(value_ticks))
        try:
            similar_words = model.most_similar(positive=train_sample['table'][0],topn = 10)
        except:
            continue
        similar_word = similar_words[9][0] #given all labels, generate 10 similar words then pick the last one as similar word 

    else:
        label_dim = 1
        sub_category_idx = random.randrange(1,len(train_sample['table']))
        sub_category = train_sample['table'][sub_category_idx] #sub_category name and value
        sub_category_name = sub_category[0]
        value_ticks = sub_category[1:]
        metadata_1_idx = 1
        metadata_2_idx = random.randrange(1, (sum(x is not None for x in train_sample['table'][0])))
        metadata_1 = train_sample['table'][0][metadata_1_idx]
        metadata_2 = train_sample['table'][0][metadata_2_idx]
        metadata_1_value = value_ticks[metadata_1_idx-1]
        metadata_2_value = value_ticks[metadata_2_idx-1]
        diff_flag = random.randrange(2)
        rand_multi = random.randrange(2,10)
        rand_diff = random.randrange(max(value_ticks)) 
        rand_question = random.randrange(max(value_ticks))
        try:
            similar_words = model.most_similar(positive=train_sample['table'][0][1:],topn = 10)
        except:
            continue
        similar_word = similar_words[9][0]

    sub_category_text = ''
    if label_dim:
        sub_category_text = ' in ' + sub_category_name + ' '

    rand_fact = random.random()

    if rand_fact < 0.4:
        fact = metadata_1.capitalize() + ' is ' + str(rand_multi) + ' times ' \
                + diff_tag[diff_flag] + ' than ' + similar_word + sub_category_text +'.'
        if diff_flag:
            similar_word_value = metadata_1_value / rand_multi
        else:
            similar_word_value = metadata_1_value * rand_multi
    elif 0.4 < rand_fact < 0.8:
        fact = metadata_1.capitalize() + ' is ' + str(rand_diff) + ' units ' \
                + diff_tag[diff_flag] + ' than ' + similar_word + sub_category_text + '.'
        if diff_flag:
            similar_word_value = metadata_1_value - rand_diff
        else:
            similar_word_value = metadata_1_value + rand_diff
    else:
        fact = 'The value of ' + similar_word + sub_category_text \
                +'is equals to the value of ' + metadata_1 + ' and ' + metadata_2 + ' combined.'
        similar_word_value = metadata_1_value + metadata_2_value

    rand_question = random.random()
    #rand_question = 0.7
    if rand_question < 0.11:
        question = 'What is the value of ' + similar_word + sub_category_text + '?'
        answer = str(similar_word_value)    
    elif 0.11 < rand_question < 0.22 and rand_fact < 0.8:
        if label_dim:
            question = 'Is the value of ' + similar_word + ' greater than ' + metadata_2 + sub_category_text + '?'
            if similar_word_value > metadata_2_value:
                answer = 'Yes'
            else:
                answer = 'No'
        else:
            question = 'Is the value of ' + similar_word + ' greater than ' + metadata_2 + sub_category_text + '?'
            if similar_word_value > metadata_2_value:
                answer = 'Yes'
            else:
                answer = 'No'

    elif 0.22 < rand_question < 0.33 and rand_fact < 0.8:
        if label_dim:
            question = 'Is the value of ' + similar_word + ' smaller than ' + metadata_2 + sub_category_text + '?'
            if similar_word_value < metadata_2_value:
                answer = 'Yes'
            else:
                answer = 'No'
        else:
            question = 'Is the value of ' + similar_word + ' smaller than ' + metadata_2 + sub_category_text + '?'
            if similar_word_value < metadata_2_value:
                answer = 'Yes'
            else:
                answer = 'No'
    elif 0.33 < rand_question < 0.44:
        question = 'How many bars have value greater than ' + str(rand_diff) + sub_category_text + '?'
        answer = reduce(lambda count, i: count + (i > rand_question), [*value_ticks,similar_word_value], 0)
    elif 0.44 < rand_question < 0.55:
        question = 'How many bars have value smaller than ' + str(rand_diff) + sub_category_text + '?'
        answer = reduce(lambda count, i: count + (i < rand_question), [*value_ticks,similar_word_value], 0)
    elif 0.55 < rand_question < 0.66:
        question = 'Which bar(s) has(have) the greatest value' + sub_category_text + '?'
        full_name_list = [*(train_sample['table'][0]),similar_word]
        g_value = max(*value_ticks,similar_word_value)
        if label_dim:
            g_idx = [i+1 for i,x in enumerate([*value_ticks,similar_word_value]) if x==g_value]
        else:
            g_idx = [i for i,x in enumerate([*value_ticks,similar_word_value]) if x==g_value]
        answer = [full_name_list[i] for i in g_idx]
    elif 0.66 < rand_question < 0.77:
        question = 'Which bar(s) has(have) the smallest value' + sub_category_text + '?'
        full_name_list = [*(train_sample['table'][0]),similar_word]
        g_value = min(*value_ticks,similar_word_value)
        if label_dim:
            g_idx = [i+1 for i,x in enumerate([*value_ticks,similar_word_value]) if x==g_value]
        else:
            g_idx = [i for i,x in enumerate([*value_ticks,similar_word_value]) if x==g_value]
        answer = [full_name_list[i] for i in g_idx]
    elif 0.77 < rand_question < 0.88:
        question = 'What is the sum of all values' + sub_category_text + '?'
        answer = sum(value_ticks,similar_word_value)
    else:
        question = 'What is the difference between the largest and the smallest value' + sub_category_text + '?'
        answer = max(*value_ticks,similar_word_value) - min(*value_ticks,similar_word_value)

    Q_templet.append({"dataset_id": "dvqa_train", "image_id": train_sample['image'], "graph_type": "bar", "passage": fact, "question": question, "answer": answer})
    if i == 500:
        break

f = open("samples.json", 'r')
samples = json.load(f)
f.close()

for sample in Q_templet:
    samples.append(sample)

f = open("samples.json", 'w')
json.dump(samples, f)
f.close()

print("Zuwei.py done")