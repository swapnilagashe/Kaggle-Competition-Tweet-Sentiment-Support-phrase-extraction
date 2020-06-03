#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:05:36 2020

@author: swapnillagashe
"""

# Q/A approach using DistillBert
from simpletransformers.question_answering import QuestionAnsweringModel
from copy import deepcopy
import os
import json
use_cuda = True


# function that returns the starting index(indexes if given word occurs multiple time) in the input string for a given word or string
def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def do_qa_train(train):

    output = []
    for line in train:
        context = line[1]

        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer.lower()})
            break
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        output.append({'context': context.lower(), 'qas': qas})
        
    return output

def do_qa_test(test):
    output = []
    for line in test:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        if type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answers = []
        answers.append({'answer_start': 1000000, 'text': '__None__'})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
        output.append({'context': context.lower(), 'qas': qas})
    return output

train_df = pd.read_csv('train.csv')
train_df.dropna(inplace=True)
test_df = pd.read_csv('test.csv')

#converting to array for training purpose
train= np.array(train_df)                                          
test= np.array(test_df)                                          

qa_train= do_qa_train(train)
with open('train.json', 'w') as outfile:
    json.dump(qa_train, outfile)
    
qa_test = do_qa_test(test)

with open('test.json', 'w') as outfile:
    json.dump(qa_test, outfile)
    


MODEL_PATH = '/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction/Distillbert_model'

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('distilbert', 
                               MODEL_PATH, 
                               args={'reprocess_input_data': True,
                                     'overwrite_output_dir': True,
                                     'learning_rate': 5e-5,
                                     'num_train_epochs': 2,
                                     'max_seq_length': 192,
                                     'doc_stride': 64,
                                     'fp16': False
                                    },
                              use_cuda=use_cuda)
train_args = {
    'learning_rate': 3e-5,
    'num_train_epochs': 2,
    'max_seq_length': 384,
    'doc_stride': 128,
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'train_batch_size': 2,
    'gradient_accumulation_steps': 8,
}

model = QuestionAnsweringModel('bert', 'bert-base-cased', args=train_args)
model.train_model(train)