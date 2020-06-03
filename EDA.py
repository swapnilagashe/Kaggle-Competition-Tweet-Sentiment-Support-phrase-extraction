#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:44:42 2020

@author: swapnillagashe
"""

#EDA for tweet sentiment analysis
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import cleaning

os.chdir('/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction')
train_data= pd.read_csv('train.csv')
test_data= pd.read_csv('test.csv')
train_data=train_data.dropna()
str1='I dont know how to do this project yet'
str2 ='lets find out how to do it'

#so we see that a.intersection(b) returns a set of common words where a and b are sets of words of two different strings
train_data = train_data.drop(['textID'],axis=1)
train_data.describe()

train_data['sentiment'].value_counts()
test_data['sentiment'].value_counts()
"""Observations:
    1. We see that most of the reviews are neutral"""
    
# define a function to return the number of common words betwwen two strings

train_data=train_data.head(200)
train_data['cleaned']=train_data.apply(lambda x: cleaning.preprocess(x['text']), axis=1)
test_data['cleaned']=test_data.apply(lambda x: cleaning.preprocess(x['text']), axis=1)



train_data['common_words'] = train_data.apply(lambda x: cleaning.common_words(x['text'], x['selected_text']), axis=1)
grouped=train_data.groupby('sentiment')
sums = grouped['common_words'].sum().add_suffix('_sum')
avgs = grouped['common_words'].mean().add_suffix('_avg')
grouped['common_words'].describe()
grouped['common_words'].hist()
new_df = pd.concat([sums, avgs], axis=1)
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
print(jaccard(str1,str2))
#------------------------------Lets check out some pos tagging
from nltk.tokenize import PunktSentenceTokenizer

train_data=train_data.dropna()
all_text_train = (' ').join(train_data['text'])
all_text_test = (' ').join(str(test_data['text']))
all_text = all_text_train + ' ' + all_text_test
train_data= train_data[train_data['sentiment']!='neutral']
""" I dropped the rows with neutral sentiment because for those rows, selected text is same as the original text

I think I will have to do chunking with chinking for pos tags to get the desired selected text
Can we use Seq2Seq model like the one used for machine translation (There also we are generating a target text from an Input text)"""
custom_sent_tokenizer=PunktSentenceTokenizer(all_text)
train_data['sentences']= train_data.apply(lambda x: custom_sent_tokenizer.tokenize(x['text']), axis=1)
train_data['words'] = train_data.apply(lambda x: nltk.word_tokenize(x['text'].lower()), axis=1)
train_data['tags1']= train_data.apply(lambda x: nltk.pos_tag(x['words']), axis=1)
train_data['selected_words']=train_data.apply(lambda x: nltk.word_tokenize(x['selected_text'].lower()), axis=1)
train_data['selected_tags1']=train_data.apply(lambda x: nltk.pos_tag(x['selected_words']), axis=1)

train_sample=train_data[0:50]
train_sample=train_sample[['text', 'selected_text','tags','selected_tags','tags1','selected_tags1']]
    
words1=nltk.word_tokenize(str1)
tags= [nltk.pos_tag(words1) for word in words1]
tag=nltk.pos_tag(['ball', 'is' ,'mine'])

chunkGram = """Chunk:{<(RB)>.?*<VB.?>*<NNP>+<NN>}"""

chunkParser = nltk.RegexpParser(chunkGram)
train_data['chunks']=train_data.apply(lambda x: nltk.chunkParser(x['selected_tags1']), axis=1)
'(<(DT)>)?(<(JJ)>)*(<(NN[^\\{\\}<>]*)>)'
test_data['sentiment'].value_counts()

import re
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
def Find(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 
    
text= 'www.abc.com, hello world!'

Find(text)
    
train_data['url_in_selected_text']= train_data.apply(lambda x : Find(x['selected_text']), axis=1)
train_data['url_in_text']=train_data.apply(lambda x : Find(x['text']), axis=1)

#train_data[train_data['sentiment']=='positive']['url_in_selected_text'].value_counts()
"""Some observations:
    For Sentiment
    1. neg (7781 rows)-> urls present : text- 238 rows, selected_text - 3 rows 
    2. pos(8582 rows)-> urls present : text - 426 rows, selected_text - 4 rows
    3. neu(11117 rows)-> urls present : text - 608 , selected_text - 368
    
    This might mean that for positive and negative rows, we can preprocess by removing the urls from training data(we can drop the rows that contain urls in selected text(only for pos and neg) or we can just remove the url keeping the remaining text)"""
    
