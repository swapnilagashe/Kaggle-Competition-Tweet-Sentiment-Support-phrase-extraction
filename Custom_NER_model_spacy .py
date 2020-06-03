#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:16:12 2020

@author: swapnillagashe
"""
"""Some improvements that can be done:
Remove urls from text as I don't think that the selected text will have url in them (check this)
Decide if any form of preprocessing that can be done based on EDA on selected text in train_data
This problem can also be aprroached as a Q/A type model, do some research on this"""


import os
import numpy as np
import pandas as pd
from tqdm import tqdm


#####training a custom NER model with spacy

import plac
import random
import warnings
from pathlib import Path
import spacy

from spacy.util import minibatch, compounding
os.chdir('/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction')
train_data= pd.read_csv('train.csv')
train_data=train_data.dropna()
test_data= pd.read_csv('test.csv')

def save_model(output_dir, nlp, new_model_name):
#    output_dir = f'/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction/NER_models/'
    output_dir=get_model_out_path(sentiment)
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        
        
def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()

        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,   # dropout - make it harder to memorise data
                    losses=losses, 
                )
            
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')
    
    
def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = '/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction/NER_models_exp/model_pos'
    elif sentiment == 'negative':
        model_out_path = '/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction/NER_models_exp/model_neg'
    else:
        model_out_path = '/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction/NER_models_exp/model_neu'
    return model_out_path

## creating data in spacy data input format

#def get_training_data(sentiment):
#    train_df = []
#    for index, row in train_data.iterrows():
#        if row.sentiment == sentiment:
#            selected_text = row.selected_text
#            text = row.text
#            start = text.find(selected_text)
#            end = start + len(selected_text)
#            train_df.append((text, {"entities": [[start, end, 'selected_text']]}))
#    return train_df

def get_training_data(sentiment):
    train_df = []
    for index, row in train_data_pos.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.cleaned_selected_Text
            text = row.cleaned
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_df.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_df
    
##train the model for each sentiment seperately
    
#sentiment='positive'
#train_pos=get_training_data(sentiment)
#model_path = get_model_out_path(sentiment)

#train(train_pos, model_path, n_iter=2, model=None)

sentiments=['positive','negative', 'neutral']
for sentiment in sentiments:
    train_df=get_training_data(sentiment)
    model_path = get_model_out_path(sentiment)
    train(train_df, model_path, n_iter=2, model=None)

#TRAINED_MODELS_BASE_PATH = '/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction/NER_models/' #path where models are saved
TRAINED_MODELS_BASE_PATH = '/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction/NER_models_exp/' #path where models are saved


def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text    

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading Models  from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
#    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
#    model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')
    
    count=0
    jaccard_score = 0
    k=0
    for row in tqdm(train_data.itertuples(), total=train_data.shape[0]):
        text = row.text
        if row.sentiment == 'neutral':
#            jaccard_score += jaccard(predict_entities(text, model_neu), row.selected_text)
#            count +=1
            k=0
        
            
        elif row.sentiment == 'positive':
            jaccard_score += jaccard(predict_entities(text, model_pos), row.selected_text)
            count +=1
#            k=0
            
        else:
#            jaccard_score += jaccard(predict_entities(text, model_neg), row.selected_text) 
#            count +=1
            k=0

            
print(f'Average Jaccard Score is {jaccard_score/count }') 


"""with 2 iters 
Jaccard score for positive is 0.45
for neg - 0.40
for neutral - 0.97
with 4 iters
pos - 0.40 (how?)
neg - 0.44
neu - 0.97 

"""

#now lets remove the url for pos and neg and then train the model
#lets try this for positive first
train_data_pos=train_data[train_data['sentiment']=='positive']
import re
def remove_urls(text):
    text=re.sub(r'http\S+', '', text)
    text= re.sub(r'www\S+', '', text)
    return(text)
    
text= 'http//www.abc.comHow are you, hello world!'
    
train_data_pos['cleaned'] = train_data_pos.apply(lambda x: remove_urls(x['text']), axis =1)
train_data_pos['cleaned_selected_Text'] = train_data_pos.apply(lambda x: remove_urls(x['selected_text']), axis =1)

    
def Find_url(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 

train_data_pos['url_in_text']=train_data_pos.apply(lambda x : Find_url(x['cleaned']), axis=1)
train_data_pos['url_in_text'].value_counts()

train_data_pos['url_in_seleceted_text']=train_data_pos.apply(lambda x : Find_url(x['cleaned_selected_Text']), axis=1)
train_data_pos['url_in_seleceted_text'].value_counts()

"""with proprocessing for urls - 
for pos(2 iters) - 0.433 (why did it decrease?)  - This means preprocessing does not help"""