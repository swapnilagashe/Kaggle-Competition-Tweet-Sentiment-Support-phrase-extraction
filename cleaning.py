#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:03:52 2020

@author: swapnillagashe
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import os
import numpy as np
import pandas as pd
from spellchecker import SpellChecker
from contractions import CONTRACTION_MAP
import re
os.chdir('/Users/DATA/Coding /Kaggle /tweet-sentiment-extraction')
stopwords=stopwords.words('English')

"""More things that can be implemented:
    1. Removing accented characters"""



def no_of_common_words(str1, str2):
    str1=str(str1)
    str2=str(str2)
    a=set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return(len(c))
    
#remove weird spaces
def rem_weird_spaces(text):
    text=text.strip()
    text= text.split()
    return " ".join(text)

#remove incorrect spellings
spell = SpellChecker()

def misspelled(list_of_words):
    return spell.unknown(list_of_words)

def spell_correct(list_of_words):
    #misspelled=spell.unknown(list_of_words) #list of words that might be mis spelled
    new_list=[spell.correction(word) for word in list_of_words]
    return new_list


#Expanding contractions (takes text and contraction mapo as arguments)
#text="ain't that a good thing don't you arent"

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def preprocess(string):
    string=str(string)
    string=rem_weird_spaces(string)
    string = ('').join(i for i in string if not i.isdigit()) #remove any numeric values in string
    string= string.lower()
    string=expand_contractions(string, CONTRACTION_MAP)
    string_tokens = word_tokenize(string) #tokenize as words
    tokens_without_sw= [word for word in string_tokens if not word in stopwords]
    tokens_without_punct=[word for word in tokens_without_sw if not word in punctuation ] #remove punctuations
    tokens_correctly_spelled=spell_correct(tokens_without_punct)
    string=(" ").join(tokens_correctly_spelled)
    print(string)
    return string

