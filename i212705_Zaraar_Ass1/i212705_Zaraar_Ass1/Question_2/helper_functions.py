import re
import torch
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict

def read_data(path):
    temp=pd.read_csv(path)
    temp.drop(temp.columns[:-1],axis=1,inplace=True)    # keeping only the PlayerLines for the Training Part
    return temp['PlayerLine']

def sentence_tokenizer(sentences):
    temp=[]
    for i in sentences:
        temp.append(i.split())
    return temp

def remove_1(sentences):
    temp=[]
    i=0
    j=len(sentences)
    while(i<j):
        if "ACT" in sentences[i]:
            i=i+2
        elif "Exit" in sentences[i] :
            i=i+1
        elif "Exeunt" in sentences[i]:
            if len(sentences[i].split())<=3:
                i=i+1
            else:
                temp.append(sentences[i])
                i=i+1                
        else:
            temp.append(sentences[i])
            i=i+1
        if i== len(sentences):
            break
    return temp

def remove_punctuation(sentences):
    temp=[]
    for i in sentences:
        temp.append(re.sub(r'[^\w\s]', '', i))
    return temp

def remove_stopwords(sentences):
    temp=[]
    stopwords=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    for i in sentences:
        z=[]
        for j in i:
            if j not in stopwords:
                    z.append(j)
        temp.append(z)
    return temp
def preprocess(sentences):
    # 1. remove ACT and next line from the List
    # 2. Remove "Exit" or "Exuent" from the List
    # remove ",", ":" from list
    # tokenize each sentence
    sentences=remove_1(sentences)
    sentences=remove_punctuation(sentences)
    sentences=sentence_tokenizer(sentences)
    sentences=remove_stopwords(sentences)


    return sentences

def vector_dict(list_of_lists):
    # Create a dictionary to store word frequencies
    word_freq = defaultdict(int)
    
    # Count the frequency of words in list_of_lists
    for sentence in list_of_lists:
        for word in sentence:
            if word in word_freq.keys():
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    zz=0
    word_to_index={}
    for idx, (word, freq) in enumerate(word_freq.items()) :
            if freq > 7:
                word_to_index[word]=zz
                zz=zz+1
    
    # Add padding token
    word_to_index['PAD'] = 0
    filtered_list_of_lists = [[word for word in sentence if word in word_to_index.keys()] for sentence in list_of_lists]
    
    return word_to_index, filtered_list_of_lists
def inverse_dict(dictionary):
    new_dict={}
    for key,value in dictionary.items():
        new_dict[value]=key
    return new_dict
def convert_text_to_vectors(helper_dict,sentences):
    temp=[]
    for i in sentences:
        z=[]
        for j in i:
            z.append(helper_dict[j])
        temp.append(z)
    return temp


def convert_TOV_y(helper_dict,outputs):
    temp=[helper_dict[i] for i in outputs]
    return temp

def generate_ngram_sequence(sentences, seq_len=5, pad_token=0):
    input_sequences = []
    output_words = []
    
    for sentence in sentences:
        if len(sentence) >= seq_len:
            for i in range(1,len(sentence)):
                if i <=seq_len:
                    input_sequences.append(sentence[:i])
                    output_words.append(sentence[i])
                else:
                    input_sequences.append(sentence[i-seq_len:i])
                    output_words.append(sentence[i])

    
    return input_sequences, output_words



def pad(input_sequences,helper_dict):
    # Padding the input sequences to have the same length
    max_len = 5

    padded_sequences = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=helper_dict['PAD']) for seq in input_sequences])
    padded_sequences=torch.tensor(padded_sequences,dtype=torch.long)
    return padded_sequences
