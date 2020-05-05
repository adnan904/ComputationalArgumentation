#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import spacy
import os
import json
import timeit
from collections import Counter
import numpy as np


# In[ ]:


#############################################
# PLEASE SET TO CORRECT PATH BEFORE RUNNING #
#############################################
CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname("__file__")))
UNIFIED_DATA_FILE_PATH = f'{CURRENT_WORKING_DIR}/code/data/unified_data.json'
TRAIN_TEST_SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/code/data/train-test-split.csv'


#############################################
# Loading the spaCy Language Model
nlp = spacy.load("en_core_web_sm")


# In[ ]:


def get_train_test_split_dict_and_num_essays():
    """
    Reads the train-test-split.csv file and returns a dict {'essayid' : 'split'} and number of essays set as TRAIN
    :return: num_essays : number od essays SET to TRAIN in the train-test-split.csv
             train_test_split_dict: a dict of the form {'essayid' : 'split'}
    """
    with open(TRAIN_TEST_SPLIT_FILE_PATH, 'r') as train_file:
        train_test_split_dict = {}
        num_essays = 0
        file_content = train_file.read().split('\n')[1:-1]
        for line in file_content:
            essay_id = line.split('";')[0].split('"essay')[1]
            split = line.split(';"')[1].split('"')[0]
            train_test_split_dict[essay_id] = split
            if split == 'TRAIN':
                num_essays += 1
        return num_essays, train_test_split_dict


# In[ ]:


def tf_score(document):
    words_freq = {}
    for k, v in document.items():
        text = '. '.join(v)
        tokens = nlp(text)
        words = [token.text.lower() for token in tokens
             if token.is_stop is not True and token.is_punct is not True]
        word_count = Counter(words)
        tf_scores = {}
        for w in word_count:
            tf_scores[w] = word_count[w] / len(words)
        words_freq[k] = tf_scores
    return words_freq


# In[ ]:


def idf_score(document, tf_score_all_arguments):
    idf_scores = {} 
    for argument in tf_score_all_arguments:
        for k in tf_score_all_arguments[argument].keys():
            count = sum([k in tf_score_all_arguments[argument_unit] for argument_unit in tf_score_all_arguments])
            idf_scores[k] = np.log(len(document)/count)
    return idf_scores


# In[ ]:


def tf_idf_score(document, idf_score_all_arguments, tf_score_all_arguments):
    tf_idf_scores_document = {} #score of all argument units such as major-claim, claims, premises  
    for argument_unit in document:
        tf_idf_scores = {}
        for k in tf_score_all_arguments[argument_unit]:
            tf_idf_scores[k] = tf_score_all_arguments[argument_unit][k] * idf_score_all_arguments[k]
        tf_idf_scores_document[argument_unit] = tf_idf_scores
    return  tf_idf_scores_document


# In[ ]:


def pic_top_10_most_specific_words(tf_idf_scores):
    ten_most_common_words = {}
    for k, v in tf_idf_scores.items():
        ten_most_common_words[k] = Counter(v).most_common(10)
    return Counter(ten_most_common_words)
    


# In[37]:


def main():
    start = timeit.default_timer()
    # Initializing the Statistic Variables
    
    document = {}
    major_claims_text = []
    claims_text = []
    premises_text = []
    # paragraph_text = []
    
    # Number of Essays = Number of Essays in the train-test-split.csv file that have been SET 'TRAIN'
    # Getting the dict of train-test-split of the form {'essayid' : 'split'}
    num_of_essays, train_test_split_dict = get_train_test_split_dict_and_num_essays()

    with open(UNIFIED_DATA_FILE_PATH, 'r') as f:
        unified_file = json.load(f)
        for essay in unified_file:
            # We only need to compute for essays SET to 'TRAIN'
            if train_test_split_dict[essay['id']] == 'TRAIN':
                # Tokenizing using the nlp() of the spaCy library
                # Appending the text of the argument unit to a list
                for major_claim in essay['major_claim']:
                    major_claims_text.append(major_claim['text'])
                for claim in essay['claims']:
                    claims_text.append(claim['text'])
                for premise in essay['premises']:
                    premises_text.append(premise['text'])
                # for paragraph in essay['paragraphs']:
                #     paragraph_text.append(premise['text'])
                    
    document['major_claim'] = major_claims_text
    document['claims'] = claims_text
    document['premises'] = premises_text
    # document['paragraphs'] = premises_text
    
    tf_score_all_arguments = tf_score(document)
    idf_score_all_arguments = idf_score(document, tf_score_all_arguments)
    tf_idf_scores = tf_idf_score(document, idf_score_all_arguments, tf_score_all_arguments)
    
    top_10_most_specific_words = pic_top_10_most_specific_words(tf_idf_scores)
        
    for element in top_10_most_specific_words:
        print("\n10 most specific words in : '{}' \n ".format(element))
        for k, v in top_10_most_specific_words[element]:
            print("\n{} : TF-IDF: {}".format(k, v))
    
    stop = timeit.default_timer()
    print('\nTime: ', stop - start)


# In[38]:


if __name__ == '__main__':
    main()



