#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # one time execution
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

csv_df = pd.read_csv('train-test-split.csv', delimiter=';')

csv_df = csv_df.loc[csv_df['SET'] == 'TEST']
csv_df.head()

test_essays_id_strings = csv_df['ID']
test_essays_ids = [e_id.split('essay')[1] for e_id in test_essays_id_strings]

df = pd.read_json('essay_prompt_corpus.json')

# select only test rows from corpus
df = df.loc[df['id'].isin(test_essays_ids)]


def tokenize_essay_sentences(essay_id, text):
    tokenized_essays = {essay_id: sent_tokenize(text)}
    return tokenized_essays


essay_sentences = [tokenize_essay_sentences(essay_id, text) for essay_id, text in zip(df['id'], df['text'])]

nltk.download('stopwords')  # one time execution

from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


clean_essay_list = []
for essay in essay_sentences:
    for k, v in essay.items():
        clean_sentences = [remove_stopwords(s.split()) for s in v]
        clean_essay = {k: clean_sentences}
        clean_essay_list.append(clean_essay)

# download pretrained GloVe word embeddings
# ! wget http://nlp.stanford.edu/data/glove.6B.zip


# ! unzip glove*.zip


# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

vectorized_essays = {}
for s in clean_essay_list:
    for key, val in s.items():
        sentence_vectors = []
        for i in val:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)
        vectorized_essays[key] = sentence_vectors


def create_sim_matrix(sim_matrix, length_of_sentences, essay_id):
    target_essay_sentence_vector = vectorized_essays[essay_id]
    for m in range(length_of_sentences):
        for j in range(length_of_sentences):
            if m != j:
                sim_matrix[m][j] = cosine_similarity(target_essay_sentence_vector[m].reshape(1, 100),
                                                     target_essay_sentence_vector[j].reshape(1, 100))[0, 0]
    return sim_matrix


# find similarities between the sentences of each essay.
output = []
for e in essay_sentences:
    for k, v in e.items():
        sim_mat = np.zeros([len(v), len(v)])
        sm = create_sim_matrix(sim_mat, len(v), k)
        nx_graph = nx.from_numpy_array(sm)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(v)), reverse=True)
        # Generate summary
        essay_obj = {'id': k, 'prompt': ranked_sentences[0][1]}
        output.append(essay_obj)

import json

json_dump = json.dumps(output, indent=4, ensure_ascii=False)
with open('predictions.json', "w", encoding='utf-8') as outfile:
    outfile.write(json_dump)
