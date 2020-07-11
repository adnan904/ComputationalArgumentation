import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import json
import os
import sys
import wget
import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
nltk.download('punkt')  # one time execution
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

nltk.download('stopwords')  # one time execution
from nltk.corpus import stopwords
import zipfile
import glob

stop_words = stopwords.words('english')
CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/../data/essay_prompt_corpus.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/train-test-split.csv'
PRED_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/predictions.json'


# download pretrained GloVe word embeddings

# create a progress bar method which is invoked automatically from wget
def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


# Downloading only once
if not os.path.isfile('glove.6B.zip'):
    wget.download('http://nlp.stanford.edu/data/glove.6B.zip', bar=bar_progress)

if not os.path.isfile(f'{CURRENT_WORKING_DIR}/../data/glove.6B.100d.txt'):
    glove = glob.glob('glove*.zip', recursive=True)[0]
    with zipfile.ZipFile(glove, 'r') as zip_ref:
        zip_ref.extractall(f'{CURRENT_WORKING_DIR}/../data/')


def create_sim_matrix(sim_matrix, length_of_sentences, essay_id):
    target_essay_sentence_vector = vectorized_essays[essay_id]
    for m in range(length_of_sentences):
        for j in range(length_of_sentences):
            if m != j:
                sim_matrix[m][j] = cosine_similarity(target_essay_sentence_vector[m].reshape(1, 100),
                                                     target_essay_sentence_vector[j].reshape(1, 100))[0, 0]
    return sim_matrix


def tokenize_essay_sentences(essay_id, text):
    tokenized_essays = {essay_id: sent_tokenize(text)}
    return tokenized_essays


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5", "-t5", required=False, help="Get Abstractive summary from top page ranked sentences",
                         type=str, default=False, metavar="t5")
    args = parser.parse_args()
    if args.t5 in ['True', 'true', 'T', 't']:
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        device = torch.device('cpu')
    csv_df = pd.read_csv(SPLIT_FILE_PATH, delimiter=';')
    csv_df = csv_df.loc[csv_df['SET'] == 'TEST']
    test_essays_id_strings = csv_df['ID']
    test_essays_ids = [e_id.split('essay')[1] for e_id in test_essays_id_strings]

    df = pd.read_json(CORPUS_PATH)

    # select only test rows from corpus
    df = df.loc[df['id'].isin(test_essays_ids)]
    df.sort_values('id', inplace=True)

    essay_sentences = [tokenize_essay_sentences(essay_id, text) for essay_id, text in zip(df['id'], df['text'])]
    clean_essay_list = []
    for essay in essay_sentences:
        for k, v in essay.items():
            clean_sentences = [remove_stopwords(s.split()) for s in v]
            clean_essay = {k: clean_sentences}
            clean_essay_list.append(clean_essay)

    # Extract word vectors
    word_embeddings = {}
    with open(f'{CURRENT_WORKING_DIR}/../data/glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs

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

    # find similarities between the sentences of each essay.
    output = []
    for e in essay_sentences:
        for k, v in e.items():
            sim_mat = np.zeros([len(v), len(v)])
            sm = create_sim_matrix(sim_mat, len(v), k)
            nx_graph = nx.from_numpy_array(sm)
            scores = nx.pagerank(nx_graph)
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(v)), reverse=True)
            prompt = ranked_sentences[0][1] + ' ' + ranked_sentences[1][1]
            if args.t5 in ['True', 'true', 'T', 't']:
                t5_prepared_Text = "summarize: " + prompt.lower()
                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
                # summmarize
                summary_ids = model.generate(tokenized_text)
                prompt = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            # Generate summary
            essay_obj = {'id': k, 'prompt': prompt}
            output.append(essay_obj)

    json_dump = json.dumps(output, indent=4, ensure_ascii=False)
    with open(PRED_FILE_PATH, "w", encoding='utf-8') as outfile:
        outfile.write(json_dump)
    print("Successfully created the predictions")
