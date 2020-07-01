import json
import random

import pandas as pd
import os
import csv
import numpy as np

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/../data/essay_prompt_corpus.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/train-test-split.csv'
PRED_FILE_PATH= f'{CURRENT_WORKING_DIR}/../data/predictions.json'


def get_train_test_split_essays(corpus, split_scheme) -> (list, list):
    """
    :param corpus: unified data file with all the essays
    :param split_scheme: train_test_split scheme file
    :rtype: list, list
    :return: pandas dataframe of train, test split essay id, text, prompt
    """

    train_test_split_dict = {}
    test_df = pd.DataFrame(columns=['id', 'text', 'prompt'])
    train_df = pd.DataFrame(columns=['id', 'text', 'prompt'])

    # create a dict of the type: {essay_id: Tag},  where Tag = 'TRAIN' or 'TEST'
    for row in split_scheme:
        if len(row) > 0:
            essay_id = int(row[0].split('essay')[1])
            train_test_split_dict[essay_id] = row[1]

    # extract essays that match the test_train_split scheme
    for essay in corpus:
        if train_test_split_dict[int(essay['id'])] == 'TRAIN':
            train_df = train_df.append({'id': essay['id'], 'text': essay['text'], 'prompt': essay['prompt']},
                                       ignore_index=True)
        else:
            test_df = test_df.append({'id': essay['id'], 'text': essay['text'], 'prompt': essay['prompt']},
                                       ignore_index=True)
    train_df.sort_values('id', inplace=True)
    test_df.sort_values('id', inplace=True)
    return train_df, test_df



class Prediction(object):
    id = ""
    prompt = ""

    def __init__(self, essay_id, prompt):
        self.id = str(essay_id)
        self.prompt = str(prompt)


if __name__ == "__main__":
    json_corpus = json.load(open(CORPUS_PATH, encoding='utf-8'))

    # Read train_test_split and get essays from the unified corpus based on the split
    with open(SPLIT_FILE_PATH, newline='', encoding='utf-8') as csvfile:
        train_test_split_file = csv.reader(csvfile, delimiter=';')
        next(train_test_split_file, None)
        train_essays, test_essays = get_train_test_split_essays(json_corpus, train_test_split_file)
        train_X = train_essays['text']
        train_y = train_essays['prompt']
        test_X = list(test_essays['text'])
        test_y = list(test_essays['prompt'])
        test_id = list(test_essays['id'])

        predictions = []

        for text, id in zip(test_X,test_id):
            sentences = text.split('.')  # so we get the sententences
            num1 = random.randrange(len(sentences)-1)
            num2 = random.randrange(len(sentences)-1)
            while num2 == num1:
                num2 = random.randrange(len(sentences)-1)
            random_result_prompt = sentences[num1] + ". " + sentences[num2]
            predictions.append(Prediction(id, random_result_prompt))

        json_dump = json.dumps([obj.__dict__ for obj in predictions], indent=4, ensure_ascii=False)
        with open(PRED_FILE_PATH, "w", encoding='utf-8') as outfile:
            outfile.write(json_dump)

        print("Successfully created prediction file in '" + PRED_FILE_PATH + "'.")

