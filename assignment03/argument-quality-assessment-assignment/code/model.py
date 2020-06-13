import json
import os
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/../data/essay_corpus.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/train-test-split.csv'
PRED_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/predictions.json'

def get_train_test_split_essays(corpus, split_scheme) -> (list, list):
    """
    :param corpus: unified data file with all the essays
    :param split_scheme: train_test_split scheme file
    :rtype: list, list
    :return: lists of train, test split essay data
    """

    train_test_split_dict = {}
    all_test_data = []
    all_train_data = []

    # create a dict of the type: {essay_id: Tag},  where Tag = 'TRAIN' or 'TEST'
    for row in split_scheme:
        if len(row) > 0:
            essay_id = int(row[0].split('essay')[1])
            train_test_split_dict[essay_id] = row[1]

    # extract essays that match the test_train_split scheme
    for essay in corpus:
        if train_test_split_dict[essay['id']] == 'TRAIN':
            all_train_data.append(essay)
        else:
            all_test_data.append(essay)

    return all_test_data, all_train_data

def get_train_test_split():

    # Read train_test_split and get essays from the unified corpus based on the split
    with open(SPLIT_FILE_PATH, newline='') as csvfile:
        train_test_split_file = csv.reader(csvfile, delimiter=';')
        next(train_test_split_file, None)
        test_essays, train_essays = get_train_test_split_essays(json_corpus, train_test_split_file)
        print('TRAIN essays: ' + str(len(train_essays)))
        print('TEST essays: ' + str(len(test_essays)))

        return train_essays, test_essays


# Object based on the sample.json file
class Prediction(object):
    id = ""
    confirmation_bias = False  # not biased until proven guilty

    # The class "constructor" - It's actually an initializer
    def __init__(self, essay_id, confirmation_bias):
        self.id = str(essay_id)        
        self.confirmation_bias = confirmation_bias

def create_prediction_file(prediction):
     # write
    json_dump = json.dumps([element.__dict__ for element in prediction], indent=4, ensure_ascii=False)
    with open(PRED_FILE_PATH, "w") as outfile:
        outfile.write(json_dump)
    print("Successfully created prediction file '" + PRED_FILE_PATH + "'.")


def all_true_prediction(train_essays, test_essays):
    # predicts for all test_essays confirmation_bias=True
    predictions = []

    for essay in test_essays:
        predictions.append(Prediction(essay['id'],  True))
    return predictions

def get_texts(essays):
    """
    :param tokens: all the tokens in lowercase from a BIO file
    :return: a list of paragraphs from from the file
    """
    text_list = []
    for essay in essays:
        text_list.append(essay['text'])
    return text_list

if __name__ == "__main__":
    json_corpus = json.load(open(CORPUS_PATH, encoding='utf-8'))

    train_essays, test_essays = get_train_test_split()
    train_essays = sorted(train_essays, key=lambda x: x["id"])
    test_essays  = sorted(test_essays, key=lambda x: x["id"])


    predictions = all_true_prediction(train_essays, test_essays)

    create_prediction_file(predictions)

