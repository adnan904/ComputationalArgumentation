import json
import os
import pandas as pd
from convert_to_bio import convert_to_bio  # make sure 'convert_to_bio.py' and the current file are in the same folder

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/data/essay_corpus.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/data/train-test-split.csv'


def train_test_split(corpus, split_scheme) -> (list, list):
    """
    :param corpus: unified data file with all the essays
    :param split_scheme: train_test_slit scheme file as Panda DF
    :rtype: list, list
    :return: tuple of list of train, test split
    """

    # split essays into training and testing
    train_data = split_scheme.loc[split_scheme['SET'] == 'TRAIN']
    test_data = split_scheme.loc[split_scheme['SET'] == 'TEST']

    # extract essays that match the test_train_split scheme
    all_test_data = search_and_extract_data(corpus, test_data)
    all_train_data = search_and_extract_data(corpus, train_data)

    return all_test_data, all_train_data


def search_and_extract_data(all_json_data, split_type_data) -> list:
    """
    :param split_type_data: train or test essays
    :param all_json_data:  unified data file with all the essays
    :rtype: list
    :return: list of extracted essays
    """
    extracted_data = []
    for essay_id in split_type_data['ID']:
        for essay in all_json_data:
            if int(essay_id.split('essay')[1]) == essay['id']:
                extracted_data = extracted_data + [essay]
    return extracted_data


if __name__ == "__main__":
    json_corpus = json.load(open(CORPUS_PATH))

    # Read train_test_split as panda data-frame
    train_test_split_file = pd.read_csv(SPLIT_FILE_PATH, sep=';')
    test, train = train_test_split(json_corpus, train_test_split_file)
    train_bio = convert_to_bio(train)
    test_bio = convert_to_bio(test)
    with open(f'{CURRENT_WORKING_DIR}/data/train_BIO.txt', "w") as train_file:
        train_file.write(''.join(train_bio))
    with open(f'{CURRENT_WORKING_DIR}/data/test_BIO.txt', "w") as test_file:
        test_file.write(''.join(test_bio))
    print("Successfully created train and test Bio in '/data/'.")
