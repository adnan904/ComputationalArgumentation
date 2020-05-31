import json
import os
import csv
from convert_to_bio import convert_to_bio  # make sure 'convert_to_bio.py' and the current file are in the same folder

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/data/essay_corpus.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/data/train-test-split.csv'


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


if __name__ == "__main__":
    json_corpus = json.load(open(CORPUS_PATH))

    # Read train_test_split and get essays from the unified corpus based on the split
    with open(SPLIT_FILE_PATH, newline='') as csvfile:
        train_test_split_file = csv.reader(csvfile, delimiter=';')
        next(train_test_split_file, None)
        test_essays, train_essays = get_train_test_split_essays(json_corpus, train_test_split_file)

    # Getting the BIO tags for each essay in the train-test split
    train_bio = convert_to_bio(train_essays)
    test_bio = convert_to_bio(test_essays)

    # Writing the BIO tags into the Train and Test BIO text files
    with open(f'{CURRENT_WORKING_DIR}/data/train_BIO.txt', "w") \
            as train_file, open(f'{CURRENT_WORKING_DIR}/data/test_BIO.txt', "w") as test_file:
        train_file.write(''.join(train_bio))
        test_file.write(''.join(test_bio))

    print("\nSuccessfully created train and test Bio in '/data/'.")

