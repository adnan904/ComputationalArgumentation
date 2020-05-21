import json
import os
import pandas as pd

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
    train_json_dump = json.dumps(train, indent=4, ensure_ascii=False)
    test_json_dump = json.dumps(test, indent=4, ensure_ascii=False)
    with open(f'{CURRENT_WORKING_DIR}/data/train_split.json', "w") as train_file:
        train_file.write(train_json_dump)
    with open(f'{CURRENT_WORKING_DIR}/data/test_split.json', "w") as test_file:
        test_file.write(test_json_dump)
    print(len(train))
    print(len(test))
    print("Successfully created train and test data in '/data/'.")
