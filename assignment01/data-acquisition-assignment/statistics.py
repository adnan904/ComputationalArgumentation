import spacy
import pandas as pd
import os
import timeit

#############################################
# PLEASE SET TO CORRECT PATH BEFORE RUNNING #
#############################################
CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname("__file__")))
UNIFIED_DATA_FILE_PATH = f'{CURRENT_WORKING_DIR}/code/data/unified_data.json'
TRAIN_TEST_SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/code/data/train-test-split.csv'
#############################################


def get_train_split() -> list:
    """
    Reads the train-test-split.csv file and results a list of ids of all the essays that have been SET for 'TRAIN'
    :return: train_ids : integer IDs of the essays SET as 'TRAIN'
    """
    with open(TRAIN_TEST_SPLIT_FILE_PATH, 'r') as train_file:
        train_ids = []
        file_content = train_file.read().split('\n')[1:]
        for line in file_content:
            if 'TRAIN' in line:
                train_ids.append(int(line.split('";')[0].split('"essay')[1]))
        return train_ids


def main():
    start = timeit.default_timer()
    # Initializing all the Statistic Variables
    num_of_essays = 0
    num_of_paragraphs = 0
    num_of_sentences = 0
    num_of_tokens = 0
    num_of_major_claims = 0
    num_of_claims = 0
    num_of_premises = 0
    num_of_essays_with_conf_bias = 0
    num_of_essays_without_conf_bias = 0
    num_of_suff_paras = 0
    num_of_insuff_paras = 0
    num_of_tokens_in_major_claims = 0
    num_of_tokens_in_claims = 0
    num_of_tokens_in_premises = 0
    avg_num_of_tokens_in_major_claims = 0
    avg_num_of_tokens_in_claims = 0
    avg_num_of_tokens_in_premises = 0
    ten_most_spec_words_in_major_claims = 0
    ten_most_spec_words_in_claims = 0
    ten_most_spec_words_in_premises = 0
    # Loading the Spacy Language Model
    nlp = spacy.load("en_core_web_sm")
    # Getting the list of IDs of train-split from train-test-split.csv
    train_split_essay_ids = get_train_split()
    # Number of Essays = Number of Essays in the train-test-split.csv file that have been SET 'TRAIN'
    num_of_essays = len(train_split_essay_ids)
    with open(UNIFIED_DATA_FILE_PATH, 'r') as data_file:
        # Creating a Pandas Dataframe from the unified data json file
        df = pd.read_json(data_file)
        for essay in df.values:
            # We only need to compute for essays SET to 'TRAIN'
            if essay[0] in train_split_essay_ids:
                # Tokenizing the text for the essay using the spaCy library
                text = nlp(essay[2])
                num_of_paragraphs += len(essay[6])
                # Using the spaCy library for calculating Sentences in the text
                num_of_sentences += len(list(text.sents))
                num_of_tokens += len(text)
                num_of_major_claims += len(essay[3])
                num_of_claims += len(essay[4])
                num_of_premises += len(essay[5])
                if essay[7]:
                    num_of_essays_with_conf_bias += 1
                else:
                    num_of_essays_without_conf_bias += 1
                for para in essay[6]:
                    if para['sufficient']:
                        num_of_suff_paras += 1
                    else:
                        num_of_insuff_paras += 1
                # Tokenizing using the nlp() of the spaCy library
                for major_claim in essay[3]:
                    num_of_tokens_in_major_claims += len(nlp(major_claim['text']))
                for claim in essay[4]:
                    num_of_tokens_in_claims += len(nlp(claim['text']))
                for premise in essay[5]:
                    num_of_tokens_in_premises += len(nlp(premise['text']))
        avg_num_of_tokens_in_major_claims = num_of_tokens_in_major_claims / num_of_major_claims
        avg_num_of_tokens_in_claims = num_of_tokens_in_claims / num_of_claims
        avg_num_of_tokens_in_premises = num_of_tokens_in_premises / num_of_premises

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
