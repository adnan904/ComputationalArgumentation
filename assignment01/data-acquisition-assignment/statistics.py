import spacy
import os
import json
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
                train_ids.append(line.split('";')[0].split('"essay')[1])
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

    with open(UNIFIED_DATA_FILE_PATH, 'r') as f:
        unified_file = json.load(f)
        for essay in unified_file:
            # We only need to compute for essays SET to 'TRAIN'
            if essay['id'] in train_split_essay_ids:
                # Tokenizing the text for the essay using the spaCy library
                text = nlp(essay['text'])
                num_of_paragraphs += len(essay['paragraphs'])
                # Using the spaCy library for calculating Sentences in the text
                num_of_sentences += len(list(text.sents))
                num_of_tokens += len(text)
                num_of_major_claims += len(essay['major_claim'])
                num_of_claims += len(essay['claims'])
                num_of_premises += len(essay['premises'])
                if essay['confirmation_bias']:
                    num_of_essays_with_conf_bias += 1
                else:
                    num_of_essays_without_conf_bias += 1
                for para in essay['paragraphs']:
                    if para['sufficient']:
                        num_of_suff_paras += 1
                    else:
                        num_of_insuff_paras += 1
                # Tokenizing using the nlp() of the spaCy library
                for major_claim in essay['major_claim']:
                    num_of_tokens_in_major_claims += len(nlp(major_claim['text']))
                for claim in essay['claims']:
                    num_of_tokens_in_claims += len(nlp(claim['text']))
                for premise in essay['premises']:
                    num_of_tokens_in_premises += len(nlp(premise['text']))
        avg_num_of_tokens_in_major_claims = num_of_tokens_in_major_claims / num_of_major_claims
        avg_num_of_tokens_in_claims = num_of_tokens_in_claims / num_of_claims
        avg_num_of_tokens_in_premises = num_of_tokens_in_premises / num_of_premises

    print("The Preliminary Statistics are:")
    print("Number of essays: " + str(num_of_essays))
    print("Number of paragraphs: " + str(num_of_paragraphs))
    print("Number of sentences: " + str(num_of_sentences))
    print("Number of tokens: " + str(num_of_tokens))
    print("Number of major claims: " + str(num_of_major_claims))
    print("Number of claims: " + str(num_of_claims))
    print("Number of premises: " + str(num_of_premises))
    print("Number of essays with confirmation bias: " + str(num_of_essays_with_conf_bias))
    print("Number of essays without confirmation bias: " + str(num_of_essays_without_conf_bias))
    print("Number of sufficient paragraphs: " + str(num_of_suff_paras))
    print("Number of insufficient paragraphs: " + str(num_of_insuff_paras))
    print("Average number of tokens in major claims: " + str(avg_num_of_tokens_in_major_claims))
    print("Average number of tokens in claims: " + str(avg_num_of_tokens_in_claims))
    print("Average number of tokens in premises: " + str(avg_num_of_tokens_in_premises))
    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
