import spacy
import os
import json
import timeit
from collections import Counter

#############################################
# PLEASE SET TO CORRECT PATH BEFORE RUNNING #
#############################################
CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname("__file__")))
UNIFIED_DATA_FILE_PATH = f'{CURRENT_WORKING_DIR}/code/data/unified_data.json'
TRAIN_TEST_SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/code/data/train-test-split.csv'


#############################################
# Loading the spaCy Language Model
nlp = spacy.load("en_core_web_sm")


def get_train_test_split_dict_and_num_essays():
    """
    Reads the train-test-split.csv file and returns a dict {essayid : split} and number of essays set as TRAIN
    :return: num_essays : number od essays SET to TRAIN in the train-test-split.csv
             train_test_split_dict: a dict of the form {essayid : split}
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


def get_most_specific_words(text_list: list) -> list:
    """
    Gets a list of lowercase sentences. It performs the following on them:
        - Joins the sentences to a single text string
        - Tokenize's the text using the spaCy library
        - Filters the tokens that are not stop-words(e.g. and, a, is, the, which etc) or punctuation
        - Calculates the frequency of all the remaining words
        - Filters the 10 most common words
        - Returns a list of tuples of the form (word, frequency)
    :param text_list: a list of lowercase sentences
    :return: a list of 10 most specific tuples of the form (word, frequency)
    """
    text = '. '.join(text_list)
    tokens = nlp(text)
    words = [token.text for token in tokens
             if token.is_stop is not True and token.is_punct is not True]
    words_freq = Counter(words)
    specific_words = words_freq.most_common(10)
    return specific_words


def main():
    start = timeit.default_timer()
    # Initializing the Statistic Variables
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
    major_claims_text = []
    claims_text = []
    premises_text = []
    # Number of Essays = Number of Essays in the train-test-split.csv file that have been SET 'TRAIN'
    # Getting the list of IDs of train-split from train-test-split.csv
    num_of_essays, train_test_split_dict = get_train_test_split_dict_and_num_essays()

    with open(UNIFIED_DATA_FILE_PATH, 'r') as f:
        unified_file = json.load(f)
        for essay in unified_file:
            # We only need to compute for essays SET to 'TRAIN'
            if train_test_split_dict[essay['id']] == 'TRAIN':
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
                    major_claims_text.append(major_claim['text'])
                for claim in essay['claims']:
                    num_of_tokens_in_claims += len(nlp(claim['text']))
                    claims_text.append(claim['text'])
                for premise in essay['premises']:
                    num_of_tokens_in_premises += len(nlp(premise['text']))
                    premises_text.append(premise['text'])
        # Calculating the avg. number of tokens in major_claims, claims, and premises
        avg_num_of_tokens_in_major_claims = num_of_tokens_in_major_claims / num_of_major_claims
        avg_num_of_tokens_in_claims = num_of_tokens_in_claims / num_of_claims
        avg_num_of_tokens_in_premises = num_of_tokens_in_premises / num_of_premises

        # Calculating the 10 most specific words in major_claims, claims, and premises
        major_claims_specific_words = get_most_specific_words(major_claims_text)
        claims_specific_words = get_most_specific_words(claims_text)
        premises_specific_words = get_most_specific_words(premises_text)

    print("The Preliminary Statistics are:")
    print("Number of essays: {}".format(num_of_essays))
    print("Number of paragraphs: {}".format(num_of_paragraphs))
    print("Number of sentences: {}".format(num_of_sentences))
    print("Number of tokens: {}".format(num_of_tokens))
    print("Number of major claims: {}".format(num_of_major_claims))
    print("Number of claims: {}".format(num_of_claims))
    print("Number of premises: {}".format(num_of_premises))
    print("Number of essays with confirmation bias: {}".format(num_of_essays_with_conf_bias))
    print("Number of essays without confirmation bias: {}".format(num_of_essays_without_conf_bias))
    print("Number of sufficient paragraphs: {}".format(num_of_suff_paras))
    print("Number of insufficient paragraphs: {}".format(num_of_insuff_paras))
    print("Average number of tokens in major claims: {}".format(avg_num_of_tokens_in_major_claims))
    print("Average number of tokens in claims: {}".format(avg_num_of_tokens_in_claims))
    print("Average number of tokens in premises: {}".format(avg_num_of_tokens_in_premises))
    print("\n10 most specific words in major claims:")
    for i, word in enumerate(major_claims_specific_words):
        print("{}) '{}' -- {} times".format(i+1, word[0], word[1]))
    print("\n10 most specific words in claims:")
    for i, word in enumerate(claims_specific_words):
        print("{}) '{}' -- {} times".format(i+1, word[0], word[1]))
    print("\n10 most specific words in premises:")
    for i, word in enumerate(premises_specific_words):
        print("{}) '{}' -- {} times".format(i+1, word[0], word[1]))

    stop = timeit.default_timer()
    print('\nTime: ', stop - start)


if __name__ == '__main__':
    main()