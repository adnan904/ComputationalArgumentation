import spacy
import os
import json
import timeit
from collections import Counter, defaultdict
import numpy as np

#############################################
# PLEASE SET TO CORRECT PATH BEFORE RUNNING #
#############################################
CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
UNIFIED_DATA_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/unified_data.json'
TRAIN_TEST_SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/train-test-split.csv'


#############################################
# Loading the spaCy Language Model
nlp = spacy.load("en_core_web_sm")


def get_train_test_split_dict_and_num_essays():
    """
    Reads the train-test-split.csv file and returns a dict {'essayid' : 'split'} and number of essays set as TRAIN
    :return: num_essays : number od essays SET to TRAIN in the train-test-split.csv
             train_test_split_dict: a dict of the form {'essayid' : 'split'}
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


def get_token_count_dict(all_essays_tokens: dict):
    """
        splits essay tokens
        :param all_essays_tokens, a dict of tokens of all essays
        :return: dict with essay tokens
        """
    all_essay_tokens = defaultdict(lambda: defaultdict(lambda: 0))
    for essay_id, tokens in all_essays_tokens.items():
        words = [token.text.lower() for token in tokens
                 if token.is_stop is not True and token.is_punct is not True]
        for word in words:
            all_essay_tokens[essay_id][word] += 1
    return all_essay_tokens


def tf_score(all_argument_units_tokens: dict):
    """
    Computes the TF score for each word in argument units
    tf(t) = count of t in argument_unit_text / number of words in argument_unit_text

    :param all_argument_units_tokens: tokens list of all 3 argument units in the train-split essays placed in a dict
    :return dict of all words with TF scores:
    """
    words_freq = {}
    for k, tokens in all_argument_units_tokens.items():
        # nlp function returns some weird words like 'educatio' etc which do not exist anywhere in the text
        # and their IDF score cannot be computed
        words = [token.text.lower() for token in tokens
                 if token.is_stop is not True and token.is_punct is not True]
        word_count = Counter(words)
        tf_scores = {}
        for w in word_count:
            tf_scores[w] = word_count[w] / len(tokens)
        words_freq[k] = tf_scores
    return words_freq


def idf(all_essays_tokens: dict, tf_score_all_arguments: dict):
    """
    Computes the IDF score for each word in  claims, major_claims, premises of all essays
    idf(t) = log_e(Total number of documents / Number of documents with term t in it)

    :param all_essays_tokens: dict of tokens of all train-split essays
    :param tf_score_all_arguments: a dict of tf scores for each word in claims, major_claims, premises
    :return dict of all words with IDF scores: 
    """
    all_idf_score = {}
    all_essay_tokens = get_token_count_dict(all_essays_tokens)
    # For each word appearing in the text of all claims, major claims , and premises - we check in how many essay texts
    # this word occurs to calculate the IDF-score of that word based on the above formula
    for argument_unit, words in tf_score_all_arguments.items():
        for word, _ in words.items():
            if word not in all_idf_score:
                count = sum([word in all_essay_tokens[essay_id] for essay_id in all_essay_tokens.keys()])
                # In order to skip the weird words returned by nlp() which do not appear anywhere in the text(count = 0)
                if count > 0:
                    all_idf_score[word] = np.log(len(all_essays_tokens) / count)
    return all_idf_score


def tf_idf(tf_score_all_arguments, idf_scores):
    """
    Computes the TF-IDF score for each word in argument units: claims|major_claims|premises
    tf-idf(t) = tf(t) * idf(t)

    :param tf_score_all_arguments: tf scores of all words in the argument units: claims, major-claims, premises
    :param idf_scores: IDF value of the each word in the argument units: claims, major-claims, premises
    :return dict with all words and their TF-IDF scores for current argument unit: 
    """
    
    tf_idf_scores = {}
    for argument_unit, words in tf_score_all_arguments.items():
        words_dict = {}
        for word, term_freq_score in words.items():
            # In order to skip the weird words returned by nlp() which do not appear anywhere in the text(count = 0)
            if idf_scores.get(word):
                words_dict[word] = term_freq_score * idf_scores[word]
        tf_idf_scores[argument_unit] = words_dict
    return tf_idf_scores


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
    all_essays_tokens = {}
    all_argument_units_tokens = {}
    major_claims_text = []
    claims_text = []
    premises_text = []

    # Number of Essays = Number of Essays in the train-test-split.csv file that have been SET 'TRAIN'
    # Getting the dict of train-test-split of the form {'essayid' : 'split'}
    num_of_essays, train_test_split_dict = get_train_test_split_dict_and_num_essays()

    with open(UNIFIED_DATA_FILE_PATH, 'r') as f:
        unified_file = json.load(f)
        for essay in unified_file:
            # We only need to compute for essays SET to 'TRAIN'
            if train_test_split_dict[essay['id']] == 'TRAIN':
                # Tokenizing the text for the essay using the spaCy library
                text = nlp(essay['text'])
                all_essays_tokens[essay['id']] = text
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
                # Appending the text of the argument unit to a list
                for major_claim in essay['major_claim']:
                    major_claims_text.append(major_claim['text'])
                for claim in essay['claims']:
                    claims_text.append(claim['text'])
                for premise in essay['premises']:
                    premises_text.append(premise['text'])
        # Generating the argument_units tokens using spaCy library's nlp()
        major_claims_tokens = nlp(' '.join(major_claims_text))
        claims_tokens = nlp(' '.join(claims_text))
        premises_tokens = nlp(' '.join(premises_text))
        all_argument_units_tokens['major_claim'] = major_claims_tokens
        all_argument_units_tokens['claims'] = claims_tokens
        all_argument_units_tokens['premises'] = premises_tokens

        # Calculating the avg. number of tokens in major_claims, claims, and premises
        avg_num_of_tokens_in_major_claims = len(major_claims_tokens) / num_of_major_claims
        avg_num_of_tokens_in_claims = len(claims_tokens) / num_of_claims
        avg_num_of_tokens_in_premises = len(premises_tokens) / num_of_premises

        # Calculating tf_score for all 3 argument units: major claims | claims | premises
        tf_score_all_arguments = tf_score(all_argument_units_tokens)

        # calculate IDF score for each word in the whole text of all claims, major-claims and premises
        idf_scores = idf(all_essays_tokens, tf_score_all_arguments)

        # calculate TF-IDF score for all words in major claims | claims | premises
        all_tf_idf_scores = tf_idf(tf_score_all_arguments, idf_scores)

        # get the top 10 scores for each 
        major_claims_ten_specific_words = Counter(all_tf_idf_scores['major_claim']).most_common(10)
        claims_ten_specific_words = Counter(all_tf_idf_scores['claims']).most_common(10)
        premises_ten_specific_words = Counter(all_tf_idf_scores['premises']).most_common(10)

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
    for i, word in enumerate(major_claims_ten_specific_words):
        print("{}) '{}' -- TF-IDF score: {}".format(i+1, word[0], word[1]))
    print("\n10 most specific words in claims:")
    for i, word in enumerate(claims_ten_specific_words):
        print("{}) '{}' -- TF-IDF score: {}".format(i+1, word[0], word[1]))
    print("\n10 most specific words in premises:")
    for i, word in enumerate(premises_ten_specific_words):
        print("{}) '{}' -- TF-IDF score: {}".format(i+1, word[0], word[1]))

    stop = timeit.default_timer()
    print('\nTime: ', stop - start)


if __name__ == '__main__':
    main()
