import spacy
import os
import json
import timeit
from collections import Counter
import numpy as np

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


def get_most_common_words(text_list: list):
    """
    Gets a list of  sentences. It performs the following on them:
        - Joins the sentences to a single text string
        - Tokenize's the text using the spaCy library
        - Filters the tokens that are not stop-words(e.g. and, a, is, the, which etc) or punctuation
        - converts the tokens to lowercase so that e.g people and People are not counted separately
        - Calculates the frequency of all the remaining words
        - Filters the most common words
        - Returns a list of tuples of the form (word, frequency) and a set of unique words occurring in the text
    :param text_list: a list of lowercase sentences
    :return: a list most common tuples of the form (word, frequency), and a set of unique words occurring in the text
    """
    text = '. '.join(text_list)
    tokens = nlp(text)
    words = [token.text.lower() for token in tokens
             if token.is_stop is not True and token.is_punct is not True]
    words_set = set(words)
    words_freq = Counter(words)
    common_words = words_freq.most_common(len(words))
    return common_words, words_set


def get_specific_words(counter, set1, set2):
    count = 0
    ten_most_specific_tuple = []

    for word in counter:
        if count >= 10:
            break
        if word[0] not in set1 and word[0] not in set2:
            ten_most_specific_tuple.append(word)
            count += 1
    return ten_most_specific_tuple


def get_all_specific_words(IDF, major_claims_if_idf_scores, claims_if_idf_scores, premises_if_idf_scores):
    """
    Creates dict with highest if_idf scores for al argument units

    :param IDF: dict with all words in the text of all essays
    :param major_claims_if_idf_scores: dict with idf-scores for major claims
    :param claims_if_idf_scores: dict with idf-scores for claims
    :param premises_if_idf_scores: dict with idf-scores for premises

    :return dicts with the highest IF-IDF scores -> most specific words for each argument type
    """
    most_specific_major_claims = {}
    most_specific_claims = {}
    most_specific_premises = {}
    for word in IDF:
        if_idf_score_major_claims = major_claims_if_idf_scores[word]
        if_idf_score_claims = claims_if_idf_scores[word]
        if_idf_score_premises = premises_if_idf_scores[word]

        if if_idf_score_major_claims > if_idf_score_claims and if_idf_score_major_claims > if_idf_score_premises:
            # if_idf_score_major_claims has the highest score
            most_specific_major_claims[word] = if_idf_score_major_claims
        elif if_idf_score_claims > if_idf_score_major_claims and if_idf_score_claims > if_idf_score_premises:
            # if_idf_score_claims has the highest score
            most_specific_claims[word] = if_idf_score_claims
        elif if_idf_score_premises > if_idf_score_major_claims and if_idf_score_premises > if_idf_score_claims:
            # if_idf_score_premises has the highest score
            most_specific_premises[word] = if_idf_score_premises

    return most_specific_major_claims, most_specific_claims, most_specific_premises

def idf(document_text: list):
    """
    Computes the IDF score for each word in all all text of all essays
    idf(t) = log(N/(df + 1))

    :param document_text: concatenated text of all texts of all essays
    :return dict of all words with IDF scores: 
    """
    text = '. '.join(document_text)
    tokens = nlp(text)
    words = [token.text.lower() for token in tokens
             if token.is_stop is not True and token.is_punct is not True]
     # df(t) = occurrence of t in documents
    df_scores = Counter(words)

    # idf(t) = log(N/(df + 1))
    idf_scores = {}    
    for w in df_scores:
        idf_scores[w] = np.log(len(words)/(df_scores[w]+1))

    return Counter(idf_scores)


def tf(argument_unit_text: list):
    """
    Computes the TF score for each word in argument unit
    tf(t,d) = count of t in argument_unit_text / number of words in argument_unit_text

    :param argument_unit_text_list: concatenated text of all major claims in the essays
    :return dict of all words with TF scores: 
    """
    tokens = nlp(argument_unit_text)
    words = [token.text.lower() for token in tokens
             if token.is_stop is not True and token.is_punct is not True]
    freq = Counter(words)
    tf_scores = {}
    for w in freq:
        tf_scores[w] = freq[w] / len(words)
    return Counter(tf_scores)


def tf_idf(argument_unit_text: list, idf_scores):
    """
    Computes the TF-IDF score for each word in argument unit
    argument unit = major claim, claim, premises
    tf-idf(t, d) = tf(t, d) * log(N/(df + 1))    

    :param argument_unit_text_list: concatenated text of all major claims in the essays
    :param IDF: IDF value of the whole text of all essays
    :return dict with all words and their TF-IDF scores for current argument unit: 
    """
    tf_scores = tf(argument_unit_text)
    
    tf_idf_scores = {}
    for w in tf_scores:
         tf_idf_scores[w] = tf_scores[w] * idf_scores[w]
    
    return Counter(tf_idf_scores)




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
    documents_text = []
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

                 # append the whole text to documents text for calculating DF score
                documents_text.append(essay['text'])     

                # Tokenizing using the nlp() of the spaCy library
                # Appending the text of the argument unit to a list
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
        # Getting the words_frequency and a set of unique words
        major_claims_common_words, major_claims_words_set = get_most_common_words(major_claims_text)
        claims_common_words, claims_words_set = get_most_common_words(claims_text)
        premises_common_words, premises_words_set = get_most_common_words(premises_text)

        # calculate IDF score for each word in the whole text of all essays 
        idf_scores = idf(documents_text)        

        # prepare texts for if-idf calculation
        major_claims_text = '. '.join(major_claims_text) 
        claims_text = '. '.join(claims_text) 
        premises_text = '. '.join(premises_text) 
  
        # calculate IF-IDF score for all words in major claims | claims | premises
        major_claims_tf_idf_scores = tf_idf(major_claims_text, idf_scores)
        claims_tf_idf_scores = tf_idf(claims_text, idf_scores)
        premises_tf_idf_scores = tf_idf(premises_text, idf_scores)

        # get specific words -> create lists with highest tf-idf score for each word
        major_claims_specific_words, claims_specific_words, premises_specific_words = get_all_specific_words(idf_scores, major_claims_tf_idf_scores, claims_tf_idf_scores, premises_tf_idf_scores)

        # get the top 10 scores for each 
        major_claims_ten_specific_words = Counter(major_claims_specific_words).most_common(10)
        claims_ten_specific_words = Counter(claims_specific_words).most_common(10)
        premises_ten_specific_words = Counter(premises_specific_words).most_common(10)


        # We iterate through the most frequently occurring words and then check if they occur in the other argument unit
        # or not. If they do we skip them and continue this until we have 10 most specific words
        major_claims_ten_common_words = get_specific_words(major_claims_common_words, claims_words_set,
                                                             premises_words_set)
        claims_ten_common_words = get_specific_words(claims_common_words, major_claims_words_set,
                                                       premises_words_set)
        premises_ten_common_words = get_specific_words(premises_common_words, major_claims_words_set,
                                                         claims_words_set)

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

    print("\n10 most common words in major claims:")
    for i, word in enumerate(major_claims_ten_common_words):
        print("{}) '{}' -- {} times".format(i+1, word[0], word[1]))
    print("\n10 most common words in claims:")
    for i, word in enumerate(claims_ten_common_words):
        print("{}) '{}' -- {} times".format(i+1, word[0], word[1]))
    print("\n10 most common words in premises:")
    for i, word in enumerate(premises_ten_common_words):
        print("{}) '{}' -- {} times".format(i+1, word[0], word[1]))

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
