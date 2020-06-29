import json
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
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/../data/essay_corpus.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/train-test-split.csv'
PRED_FILE_PATH= f'{CURRENT_WORKING_DIR}/../data/predictions.json'


def get_train_test_split_essays(corpus, split_scheme) -> (list, list):
    """
    :param corpus: unified data file with all the essays
    :param split_scheme: train_test_split scheme file
    :rtype: list, list
    :return: pandas dataframe of train, test split essay id, text, bias
    """

    train_test_split_dict = {}
    test_df = pd.DataFrame(columns=['id', 'text', 'bias'])
    train_df = pd.DataFrame(columns=['id', 'text', 'bias'])

    # create a dict of the type: {essay_id: Tag},  where Tag = 'TRAIN' or 'TEST'
    for row in split_scheme:
        if len(row) > 0:
            essay_id = int(row[0].split('essay')[1])
            train_test_split_dict[essay_id] = row[1]

    # extract essays that match the test_train_split scheme
    for essay in corpus:
        if train_test_split_dict[int(essay['id'])] == 'TRAIN':
            text = essay['text'].replace('\n \n', '\n\n').split('\n\n')
            if len(text) == 1:
                text = essay['text'].split('\n  \n')
            text = text[1:]
            if len(text) > 1:
                text = [text[0]]
            train_df = train_df.append({'id': essay['id'], 'text': ''.join(text), 'bias': essay['confirmation_bias']},
                                       ignore_index=True)
        else:
            text = essay['text'].replace('\n \n', '\n\n').split('\n\n')
            text = text[1:]
            test_df = test_df.append({'id': essay['id'], 'text': ''.join(text), 'bias': essay['confirmation_bias']},
                                     ignore_index=True)

    train_df.sort_values('id', inplace=True)
    test_df.sort_values('id', inplace=True)
    return train_df, test_df


def contains_adv_trans_phrase(essay_text, adv_trans_phrases, lower=False): 
    if lower:
        for w in adv_trans_phrases:
            if w.lower() in essay_text:        
                return 1   
    else:
        for w in adv_trans_phrases:
            if w in essay_text:        
                return 1
    return 0


class AdversativeTransitionFeatures(BaseEstimator):
    '''
    Class to add adversative transition features for confirmation bias classification.
    - 15 adversative transition phrases: https://msu.edu/user/jdowell/135/transw.html#anchor1782036
    - 2 different categories
    - distinguished between lower/upper case
    - distinguished between presence in surrounding paragraphs (introduction or conclusion) or in a body paragraph.    
    '''

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):

        # create 2 dimensional lists, each inner array stands for one adversative transition phrase category

        # --- introduction + conclusion --- 
        # features for all categories of adversative transition phrases in lower case
        ic_lc = [[], []]

        # features for all categories of adversative transition phrases in upper case
        ic_uc = [[], []]

        # --- body ---
        # features for all categories of adversative transition phrases in lower case
        b_lc = [[], []]

        # features for all categories of adversative transition phrases in upper case
        b_uc = [[], []]

        # lists of adversative transition phrases
        concession_phrases = ['Nevertheless', 'Even though', 'On the other hand', 'Admittedly', 'Yet',
                              'Albeit', 'Nonetheless', 'Regardless', 'Notwithstanding', 'But even so']
        conflict_phrases = ['By way of contrast', 'In contrast', 'Still', 'Conversely', 'When in fact']

        all_phrasetypes = [concession_phrases, conflict_phrases]

        for essay_text in x_dataset:
            # We identify paragraphs by checking for line breaks and consider 
            # the first paragraph as introduction, 
            # the last as conclusion 
            # and all remaining ones as body paragraphs.
            paragraphs = essay_text.split('\n')
            introduction_and_conclusion = paragraphs[0] + paragraphs[len(paragraphs)-1]
            body = ''
            for i in range(1, len(paragraphs)-2):
                body += paragraphs[i]

            for i in range(len(all_phrasetypes)):
                # introduction an conclusion features (ic)
                # lower case
                ic_lc[i].append(contains_adv_trans_phrase(introduction_and_conclusion, all_phrasetypes[i], lower=True))
                # upper case
                ic_uc[i].append(contains_adv_trans_phrase(introduction_and_conclusion, all_phrasetypes[i]))

                # body features (b)
                # lower case
                b_lc[i].append(contains_adv_trans_phrase(body, all_phrasetypes[i], lower=True))

                # upper case
                b_uc[i].append(contains_adv_trans_phrase(body, all_phrasetypes[i]))

        features = []
        for i in range(len(all_phrasetypes)):
            features.append(ic_lc[i])
            features.append(ic_uc[i])
            features.append(b_lc[i])
            features.append(b_uc[i])

        X = np.array(features).T

        if not hasattr(self, 'scalar'):
            self.scalar = StandardScaler().fit(X)
        return self.scalar.transform(X)


def adv_trans_text_analysis(test_essays):
    concession_phrases = ['Nevertheless', 'Even though', 'On the other hand', 'Admittedly', 'Yet', 'despite this',
                          'Albeit']
    conflict_phrases = ['By way of contrast', 'On the other hand', 'Yet', 'In contrast', 'Still']
    # dismissal_phrases = ['All the same']
    # emphasis_phrases = []
    # replacement_phrases = ['Rather', 'Instead']

    all_phrases = [concession_phrases, conflict_phrases]
    true_dict = {}
    false_dict = {}
    count_true = 0
    count_false = 0
    for text, bias in zip(test_essays['text'], test_essays['bias']): 
            for phrase_type in all_phrases:
                for phrase in phrase_type:                    
                    if phrase in text or phrase.lower() in text: 
                        if bias:
                            count_true += 1
                            if phrase.lower() not in true_dict.keys():
                                true_dict[phrase.lower()] = 0
                            true_dict[phrase.lower()] += 1
                        else:
                            count_false += 1
                            if phrase.lower() not in false_dict.keys():
                                false_dict[phrase.lower()] = 0
                            false_dict[phrase.lower()] += 1
    print('Bias true: ' + str(count_true))
    print('Bias false: ' + str(count_false))

    print('Bias = true: ' + str(sorted(true_dict.items())))
    print('Bias = false: ' + str(sorted(false_dict.items())))

    for trueItem in sorted(true_dict.items()):
        # bias=true means there are counter args
        if trueItem[0] in false_dict:
            if trueItem[1] > false_dict[trueItem[0]]:
                print(str(trueItem)+" > ('"+str(trueItem[0])+", "+str(false_dict[trueItem[0]])+")")


def grid_search(train_X, train_y, features):
    grid_values = {'clf__C': [5, 10, 100, 1000, 2000],
                   'clf__gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}
    grid_pipeline = Pipeline(
            [('all', FeatureUnion(features)),
             ('clf', SVC(kernel='rbf', max_iter=5000, random_state=42)),
             ])

    gs = GridSearchCV(grid_pipeline, grid_values, scoring='f1', cv=5, n_jobs=-1)
    gs.fit(train_X, train_y)
    print('Best parameter combination = ')

    print(gs.best_params_)


class Prediction(object):
    id = ""
    confirmation_bias = False

    def __init__(self, essay_id, confirmation_bias):
        self.id = str(essay_id)
        self.confirmation_bias = bool(confirmation_bias)


def create_prediction_file(test_essay_ids, pred):
    predictions = []
    
    for id, bias in zip(test_essay_ids, pred):
        predictions.append(Prediction(id, bias))

    json_dump = json.dumps([obj.__dict__ for obj in predictions], indent=4, ensure_ascii=False)
    with open(PRED_FILE_PATH, "w", encoding='utf-8') as outfile:
        outfile.write(json_dump)
    
    print("Successfully created prediction file in '" + PRED_FILE_PATH + "'.")


if __name__ == "__main__":
    json_corpus = json.load(open(CORPUS_PATH, encoding='utf-8'))

    # Read train_test_split and get essays from the unified corpus based on the split
    with open(SPLIT_FILE_PATH, newline='', encoding='utf-8') as csvfile:
        train_test_split_file = csv.reader(csvfile, delimiter=';')
        next(train_test_split_file, None)
        train_essays, test_essays = get_train_test_split_essays(json_corpus, train_test_split_file)
        train_X = train_essays['text']
        train_y = train_essays['bias']
        test_X = list(test_essays['text'])
        test_y = list(test_essays['bias'])

        # Analysing the occurrence of Adversative Transition Phrases on training data. Uncomment the code to see
        # print('Train data:')
        # adv_trans_text_analysis(train_essays)
        # print()
        # print('Test data:')
        # adv_trans_text_analysis(test_essays)
        # print()

        # load custom featues and FeatureUnion with Vectorizer
        features = [('adv_trans_features', AdversativeTransitionFeatures()),
                    ('vec', TfidfVectorizer(ngram_range=(1, 3), lowercase=False))]

        # Performing Grid Search for hyper-parameter tuning, uncomment to check
        # grid_search(list(train_X), list(train_y), features)

        svm_pipeline = Pipeline(
            [('all', FeatureUnion(features)),
             ('clf', SVC(kernel='linear', C=2000, gamma=1e-5, max_iter=5000, random_state=42))
             ])

        # 10-fold cross-validation
        f1_list = []
        cv = KFold(n_splits=10)
        for train_index, test_index in cv.split(train_X):
            train_text = list(train_X[train_index])
            train_label = list(train_y[train_index])
            val_text = list(train_X[test_index])
            val_y = list(train_y[test_index])
            svm_pipeline.fit(train_text, train_label)
            predictions = svm_pipeline.predict(val_text)
            f1 = f1_score(y_true=val_y, y_pred=predictions)
            f1_list.append(f1)

        print("The f1 scores for 10-fold cross-validation:  " + str(f1_list))
        print("\n==================Scores for Test Data======================")
        # svm_pipeline.fit(train_X, train_y)
        pred = svm_pipeline.predict(test_X)
        f1 = f1_score(y_true=test_y, y_pred=pred)
        f1_macro = f1_score(y_true=test_y, y_pred=pred, average='macro')
        print('F1 for SVM: ' + str(f1))
        print('F1-macro for SVM: ' + str(f1_macro))

        print("=============================================================\n")
        create_prediction_file(test_essays['id'], pred)
