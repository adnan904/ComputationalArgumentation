import json
import pandas as pd
import os
import csv
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

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
    - 47 adversative transition phrases: https://msu.edu/user/jdowell/135/transw.html#anchor1782036
    - 5 different categories
    - distinguished between lower/upper case
    - distinguished between presence in surrounding paragraphs (introduction or conclusion) or in a body paragraph.    
    '''

    def __init__(self):
        pass

    def get_feature_names(self):
        return np.array(['concession_phrases_uc', 'conflict_phrases_uc', 'dismissal_phrases_uc', 'emphasis_phrases_uc', 'replacement_phrases_uc',
                        'concession_phrases_lc', 'conflict_phrases_lc', 'dismissal_phrases_lc', 'emphasis_phrases_lc', 'replacement_phrases_lc'])

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):

        # create 2 dimensional lists, each inner array stands for one adversative transition phrase category

        # --- introduction + conclusion --- 
        # features for all categories of adversative transition phrases in lower case
        ic_lc = [[],[]]

        # features for all categories of adversative transition phrases in upper case
        ic_uc = [[],[]]

        # --- body ---
        # features for all categories of adversative transition phrases in lower case
        b_lc = [[],[]]

        # features for all categories of adversative transition phrases in upper case
        b_uc = [[],[]]

        # lists of adversative transition phrases
        concession_phrases = ['Nevertheless', 'Even though', 'On the other hand', 'Admittedly', 'Yet', 'despite this','albeit']
        conflict_phrases = ['By way of contrast', 'On the other hand', 'Yet', 'In contrast', 'Still']
        #dismissal_phrases = ['All the same']
        #replacement_phrases = ['Rather', 'Instead']

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

        # print(X)
        # print(X.shape)

        if not hasattr(self, 'scalar'):
            self.scalar = StandardScaler().fit(X)
        return self.scalar.transform(X)




def adv_trans_text_analysis(test_essays):
    concession_phrases = ['Nevertheless', 'Even though', 'On the other hand', 'Admittedly', 'Yet', 'despite this','albeit']
    conflict_phrases = ['By way of contrast', 'On the other hand', 'Yet', 'In contrast', 'Still']
    #dismissal_phrases = ['All the same']
    #emphasis_phrases = []
    #replacement_phrases = ['Rather', 'Instead']

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
        train_X = list(train_essays['text'])
        train_y = list(train_essays['bias'])
        test_X = list(test_essays['text'])
        test_y = list(test_essays['bias'])

       
        # # Naive Bayes
        # nb_pipeline = Pipeline([('vec', TfidfVectorizer(ngram_range=(2, 3))),
        #                         ('clf', MultinomialNB())
        #                         ])
        #
        # nb_pipeline.fit(train_X, train_y)
        # pred = nb_pipeline.predict(test_X)
        # f1 = f1_score(y_true=test_y, y_pred=pred)
        # f1_macro = f1_score(y_true=test_y, y_pred=pred, average='macro')
        # print('F1 for Naive-Bayes: ' + str(f1))
        # print('F1-macro for Naive-Bayes: ' + str(f1_macro))

        # Kernel SVM
        kernel_features = []
        kernel_features.append(('adv_trans_features', AdversativeTransitionFeatures()))

        countVecWord = TfidfVectorizer(ngram_range=(1, 3))
        kernel_features.append(('vec', countVecWord))

        all_kernel_features = FeatureUnion(kernel_features)

        svm_pipeline = Pipeline(
            [('all', all_kernel_features),
            ('clf',  SVC(kernel='rbf', C=100, gamma=1e-2,  max_iter=1000)),
            ])

        svm_pipeline.fit(train_X, train_y)

        pred = svm_pipeline.predict(test_X)
        f1 = f1_score(y_true=test_y, y_pred=pred)
        f1_macro = f1_score(y_true=test_y, y_pred=pred, average='macro')
        accuracy = accuracy_score(y_true=test_y, y_pred=pred)
        precision = precision_score(y_true=test_y, y_pred=pred)
        recall = recall_score(y_true=test_y, y_pred=pred)
        print('F1 for rbf-SVM: ' + str(f1))
        print('F1-macro for rbf-SVM: ' + str(f1_macro))
        print('Accuracy for rbf-SVM: ' + str(accuracy))
        print('Precision for rbf-SVM: ' + str(precision))
        print('Recall for rbf-SVM: ' + str(recall))
        print("=============================================================")

        # print('Train data:')
        # adv_trans_text_analysis(train_essays)
        # print()
        # print('Test data:')
        # adv_trans_text_analysis(test_essays)
        # print()

        # load custom featues and FeatureUnion with Vectorizer
        lin_features = []
        adv_trans_features = AdversativeTransitionFeatures() # this class includes my custom features
        lin_features.append(('adv_trans_features', adv_trans_features))

        countVecWord = TfidfVectorizer(ngram_range=(1, 3), lowercase=False)
        lin_features.append(('vec', countVecWord))

        all_features = FeatureUnion(lin_features)

        lin_svm_pipeline = Pipeline(
            [('all', all_features),
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5, max_iter=2000,
                                                           tol=None, class_weight='balanced', n_jobs=-1,
                                                           random_state=42)),
            ])

        lin_svm_pipeline.fit(train_X, train_y)
        pred = lin_svm_pipeline.predict(test_X)
        f1 = f1_score(y_true=test_y, y_pred=pred)
        f1_macro = f1_score(y_true=test_y, y_pred=pred, average='macro')
        accuracy = accuracy_score(y_true=test_y, y_pred=pred)
        precision = precision_score(y_true=test_y, y_pred=pred)
        recall = recall_score(y_true=test_y, y_pred=pred)
        print('F1 for lin-SVM: ' + str(f1))
        print('F1-macro for lin-SVM: ' + str(f1_macro))
        print('Accuracy for lin-SVM: ' + str(accuracy))
        print('Precision for lin-SVM: ' + str(precision))
        print('Recall for lin-SVM: ' + str(recall))

        print("=============================================================")
        create_prediction_file(test_essays['id'], pred)