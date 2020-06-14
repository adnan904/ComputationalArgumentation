import json
import pandas as pd
import os
import csv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/../data/unified_data.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/train-test-split.csv'


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
            text = essay['text'].split('\n\n')
            train_df = train_df.append({'id': essay['id'], 'text': text, 'bias': essay['confirmation_bias']},
                                       ignore_index=True)
        else:
            text = essay['text'].split('\n\n')
            test_df = test_df.append({'id': essay['id'], 'text': text, 'bias': essay['confirmation_bias']},
                                     ignore_index=True)
    train_df.sort_values('id', inplace=True)
    test_df.sort_values('id', inplace=True)
    return train_df, test_df


if __name__ == "__main__":
    json_corpus = json.load(open(CORPUS_PATH))

    # Read train_test_split and get essays from the unified corpus based on the split
    with open(SPLIT_FILE_PATH, newline='') as csvfile:
        train_test_split_file = csv.reader(csvfile, delimiter=';')
        next(train_test_split_file, None)
        train_essays, test_essays = get_train_test_split_essays(json_corpus, train_test_split_file)
        train_X = [x[0] for x in train_essays['text']]
        train_y = list(train_essays['bias'])
        test_X = [x[0] for x in test_essays['text']]
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
        svm_pipeline = Pipeline([('vec', TfidfVectorizer(ngram_range=(1, 3))),
                                ('clf',  SVC(kernel='rbf', C=100, gamma=1e-2, max_iter=1000))
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

        # Linear SVM
        lin_svm_pipeline = Pipeline([('vec', TfidfVectorizer(ngram_range=(2, 3), lowercase=False)),
                                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5, max_iter=2000,
                                                           tol=None, class_weight='balanced', n_jobs=-1,
                                                           random_state=42))
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