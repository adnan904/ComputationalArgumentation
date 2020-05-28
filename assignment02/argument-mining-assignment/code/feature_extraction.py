import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models import Word2Vec
import os
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import random

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname("__file__")))
TRAINING_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/train_BIO.txt'
TEST_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/test_BIO.txt'


def predict(vectorizer, classifier, data):
    data_features = vectorizer.transform(data)
    predictions = classifier.predict(data_features)
    return predictions


def get_sentences(tokens):
    """
    :param tokens: all the tokens in lowercase from a BIO file
    :return: a list of paragraphs from from the file
    """
    sentences_list = []
    sentence = []
    for token in tokens:
        if token == '__end_paragraph__':
            if len(sentence) > 0:
                sentences_list.append(sentence)
                sentence = []
        elif token == '__end_essay__' or token == 'nan':
            continue
        else:
            sentence.append(token)
    return sentences_list


if __name__ == '__main__':
    # reading Files
    train_df = pd.read_csv(TRAINING_DATA_PATH, names=['token', 'tag'], sep='\t')
    test_df = pd.read_csv(TEST_DATA_PATH, names=['token', 'tag'], sep='\t')

    # getting training and testing data
    train_X = train_df['token'].astype('U')
    test_X = test_df['token'].astype('U')
    train_y = train_df['tag'].astype('U')
    test_y = test_df['tag'].astype('U')
    labels_map = {0: 'B-CLAIM', 1: 'B-MAJOR-CLAIM', 2: 'B-PREMISE', 3: 'I-CLAIM', 4: 'I-MAJOR-CLAIM', 5: 'I-PREMISE',
                  6: 'O', 7: 'O'}

    # Class Distribution. Mostly I-PREMISE and O are majority_classes
    print("\nTotal data-points in training data: " + str(len(train_X)))
    print("\nTotal data-points in test data: " + str(len(test_X)))
    print("\nTraining Data Class Distribution:")
    print(train_df.tag.value_counts())
    print("\nTest Data Class Distribution:")
    print(test_df.tag.value_counts())
    count_vectorizer = CountVectorizer(analyzer="word", preprocessor=None, lowercase=True)

    # BoW for training data
    train_data_bag_of_words = count_vectorizer.fit_transform(train_X)

    # Logistic Regression for our model with bag-of-words features
    logreg = linear_model.LogisticRegression(n_jobs=4, C=1e5, max_iter=100)
    logreg = logreg.fit(train_data_bag_of_words, train_y)
    y_pred = predict(count_vectorizer, logreg, test_X)
    print('LogReg accuracy: %s' % accuracy_score(y_pred, test_y))
    print('LogReg F1-macro: %s' % f1_score(y_pred, test_y, average='macro'))
    print('LogReg F1-weighted: %s' % f1_score(y_pred, test_y, average='weighted'))

    # Majority-class
    majority_class_pred = []
    for _ in range(len(test_y)):
        majority_class_pred.append('I-PREMISE')
    print('majority-class accuracy: %s' % accuracy_score(majority_class_pred, test_y))
    print('majority-class F1-macro: %s' % f1_score(majority_class_pred, test_y, average='macro'))
    print('majority-class F1-weighted: %s' % f1_score(majority_class_pred, test_y, average='weighted'))

    # Random-majority class
    random_class_pred = []
    for _ in range(len(test_y)):
        choice = random.choice([0, 1])
        if choice == 0:
            random_class_pred.append('I-PREMISE')
        else:
            random_class_pred.append('O')
    print('random-majority-class accuracy: %s' % accuracy_score(random_class_pred, test_y))
    print('random-majority-class F1-macro: %s' % f1_score(random_class_pred, test_y, average='macro'))
    print('random-majority-class F1-weighted: %s' % f1_score(random_class_pred, test_y, average='weighted'))

    # Random-Forrest
    rf_vectorizer = CountVectorizer(analyzer="word", preprocessor=None, lowercase=True)
    train_data = rf_vectorizer.fit_transform(train_X)

    forest = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=0)
    forest = forest.fit(train_data, train_y)

    rf_pred = predict(rf_vectorizer, forest, test_X)
    print('random-forest accuracy: %s' % accuracy_score(rf_pred, test_y))
    print('random-forest F1-macro: %s' % f1_score(rf_pred, test_y, average='macro'))
    print('random-forest F1-weighted: %s' % f1_score(rf_pred, test_y, average='weighted'))

    # Naive-BAyes
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])

    nb.fit(train_X, train_y)
    nb_pred = nb.predict(test_X)
    print('NB accuracy: %s' % accuracy_score(nb_pred, test_y))
    print('NB F1-macro: %s' % f1_score(nb_pred, test_y, average='macro'))
    print('NB F1-weighted: %s' % f1_score(nb_pred, test_y, average='weighted'))

    # Linear SVM
    sgd = Pipeline([('vect', CountVectorizer()),        # Counts occurrences of each word
                    ('tfidf', TfidfTransformer()),      # Normalize the counts based on document length
                    ('clf',   SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=100, tol=None, n_jobs=-1,
                                            random_state=9976))
                    ])
    f1_list = []
    acc_list = []
    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(train_X):
        X_train, X_val = train_X[train_index], train_X[test_index]
        y_train, y_val = train_y[train_index], train_y[test_index]
        sgd.fit(X_train, y_train)
        predicted = sgd.predict(X_val)
        f1 = f1_score(y_val, predicted, average='macro')
        acc = accuracy_score(y_val, predicted)
        f1_list.append(f1)
        acc_list.append(acc)
    print(f1_list)
    print(acc_list)
    sgd_pred = sgd.predict(test_X)
    print('SVM accuracy: %s' % accuracy_score(sgd_pred, test_y))
    print('SVM F1-macro: %s' % f1_score(sgd_pred, test_y, average='macro'))
    print('SVM F1-weighted: %s' % f1_score(sgd_pred, test_y,  average='weighted'))
    result_df = pd.DataFrame()
    result_df['token'] = test_X
    result_df['tag'] = sgd_pred
    result_df.to_csv(f'{CURRENT_WORKING_DIR}/data/pred.txt', header=None, index=None, sep='\t', mode='w')
    print()

    # Word-Embeddings
    # sentences = get_sentences(train_tokens_lowercase)
    # model = Word2Vec(sentences, min_count=1, size=100, window=10)
    # words = list(model.wv.vocab)
    # access vector for one word
    # print(model['should'])
    # find most_similar for 'should'
    # result = model.most_similar(positive=['should'], topn=1)
    # print(result)

