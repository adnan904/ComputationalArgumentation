import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import os
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
import random
from numpy import zeros

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Embedding, Flatten


CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname("__file__")))
TRAINING_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/train_BIO.txt'
TEST_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/test_BIO.txt'


def predict(vectorizer, classifier, data):
    data_features = vectorizer.transform(data)
    predictions = classifier.predict(data_features)
    return predictions


def get_sentences(tokens):
    '''

    :param tokens: all the tokens in lowercase from a BIO file
    :return: a list of paragraphs from from the file
    '''
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
    train_df = pd.read_csv(TRAINING_DATA_PATH, names=['token', 'tag'], sep='\t', skipinitialspace=True, quotechar='"')
    train_tokens = train_df['token'].replace("\t", "", regex=True).replace("\n", "", regex=True)
    train_tokens_lowercase = np.array([x.lower() if isinstance(x, str) else x for x in train_tokens]).astype('U')
    train_tags = train_df['tag'].astype('U')
    test_df = pd.read_csv(TEST_DATA_PATH, names=['token', 'tag'], sep='\t', skipinitialspace=True, quotechar='"')
    test_tokens = test_df['token']
    test_tokens_lowercase = np.array([x.lower() if isinstance(x, str) else x for x in test_tokens]).astype('U')
    test_tags = test_df['tag'].astype('U')
    # Class Distribution. Mostly I-PREMISE and O are majority_classes
    print(test_df.tag.value_counts())
    count_vectorizer = CountVectorizer(
        analyzer="word", preprocessor=None, lowercase=True)

    # BoW for training data
    train_data_bag_of_words = count_vectorizer.fit_transform(train_tokens_lowercase)
    # cv = count_vectorizer.vocabulary_

    # Logistic Regression for our model with bag-of-words features
    logreg = linear_model.LogisticRegression(n_jobs=4, C=1e5, max_iter=100)
    logreg = logreg.fit(train_data_bag_of_words, train_tags)
    y_pred = predict(count_vectorizer, logreg, test_tokens_lowercase)
    print('LogReg accuracy: %s' % accuracy_score(y_pred, test_tags))
    print('LogReg F1: %s' % f1_score(y_pred, test_tags, average='macro'))

    majority_class_pred = []
    for _ in range(len(test_tags)):
        majority_class_pred.append('I-PREMISE')
    print('majority-class accuracy: %s' % accuracy_score(majority_class_pred, test_tags))
    print('majority-class F1: %s' % f1_score(majority_class_pred, test_tags, average='macro'))

    random_class_pred = []
    for _ in range(len(test_tags)):
        choice = random.choice([0, 1])
        if choice == 0:
            random_class_pred.append('I-PREMISE')
        else:
            random_class_pred.append('O')
    print('random-majority-class accuracy: %s' % accuracy_score(random_class_pred, test_tags))
    print('random-majority-class F1: %s' % f1_score(random_class_pred, test_tags, average='macro'))
    rf_vectorizer = CountVectorizer(analyzer="word", preprocessor=None, lowercase=True)
    train_data = rf_vectorizer.fit_transform(train_tokens_lowercase)

    forest = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0)
    forest = forest.fit(train_data, train_tags)

    rf_pred = predict(rf_vectorizer, forest, test_tokens_lowercase)
    print('random-forest accuracy: %s' % accuracy_score(rf_pred, test_tags))
    print('random-forest F1: %s' % f1_score(rf_pred, test_tags, average='macro'))

    # Naive-BAyes
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])

    nb.fit(train_tokens_lowercase, train_tags)
    nb_pred = nb.predict(test_tokens_lowercase)
    print('NB accuracy: %s' % accuracy_score(nb_pred, test_tags))
    print('NB F1: %s' % f1_score(nb_pred, test_tags, average='macro'))

    # Linear SVM
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=100, tol=None,
                                   n_jobs=-1)),
                    ])
    sgd.fit(train_tokens_lowercase, train_tags)
    sgd_pred = sgd.predict(test_tokens_lowercase)
    print('SVM accuracy: %s' % accuracy_score(sgd_pred, test_tags))
    print('SVM F1: %s' % f1_score(sgd_pred, test_tags, average='macro'))

    # Word-Embeddings
    sentences = get_sentences(train_tokens_lowercase)
    model = Word2Vec(sentences, min_count=1, size=100, window=10)
    words = list(model.wv.vocab)
    # access vector for one word
    print(model['should'])
    # find most_similar for 'should'
    result = model.most_similar(positive=['should'], topn=1)
    print(result)


    # prepare tokenizer 
    t = Tokenizer()
    t.fit_on_texts(sentences)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(sentences)
    # pad documents to a max length of 4 words
    

    # get the embedding matrix from the embedding layer 
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = model[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

    max_length = max([len(arr) for arr in sentences])
    X = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    
    # main model    
    input = keras.Input(shape=(max_length,))
    model = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length)(input)    
    model =  Bidirectional(LSTM (100, return_sequences=False, dropout=0.50), merge_mode='concat')(model)
    # model = TimeDistributed(Dense(100,activation='relu'))(model)
    model = Flatten()(model)

    model = Dense(100,activation='relu')(model)
    output = Dense(3,activation='softmax')(model)
    model = keras.Model(input,output)
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # TODO: train_tags has to be encoded -> is it necessary to pad the input?
    # Consequently, we also have to add a padding to the train tags
    # model.fit(X, train_tags, validation_split=0.25,epochs= 10, verbose = 2)

    # evaluate the model
    # bilstm_pred = model.predict(test_tokens_lowercase) # .evaluate(train_df, test_df, verbose=2)
    # print('Bi-LSTM accuracy: %s' % accuracy_score(bilstm_pred, test_tags))
    # print('Bi-LSTM  F1: %s' % f1_score(bilstm_pred, test_tags, average='macro'))
