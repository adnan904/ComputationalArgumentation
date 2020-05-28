import pandas as pd
import os
import numpy as np
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import one_hot, Tokenizer
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.layers.embeddings import Embedding

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname("__file__")))
TRAINING_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/train_BIO.txt'
TEST_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/test_BIO.txt'


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

    # Parameters
    MAX_WORDS_COUNT = 119083
    Embedding_dim = 100

    train_df = pd.read_csv(TRAINING_DATA_PATH, names=['token', 'tag'], sep='\t', skipinitialspace=True,
                           quotechar='"').dropna()
    print(train_df.info)

    train_tokens = train_df['token'].replace("\t", "", regex=True).replace("\n", "", regex=True)
    train_tokens_lowercase = np.array([x.lower() if isinstance(x, str) else x for x in train_tokens]).astype('U')

    # output labels
    labels = train_df['tag']

    tokenizer = Tokenizer(num_words=MAX_WORDS_COUNT, lower=True)
    tokenizer.fit_on_texts(train_tokens.values)
    word_index = tokenizer.word_index
    print('%s unique tokens.' % len(word_index))

    vocab_size = len(word_index) + 1

    X_train = tokenizer.texts_to_sequences(train_tokens.values)
    X_train = pad_sequences(X_train, maxlen=4)
    print("Shape of X:", X_train.shape)

    Y_train = pd.get_dummies(labels).values
    print('Shape of labels:', Y_train.shape)

    print(X_train.shape[1])

    model = Sequential()
    model.add(Embedding(MAX_WORDS_COUNT, Embedding_dim, input_length=X_train.shape[1]))
    # model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 5
    batch = 2000

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    loss, accuracy = model.evaluate(X_train, Y_train, verbose=1)

    print('Accuracy: %f' % (accuracy * 100))

    # vocab_length = 15000

    # encode train_tokens
    # TODO, later used bow or TF_IDF encoding instead of simple encoding to see if it improves the results
    # encode_train_tokens = [one_hot(d, vocab_length) for d in
    #                        train_tokens_lowercase]  # remove blank encoded tokens . ? are not encoded

    # pad_encoded_tokens = pad_sequences(encode_train_tokens, maxlen=4, padding='post')

    # encode_train_labels = [one_hot(d, 30) if isinstance(d, str) else d for d in labels]

    # pad_encoded_labels = pad_sequences(encode_train_labels, maxlen=7, padding='post')
    # print(pad_encoded_labels)

    # model = Sequential()
    # model.add(Embedding(vocab_length, 8, input_length=4))
    # model.add(Flatten())
    # #model.add(LSTM(100))
    # model.add(Dense(7, activation='sigmoid'))
    #
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # print(model.summary())
    #
    # model.fit(pad_encoded_tokens, pad_encoded_labels, epochs=5, verbose=0)
    #
    # loss, accuracy = model.evaluate(pad_encoded_tokens, pad_encoded_labels, verbose=1)
    #
    # print('Accuracy: %f' % (accuracy * 100))

    # # Word-Embeddings
    # sentences = get_sentences(train_tokens_lowercase)
    # model = Word2Vec(sentences, min_count=1, size=100, window=10)
    # words = list(model.wv.vocab)
    # # access vector for one word
    # print(model['should'])
    # # find most_similar for 'should'
    # result = model.most_similar(positive=['should'], topn=1)
    # print(result)
