import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, LSTM, Dense, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname("__file__")))
TRAINING_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/train_BIO.txt'
TEST_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/test_BIO.txt'
GLOVE_FILE_PATH = f'{CURRENT_WORKING_DIR}/data/glove.6B.100d.txt'

if __name__ == '__main__':

    # Parameters
    EMBEDDING_DIM = 100

    train_df = pd.read_csv(TRAINING_DATA_PATH, names=['token', 'tag'], sep='\t', skipinitialspace=True,
                           quotechar='"').dropna()
    test_df = pd.read_csv(TEST_DATA_PATH, names=['token', 'tag'], sep='\t', skipinitialspace=True,
                          quotechar='"').dropna()

    print("Training Data Info: ", train_df.info)
    print("Testing data Info: ", test_df.info)

    train_tokens = train_df['token'].replace("\t", "", regex=True).replace("\n", "", regex=True)
    test_tokens = test_df['token'].replace("\t", "", regex=True).replace("\n", "", regex=True)

    # output labels
    train_labels = train_df['tag']
    test_labels = test_df['tag']

    train_tokenizer = Tokenizer()
    train_tokenizer.fit_on_texts(train_tokens.values)
    word_index = train_tokenizer.word_index
    print('%s unique tokens.' % len(word_index))

    vocab_size = len(word_index) + 1

    X_train = train_tokenizer.texts_to_sequences(train_tokens.values)
    X_train = pad_sequences(X_train, padding='post', maxlen=4)
    print("Shape of X_train:", X_train.shape)

    test_tokenizer = Tokenizer()
    test_tokenizer.fit_on_texts(test_tokens.values)

    X_test = test_tokenizer.texts_to_sequences(test_tokens.values)
    X_test = pad_sequences(X_test, padding='post', maxlen=4)
    print("Shape of X_test:", X_test.shape)

    Y_train = pd.get_dummies(train_labels).values
    print('Shape of Y_train:', Y_train.shape)

    Y_test = pd.get_dummies(test_labels).values
    print('Shape of Y_test:', Y_test.shape)

    # Use Pre-trained Glove embeddings
    embeddings_index = {}
    f = open(GLOVE_FILE_PATH)
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words that are not found in glove embeddings will be set to 0.
            embedding_matrix[i] = embedding_vector

    print('Found %s word vectors.' % len(embeddings_index))

    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=X_train.shape[1], weights=[embedding_matrix],
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())

    epochs = 2
    batch = 2000

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

    print('Accuracy: %f' % (accuracy * 100))

    blstm_pred = model.predict(X_test)

    blstm_highest_pred = [np.argmax(pred) for pred in blstm_pred]
    blstm_highest_pred = list(map(lambda el: [el], blstm_highest_pred))

    print(blstm_highest_pred)

    # result_df = pd.DataFrame(list(zip(X_test, blstm_highest_pred)))
    # reverse_word_map = dict(map(reversed, test_tokenizer.word_index.items()))
    # words = [reverse_word_map.get(letter) for letter in X_test]
    output = test_tokenizer.sequences_to_texts(blstm_highest_pred)
    result_df = pd.DataFrame(list(zip(output, blstm_highest_pred)))
    #     result_df['token'] = X_test
    #     result_df['tag'] = blstm_highest_pred
    result_df.to_csv('pred.txt', header=None, index=None, sep='\t', mode='w')
