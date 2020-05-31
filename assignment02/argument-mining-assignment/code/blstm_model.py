import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle
from matplotlib import pyplot

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname("__file__")))
TRAINING_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/train_BIO.txt'
TEST_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/test_BIO.txt'
GLOVE_FILE_PATH = f'{CURRENT_WORKING_DIR}/data/glove.6B.300d.txt'

if __name__ == '__main__':

    # Parameters
    EMBEDDING_DIM = 300
    MAX_TOKEN_LENGTH = 100

    train_df = pd.read_csv(TRAINING_DATA_PATH, names=['token', 'tag'], sep='\t', skipinitialspace=True,
                           quotechar='"').fillna('O')
    test_df = pd.read_csv(TEST_DATA_PATH, names=['token', 'tag'], sep='\t', skipinitialspace=True,
                          quotechar='"').fillna('O')

    train_df = shuffle(train_df)
    test_df = shuffle(test_df)


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
    X_train = pad_sequences(X_train, padding='post', maxlen=MAX_TOKEN_LENGTH)
    print("Shape of X_train:", X_train.shape)

    test_tokenizer = Tokenizer()
    test_tokenizer.fit_on_texts(test_tokens.values)

    X_test = test_tokenizer.texts_to_sequences(test_tokens.values)
    X_test = pad_sequences(X_test, padding='post', maxlen=MAX_TOKEN_LENGTH)
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

    # embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, input_length=X_train.shape[1], weights=[embedding_matrix],
    #                     trainable=False)
    # sequence_input = Input(shape=(X_train.shape[1],), dtype='int32')
    # embedded_sequences = embedding_layer(sequence_input)
    # l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    # preds = Dense(Y_train.shape[1], activation='softmax')(l_lstm)
    # model = Model(sequence_input, preds)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

###########>>>>>

    # x = Conv1D(128, 3, activation='relu')(embedded_sequences)
    # x = MaxPooling1D(3)(x)
    # x = Conv1D(128, 3, activation='relu')(x)
    # x = MaxPooling1D(3)(x)
    # x = Conv1D(128, 3, activation='relu')(x)
    # x = MaxPooling1D(3)(x)  # global max pooling
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # preds = Dense(Y_train.shape[1], activation='softmax')(x)
    # model = Model(sequence_input, preds)
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


#################>>>>>
    # embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=X_train.shape[1],
    #                             trainable=False)
    # sequence_input = Input(shape=(X_train.shape[1],), dtype='int32')
    # embedded_sequences = embedding_layer(sequence_input)
    #
    # print("embedded_sequences.shape : ", embedded_sequences.shape)
    #
    # blstm_layer = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    #
    # # First BLSTM layer
    # l1 = blstm_layer(sequence_input)
    #
    # preds = Dense(1, activation='softmax')
    #
    # model = Model(input=sequence_input,
    #               outputs=preds)

###################>>>>>>

    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=X_train.shape[1], weights=[embedding_matrix],
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(Y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    # plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

    epochs = 2
    batch = 2000

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, validation_split=0.2,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    # pyplot.plot(history.history['loss'])
    # pyplot.plot(history.history['val_loss'])
    # pyplot.title('model train vs validation loss')
    # pyplot.ylabel('loss')
    # pyplot.xlabel('epoch')
    # pyplot.legend(['train', 'validation'], loc='upper right')
    # pyplot.show()
    #
    # pyplot.title('Accuracy')
    # pyplot.plot(history.history['acc'], label='train')
    # pyplot.plot(history.history['val_acc'], label='test')
    # pyplot.legend()
    # pyplot.show()

    accr = model.evaluate(X_test, Y_test)
    print('Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    blstm_pred = model.predict(X_test)

    blstm_highest_pred = [np.argmax(pred) for pred in blstm_pred]

    result_df = pd.DataFrame(list(zip(X_test, blstm_highest_pred)))
    #     result_df['token'] = X_test
    #     result_df['tag'] = blstm_highest_pred
    result_df.to_csv('pred.txt', header=None, index=None, sep='\t', mode='w')
    result_df.to_csv(f'{CURRENT_WORKING_DIR}/data/pred.txt', header=None, index=None, sep='\t', mode='w')

    print(blstm_pred)

    labels = ['B-CLAIM', 'B-MAJOR-CLAIM', 'B-PREMISE', 'I-CLAIM',  'I-MAJOR-CLAIM', 'I-PREMISE', 'O']
