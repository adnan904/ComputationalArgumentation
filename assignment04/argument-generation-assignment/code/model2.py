import json
import numpy as np
import pandas as pd
import os
import csv
from attention import AttentionLayer
from contraction import contraction_mapping
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
stop_words = set(stopwords.words('english'))


CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/../data/essay_prompt_corpus.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/train-test-split.csv'
PRED_FILE_PATH= f'{CURRENT_WORKING_DIR}/../data/predictions.json'


def get_train_test_split_essays(corpus, split_scheme) -> (list, list):
    """
    :param corpus: unified data file with all the essays
    :param split_scheme: train_test_split scheme file
    :rtype: list, list
    :return: pandas dataframe of train, test split essay id, text, prompt
    """

    train_test_split_dict = {}
    test_df = pd.DataFrame(columns=['id', 'text', 'prompt'])
    train_df = pd.DataFrame(columns=['id', 'text', 'prompt'])

    # create a dict of the type: {essay_id: Tag},  where Tag = 'TRAIN' or 'TEST'
    for row in split_scheme:
        if len(row) > 0:
            essay_id = int(row[0].split('essay')[1])
            train_test_split_dict[essay_id] = row[1]

    # extract essays that match the test_train_split scheme
    for essay in corpus:
        if train_test_split_dict[int(essay['id'])] == 'TRAIN':
            train_df = train_df.append({'id': essay['id'], 'text': essay['text'], 'prompt': essay['prompt']},
                                       ignore_index=True)
        else:
            test_df = test_df.append({'id': essay['id'], 'text': essay['text'], 'prompt': essay['prompt']},
                                     ignore_index=True)
    train_df.sort_values('id', inplace=True)
    test_df.sort_values('id', inplace=True)
    return train_df, test_df


def text_cleaner(text, num):
    """
        Performs the following on input text:
            1.Convert everything to lowercase
            2.Contraction mapping
            3.Remove (‘s)
            4.Remove any text inside the parenthesis ( )
            5.Eliminate punctuations and special characters
            6.Remove stopwords
            7.Remove short words
    """
    new_string = text.lower()
    new_string = re.sub(r'\([^)]*\)', '', new_string)
    new_string = re.sub('"', '', new_string)
    new_string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in new_string.split(" ")])
    new_string = re.sub(r"'s\b", "", new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string)
    new_string = re.sub('[m]{2,}', 'mm', new_string)
    if num == 0:
        tokens = [w for w in new_string.split() if w not in stop_words]
    else:
        tokens = new_string.split()
    long_words = []
    for i in tokens:
        if len(i) > 1:                                                 # removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()


def preprocess_text(text, num):
    cleaned_text = []
    for t in text:
        cleaned_text.append(text_cleaner(t, num))
    return cleaned_text


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if (sampled_token != 'eostok'):
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_prompt_len - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


class Prediction(object):
    id = ""
    prompt = ""

    def __init__(self, essay_id, prompt):
        self.id = str(essay_id)
        self.prompt = str(prompt)


if __name__ == "__main__":
    json_corpus = json.load(open(CORPUS_PATH, encoding='utf-8'))

    # Read train_test_split and get essays from the unified corpus based on the split
    with open(SPLIT_FILE_PATH, newline='', encoding='utf-8') as csvfile:
        train_test_split_file = csv.reader(csvfile, delimiter=';')
        next(train_test_split_file, None)
        train_essays, test_essays = get_train_test_split_essays(json_corpus, train_test_split_file)
        train_X = list(train_essays['text'])
        train_y = list(train_essays['prompt'])
        test_X = list(test_essays['text'])
        test_y = list(test_essays['prompt'])
        test_id = list(test_essays['id'])

        # Preprocessing the Training and Test data
        train_X = preprocess_text(train_X, 0)
        train_y = preprocess_text(train_y, 1)
        test_X = preprocess_text(test_X, 0)

        for i in range(len(train_y)):
            train_y[i] = 'sostok ' + train_y[i] + ' eostok'

        # Counting the distribution
        train_word_count = []
        trainY_word_count = []

        # populate the lists with word lengths
        for i, j in zip(train_X, train_y):
            train_word_count.append(len(i.split()))
            train_word_count.append(len(j.split()))

        for i in train_y:
            trainY_word_count.append(len(i.split()))

        max_text_len = max(train_word_count)
        max_prompt_len = max(trainY_word_count)

        # prepare a tokenizer for texts in training data
        x_tokenizer = Tokenizer()
        x_tokenizer.fit_on_texts(train_X)

        thresh = 4

        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in x_tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if value < thresh:
                cnt = cnt + 1
                freq = freq + value

        # prepare a tokenizer for reviews on training data
        x_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        x_tokenizer.fit_on_texts(train_X)

        # convert text sequences into integer sequences
        x_tr = x_tokenizer.texts_to_sequences(train_X)
        x_test = x_tokenizer.texts_to_sequences(test_X)

        # padding zero upto maximum length
        x_tr = pad_sequences(x_tr, maxlen=max_text_len, padding='post')
        x_test = pad_sequences(x_test, maxlen=max_text_len, padding='post')
        x_voc_size = len(x_tokenizer.word_index) + 1

        # preparing a tokenizer for prompts on training data
        y_tokenizer = Tokenizer()
        y_tokenizer.fit_on_texts(train_y)

        thresh = 6

        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in y_tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if (value < thresh):
                cnt = cnt + 1
                freq = freq + value

        # prepare a tokenizer for reviews on training data
        y_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        y_tokenizer.fit_on_texts(train_y)

        # convert summary sequences into integer sequences
        y_tr = y_tokenizer.texts_to_sequences(train_y)

        # padding zero upto maximum length
        y_tr = pad_sequences(y_tr, maxlen=max_prompt_len, padding='post')

        y_voc_size = len(y_tokenizer.word_index) + 1

        # Building the LSTM Model
        latent_dim = 300
        embedding_dim = 100

        # Encoder
        encoder_inputs = Input(shape=(max_text_len,))

        # embedding layer
        enc_emb = Embedding(x_voc_size, embedding_dim, trainable=True)(encoder_inputs)

        # encoder lstm 1
        encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        # encoder lstm 2
        encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        # encoder lstm 3
        encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))

        # embedding layer
        dec_emb_layer = Embedding(y_voc_size, embedding_dim, trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

        # Attention layer
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention input and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        # dense layer
        decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.summary()

        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        es = EarlyStopping(monitor='loss', mode='min', verbose=1)
        history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=1,
                            callbacks=[es], batch_size=512)

        reverse_target_word_index = y_tokenizer.index_word
        reverse_source_word_index = x_tokenizer.index_word
        target_word_index = y_tokenizer.word_index

        # encoder inference
        encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

        # decoder inference
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h,
                                                                                     decoder_state_input_c])

        # attention inference
        attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_inf_concat)

        # Final decoder model
        decoder_model = Model(
            [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2] + [state_h2, state_c2])


        predictions = []

        for text, id in zip(x_test[:1], test_id[:1]):
            random_result_prompt = decode_sequence(text.reshape(1, max_text_len))
            predictions.append(Prediction(id, random_result_prompt))

        json_dump = json.dumps([obj.__dict__ for obj in predictions], indent=4, ensure_ascii=False)
        with open(PRED_FILE_PATH, "w", encoding='utf-8') as outfile:
            outfile.write(json_dump)

        print("Successfully created prediction file in '" + PRED_FILE_PATH + "'.")

