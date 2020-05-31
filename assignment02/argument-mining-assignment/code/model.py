import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
TRAINING_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/train_BIO.txt'
TEST_DATA_PATH = f'{CURRENT_WORKING_DIR}/data/test_BIO.txt'


def get_sentences(tokens):
    """
    :param tokens: all the tokens in lowercase from a BIO file
    :return: a list of paragraphs from from the file
    """
    sentences_list = []
    sentence = []
    for token in tokens:
        token = token.lower()
        if token == '__end_paragraph__' or token == '__end_essay__' or token == '.':
            sentence.append(token)
            sentences_list.append(sentence)
            sentence = []
        else:
            sentence.append(token)
    return sentences_list


def get_features(paragraphs):
    features_list = []
    for para in paragraphs:
        for i in range(len(para)):
            feature_dict = []
            if len(para) == 1:
                feature_dict.append(para[i])
                feature_dict.append('empty')
                feature_dict.append('empty')
                feature_dict.append('empty')
                feature_dict.append('empty')
                # feature_dict.append('yes')
            elif len(para) == 2:
                if i == 0:
                    feature_dict.append(para[i])
                    feature_dict.append('empty')
                    feature_dict.append('empty')
                    feature_dict.append(para[i+1])
                    feature_dict.append('empty')
                    # feature_dict.append('yes')
                else:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i - 1])
                    feature_dict.append('empty')
                    feature_dict.append('empty')
                    feature_dict.append('empty')
                    # feature_dict.append('yes')
            elif len(para) == 3:
                if i == 0:
                    feature_dict.append(para[i])
                    feature_dict.append('empty')
                    feature_dict.append('empty')
                    feature_dict.append(para[i+1])
                    feature_dict.append(para[i+2])
                    # feature_dict.append('yes')
                elif i == 1:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i - 1])
                    feature_dict.append('empty')
                    feature_dict.append(para[i+1])
                    feature_dict.append('empty')
                    # feature_dict.append('no')
                else:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i - 1])
                    feature_dict.append(para[i - 2])
                    feature_dict.append('empty')
                    feature_dict.append('empty')
                    # feature_dict.append('yes')
            elif len(para) == 4:
                if i == 0:
                    feature_dict.append(para[i])
                    feature_dict.append('empty')
                    feature_dict.append('empty')
                    feature_dict.append(para[i+1])
                    feature_dict.append(para[i+2])
                    # feature_dict.append('yes')
                elif i == 1:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i-1])
                    feature_dict.append('empty')
                    feature_dict.append(para[i+1])
                    feature_dict.append(para[i+2])
                    # feature_dict.append('no')
                elif i == 2:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i-1])
                    feature_dict.append(para[i-2])
                    feature_dict.append(para[i+1])
                    feature_dict.append('empty')
                    # feature_dict.append('no')
                else:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i-1])
                    feature_dict.append(para[i-2])
                    feature_dict.append('empty')
                    feature_dict.append('empty')
            if len(para) >= 5:
                if i == 0:
                    feature_dict.append(para[i])
                    feature_dict.append('empty')
                    feature_dict.append('empty')
                    feature_dict.append(para[i+1])
                    feature_dict.append(para[i+2])
                elif i == 1:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i - 1])
                    feature_dict.append('empty')
                    feature_dict.append(para[i + 1])
                    feature_dict.append(para[i + 2])
                elif i == len(para) - 2:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i - 1])
                    feature_dict.append(para[i - 2])
                    feature_dict.append(para[i + 1])
                    feature_dict.append('empty')
                elif i == len(para) - 1:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i - 1])
                    feature_dict.append(para[i - 2])
                    feature_dict.append('empty')
                    feature_dict.append('empty')
                else:
                    feature_dict.append(para[i])
                    feature_dict.append(para[i - 1])
                    feature_dict.append(para[i - 2])
                    feature_dict.append(para[i + 1])
                    feature_dict.append(para[i + 2])
            features_list.append(' '.join(feature_dict))
    return features_list


if __name__ == '__main__':
    # reading Files
    train_df = pd.read_csv(TRAINING_DATA_PATH, names=['token', 'tag'], sep='\t').dropna().reset_index(drop=True)
    test_df = pd.read_csv(TEST_DATA_PATH, names=['token', 'tag'], sep='\t')

    # getting training and testing data
    train_X = train_df['token'].astype('U')
    test_X = test_df['token'].astype('U')
    train_y = train_df['tag'].astype('U')
    test_y = test_df['tag'].astype('U')

    train_sentences = get_sentences(train_X)
    train_X_features = get_features(train_sentences)

    test_sentences = get_sentences(test_X)
    test_X_features = get_features(test_sentences)

    # Linear SVM
    clf_pipeline = Pipeline([('vec', CountVectorizer()),
                             ('clf', MultinomialNB())
                             ])

    clf_pipeline.fit(train_X_features, train_y)
    pred = clf_pipeline.predict(test_X_features)
    result_df = pd.DataFrame()
    result_df['token'] = test_X
    result_df['tag'] = pred
    result_df.to_csv(f'{CURRENT_WORKING_DIR}/data/pred.txt', header=None, index=None, sep='\t', mode='w')




