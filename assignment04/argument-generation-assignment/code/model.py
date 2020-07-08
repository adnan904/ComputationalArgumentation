import json
import pandas as pd
import os
import csv
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CORPUS_PATH = f'{CURRENT_WORKING_DIR}/../data/essay_prompt_corpus.json'
SPLIT_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/train-test-split.csv'
PRED_FILE_PATH = f'{CURRENT_WORKING_DIR}/../data/predictions.json'


def get_train_test_split_essays(corpus, split_scheme) -> (list, list):
    """
    :param corpus: unified data file with all the essays
    :param split_scheme: train_test_split scheme file
    :rtype: list, list
    :return: pandas dataframe of train, test split essay id, text, prompt
    """

    train_test_split_dict = {}
    test_df = pd.DataFrame(columns=['id', 'text', 'prompt'])

    # create a dict of the type: {essay_id: Tag},  where Tag = 'TRAIN' or 'TEST'
    for row in split_scheme:
        if len(row) > 0:
            essay_id = int(row[0].split('essay')[1])
            train_test_split_dict[essay_id] = row[1]

    # extract essays that match the test_train_split scheme
    for essay in corpus:
        if train_test_split_dict[int(essay['id'])] == 'TEST':
            test_df = test_df.append({'id': essay['id'], 'text': essay['text'], 'prompt': essay['prompt']},
                                     ignore_index=True)
    test_df.sort_values('id', inplace=True)
    return test_df


class Prediction(object):
    id = ""
    prompt = ""

    def __init__(self, essay_id, prompt):
        self.id = str(essay_id)
        self.prompt = str(prompt)


if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
    json_corpus = json.load(open(CORPUS_PATH, encoding='utf-8'))

    # Read train_test_split and get essays from the unified corpus based on the split
    with open(SPLIT_FILE_PATH, newline='', encoding='utf-8') as csvfile:
        train_test_split_file = csv.reader(csvfile, delimiter=';')
        next(train_test_split_file, None)
        test_essays = get_train_test_split_essays(json_corpus, train_test_split_file)
        test_X = list(test_essays['text'])
        test_y = list(test_essays['prompt'])
        test_id = list(test_essays['id'])

        predictions = []

        for text, id in zip(test_X, test_id):
            t5_prepared_Text = "summarize: " + text.lower()[:512]
            tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

            # summmarize
            summary_ids = model.generate(tokenized_text)

            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            predictions.append(Prediction(id, output))

        json_dump = json.dumps([obj.__dict__ for obj in predictions], indent=4, ensure_ascii=False)
        with open(PRED_FILE_PATH, "w", encoding='utf-8') as outfile:
            outfile.write(json_dump)

        print("Successfully created prediction file in '" + PRED_FILE_PATH + "'.")

