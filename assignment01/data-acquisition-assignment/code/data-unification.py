from os import listdir
import json
import os
import re

#############################################
# PLEASE SET TO CORRECT PATH BEFORE RUNNING #
#############################################
CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
CONST_ESSAYPATH = f'{CURRENT_WORKING_DIR}/../data/ArgumentAnnotatedEssays-2.0/brat-project-final/'
CONST_SUFFICIENTPATH = f'{CURRENT_WORKING_DIR}/../data/UKP-InsufficientArguments_v1.0/data-tokenized.tsv'
CONST_CONFIRMATIONBIAS = f'{CURRENT_WORKING_DIR}/../data/UKP-OpposingArgumentsInEssays_v1.0/labels.tsv'


# Object based on the sample.json file
class OutputObject(object):
    id = ""
    title = ""
    text = ""
    major_claim = []
    claims = []
    premises = []
    confirmation_bias = False  # not biased until proven guilty
    paragraphs = []

    # The class "constructor" - It's actually an initializer
    def __init__(self, essay_id, title, text, major_claim, claims, premises, paragraphs, confirmation_bias):
        self.id = essay_id
        self.title = title
        self.text = text
        self.major_claim = major_claim
        self.claims = claims
        self.premises = premises
        self.paragraphs = paragraphs
        self.confirmation_bias = confirmation_bias


def get_entity_contents(file_content: str, entity_name: str) -> list:
    """
    :param file_content: content of the essayXXX.ann
    :param entity_name: MajorClaim or Claim or Premise
    :rtype: list
    :return: list of dicts with span, text of given entity and ann-file content
    """
    lines = file_content.split("\n")
    entity_content = []
    for line in lines:
        if re.search(r'\b' + entity_name + r'\b', line):
            line = line.split("\t")
            span_start = line[1].split(" ")[1]
            span_end = line[1].split(" ")[2]
            entity_content = entity_content + [{"span": [span_start, span_end], "text": line[2]}]
    return entity_content


def get_paragraphs_and_sufficient_per_id(essay_id: str) -> list:
    """
    :rtype: list
    :param essay_id: ID as string including preceding zeros
    :return: list of dicts of the paragraphs with text and sufficient parameter
    """
    tsv_file = open(CONST_SUFFICIENTPATH, "r", errors='ignore')  # Has some weird characters, this removes them
    file_content = tsv_file.read()
    lines = file_content.split("\n")  # Format in first Line: ESSAY	ARGUMENT	TEXT	ANNOTATION
    lines.pop(0)
    paragraphs = []
    for line in lines:
        line = line.split("\t")
        if int(line[0]) == int(essay_id):
            sufficient = True
            if "insufficient" in line[3]:
                sufficient = False
            paragraphs = paragraphs + [{"text": line[2], "sufficient": sufficient}]
    return paragraphs


def get_confirmation_bias(essay_id: str) -> bool:
    """
    :rtype: bool
    :param essay_id: ID as string including preceding zeros
    :return: true if confirmation bias true
    """
    tsv_file = open(CONST_CONFIRMATIONBIAS, "r")
    file_content = tsv_file.read()
    lines = file_content.split("\n")  # Format in first Line: id    label
    lines.pop(0)
    for line in lines:
        line = line.split("\t")
        if line[0].split("essay")[1] == essay_id:
            if line[1] == "positive":
                return True
            else:
                return False


def get_all_essay_data() -> list:
    # get all essayXXX.txt file names as basis
    essay_texts = list(filter(lambda x: ".txt" in x, listdir(CONST_ESSAYPATH)))
    essay_texts.sort()
    all_output_elements = []
    # go through all essayXXX.txt files and gather all corresponding information
    for fileName in essay_texts:
        text_file = open(CONST_ESSAYPATH + fileName, "r", encoding="utf8")
        file_id = fileName.split("essay")[1].split(".txt")[0]  # get ID of current file

        # read text-file and clean
        content = text_file.read()
        content = content.replace("\n \n", "\n\n")  # slight cleaning: essay140.txt has "/n /n" with a space
        content = content.replace("\n  \n", "\n\n")  # slight cleaning: essay402.txt has "/n  /n" with a space

        # title is contained in first part of the text-file
        title = content.split("\n\n")[0]
        # all text is in the second part of the text-file
        text = content.split("\n\n")[1]

        # gather corresponding essayXXX.ann file
        file_ann = open(CONST_ESSAYPATH + "essay" + file_id + ".ann", "r")
        ann_content = file_ann.read()

        major_claims = get_entity_contents(ann_content, "MajorClaim")
        claims = get_entity_contents(ann_content, "Claim")
        premises = get_entity_contents(ann_content, "Premise")

        paragraphs = get_paragraphs_and_sufficient_per_id(file_id)

        bias = get_confirmation_bias(file_id)

        # create a output object as contained in output.json and save it
        obj = OutputObject(file_id, title, text, major_claims, claims, premises, paragraphs, bias)
        all_output_elements = all_output_elements + [obj]
    return all_output_elements


def main():
    print("Started to create the unified data file...")
    all_essay_data = get_all_essay_data()
    # write
    json_dump = json.dumps([element.__dict__ for element in all_essay_data], indent=4, ensure_ascii=False)
    with open(f'{CURRENT_WORKING_DIR}/../data/unified_data.json', "w") as outfile:
        outfile.write(json_dump)
    print("Successfully created unified data in '/data/unified_data.json'.")


# run main function
if __name__ == '__main__':
    main()
