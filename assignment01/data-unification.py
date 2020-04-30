from os import listdir
import json


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
    def __init__(self, id, title, text, major_claim, claims, premises, paragraphs, confirmation_bias):
        self.id = id
        self.title = title
        self.text = text
        self.major_claim = major_claim
        self.claims = claims
        self.premises = premises
        self.paragraphs = paragraphs
        self.confirmation_bias = confirmation_bias


def getEntityContents(fileContent: str, entityName: str) -> list:
    """
    :param fileContent: content of the essayXXX.ann
    :param entityName: MajorClaim or Claim or Premise
    :rtype: list
    :return: list of dicts with span, text of given entity and ann-file content
    """
    lines = fileContent.split("\n")
    entityContent = []
    for line in lines:
        if entityName in line:
            line = line.split("\t")
            spanStart = line[1].split(" ")[1]
            spanEnd = line[1].split(" ")[2]
            entityContent = entityContent + [{"span": [spanStart, spanEnd], "text": line[2]}]
    return entityContent


def getParagraphsAndSufficientPerID(essayId: str) -> list:
    """
    :rtype: list
    :param essayId: ID as string including preceding zeros
    :return: list of dicts of the paragraphs with text and sufficient parameter
    """
    paragraphs = []
    tsvFile = open(CONST_SUFFICIENTPATH, "r", errors='ignore')  # Has some weird characters, this removes them
    fileContent = tsvFile.read()
    lines = fileContent.split("\n")  # Format in first Line: ESSAY	ARGUMENT	TEXT	ANNOTATION
    lines.pop(0)
    paragraphs = []
    for line in lines:
        line = line.split("\t")
        if int(line[0]) == int(essayId):
            sufficient = True
            if "insufficient" in line[3]:
                sufficient = False
            paragraphs = paragraphs + [{"text": line[2], "sufficient": sufficient}]
    return paragraphs


def getConfirmationBias(essayId: str) -> bool:
    """
    :rtype: bool
    :param essayId: ID as string including preceding zeros
    :return: true if confirmation bias true
    """
    paragraphs = []
    tsvFile = open(CONST_CONFIRMATIONBIAS, "r")
    fileContent = tsvFile.read()
    lines = fileContent.split("\n")  # Format in first Line: id    label
    lines.pop(0)
    bias = []
    for line in lines:
        line = line.split("\t")
        if line[0].split("essay")[1] == essayId:
            if line[0] == "positive":
                return True
            else:
                return False


def getAllEssayData() -> list:
    # get all essayXXX.txt file names as basis
    essayTexts = list(filter(lambda x: ".txt" in x, listdir(CONST_ESSAYPATH)))
    allOutputElements = []
    # go through all essayXXX.txt files and gather all corresponding information
    for fileName in essayTexts:
        textFile = open(CONST_ESSAYPATH + fileName, "r")
        id = fileName.split("essay")[1].split(".txt")[0]  # get ID of current file

        # read text-file and clean
        content = textFile.read()
        content = content.replace("\n \n", "\n\n")  # slight cleaning: essay140.txt has "/n /n" with a space
        content = content.replace("\n  \n", "\n\n")  # slight cleaning: essay402.txt has "/n  /n" with a space

        # title is contained in first part of the text-file
        title = content.split("\n\n")[0]
        # all text is in the second part of the text-file
        text = content.split("\n\n")[1]

        # gather corresponding essayXXX.ann file
        fileAnn = open(CONST_ESSAYPATH + "essay" + id + ".ann", "r")
        annContent = fileAnn.read()

        majorClaims = getEntityContents(annContent, "MajorClaim")
        claims = getEntityContents(annContent, "Claim")
        premises = getEntityContents(annContent, "Premise")

        paragraphs = getParagraphsAndSufficientPerID(id)

        bias = getConfirmationBias(id)

        # create a output object as contained in output.json and save it
        obj = OutputObject(id, title, text, majorClaims, claims, premises, paragraphs, bias)
        allOutputElements = allOutputElements + [obj]
    return allOutputElements


#############################################
# PLEASE SET TO CORRECT PATH BEFORE RUNNING #
#############################################
CONST_ESSAYPATH = "./data/ArgumentAnnotatedEssays-2.0/brat-project-final/"
CONST_SUFFICIENTPATH = "./data/UKP-InsufficientArguments_v1.0/data-tokenized.tsv"
CONST_CONFIRMATIONBIAS = "./data/UKP-OpposingArgumentsInEssays_v1.0/labels.tsv"

def main():
    allEssayData = getAllEssayData()
    # write
    jsonDump = json.dumps([element.__dict__ for element in allEssayData], indent=4)
    with open("./output.json", "w") as outfile:
        outfile.write(jsonDump)