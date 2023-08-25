import csv
import random
from os.path import isdir
from os import mkdir
import spacy
import re
from tqdm import tqdm
import pandas as pd
from collections import Counter

random.seed(23)
MAX_TEXT_LENGTH = 200
TRAIN_DEV_SPLIT = 0.05

TASK = "politicit"

nlp = spacy.load("it_core_news_sm", disable=["lemmatizer", "tagger"])

labelmap = {
    "female": "donna",
    "male": "uomo",
    "left": "sinistra",
    "right": "destra",
    "moderate_left": "centrosinistra",
    "moderate_right": "centrodestra"
}

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")

    with open(f"data/{TASK}/politicIT_phase_2_train_public.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = dict()
        for row in tqdm(reader):
            text = clean_input_text(row['tweet'])
            gender = labelmap[row['gender']]
            label_b = labelmap[row['ideology_binary']]
            label_m = labelmap[row['ideology_multiclass']]
            id = row['label']
            
            doc = nlp(text)
            tokens = [token.text for token in doc]

            if not id in data:
                data[id] = {
                    "gender": gender,
                    "label_b": label_b,
                    "label_m": label_m,
                    "sentences": []
                }
            data[id]["sentences"].append({
                    "text":text, 
                    "tokens":len(tokens)})

    with open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:
        for id, item in data.items():
            t = 0
            output = ""
            for sentence in item["sentences"]:
                if t+sentence["tokens"] <= MAX_TEXT_LENGTH:
                    output += " "+sentence["text"]
                    t += sentence["tokens"]
                else:
                    output = output.strip()
                    line = f"{id}\tpoliticit\t{output}\t{item['gender']} {item['label_b']} {item['label_m']}\n"
                    if random.random()<TRAIN_DEV_SPLIT:
                        f_dev.write(line)
                    else:
                        f_train.write(line)
                    t = 0
                    output = ""


def get_majority(list_, type_, pib=""):
    most_frequent_gender = "male"
    most_frequent_pib = "left"
    
    # pim has 4 total labels
    most_frequent_pim = "moderate_left"
    second_most_frequent_pim = "right"
    third_most_frequent_pim = "left"

    # count here occurencies 
    occurencies = Counter(list_).most_common()

    # if there are equal occurencies store them
    equals = []
    for el in occurencies:
        if len(equals) == 0:
            equals.append(el)
            continue
        if el[1] == equals[-1][1]:
            equals.append(el)

    # if there is more than 1 maximum occurency, return the most frequent class based on type
    if len(equals) > 1:
        if type_ == "gender":
            return most_frequent_gender
        elif type_ == "pib":
            return most_frequent_pib
        else:
            if pib in [el[0] for el in equals]:
                return pib
            elif pib != "":
                for pim in [el[0] for el in equals]:
                    if pib in pim:
                        return pim
            # pim has 4 total labels, need to check if the most frequent label in the training set is contained in the maximum occurencies for this example
            elif most_frequent_pim in [el[0] for el in equals]:
                return most_frequent_pim
            elif second_most_frequent_pim in [el[0] for el in equals]:
                return second_most_frequent_pim
            else:
                return third_most_frequent_pim
    # else no equals and return the most frequent element ([0] only the text)
    return equals[0][0]


def decode():
    """
    Reads the file from 'out/politicit/predictions.txt'. \\
    Creates a CSV file in 'out/politicit/predictions.csv' \\
        to submit in the form 'word, label' with no header and sentences divided by an empty line.
    """

    labelmap = {
        "donna": "female",
        "uomo": "male",
        "sinistra": "left",
        "destra": "right",
        "centrosinistra": "moderate_left",
        "centrodestra": "moderate_right"
    }
    
    data = dict()
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")

            id = line_split[0]
            label = clean_input_text(line_split[3])
            if label == "UNK":
                self_assigned_gender = "male"
                pib = "left"
                pim = "moderate_left"
            else:
                label_split = label.split(" ")

                self_assigned_gender = labelmap[label_split[0]]
                pib = labelmap[label_split[1]]
                pim = labelmap[label_split[-1]]

            if id not in data:
                data[id] = {
                    "gender": [],
                    "pib": [],
                    "pim": []
                }
            data[id]['gender'].append(self_assigned_gender)
            data[id]['pib'].append(pib)
            data[id]['pim'].append(pim)

    cluster_of_texts_id, self_assigned_gender, pib, pim = [], [], [], []
    for id, features in data.items():
        cluster_of_texts_id.append(id)
        self_assigned_gender.append(get_majority(features['gender'], "gender"))
        pib.append(get_majority(features['pib'], "pib"))
        
        # pass here last pib in order to better choose pim
        pim.append(get_majority(features['pim'], "pim", pib[-1]))

    data = pd.DataFrame({
        "label": cluster_of_texts_id, 
        "gender": self_assigned_gender, 
        "ideology_binary": pib, 
        "ideology_multiclass": pim
    })
    data.to_csv(f"out/{TASK}/results_post_eval.csv", index=False)
    print("REMEMBER TO ZIP THE FILE BEFORE SUBMITTING AND BEFORE GENERATING ANOTHER ONE!")

