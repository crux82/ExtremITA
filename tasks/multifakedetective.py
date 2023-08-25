from os.path import isdir
from os import mkdir

import pandas as pd
import random
import re

random.seed(23)
TRAIN_DEV_SPLIT = 0.05

TASK = "multifakedetective"

labelmap = {
    "0": "certamente falso",
    "1": "probabilmente falso",
    "2": "probabilmente vero",
    "3": "certamente vero"
}

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
        
    df = pd.read_csv("data/MULTI-Fake-DetectiVE/MULTI-Fake-Detective_Task1_Data.tsv", sep="\t", encoding="utf-8")

    added_ids = []
    with open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:
        
        for id, text, label in zip(df['ID'], df['Text'], df['Label']):
            # there are more than 100 duplicates, check here by id
            # if the example has not been added (id not in added_ids), then add the id to the list and write the line on file
            # else this is a duplicate: go to the next example
            if id not in added_ids:
                added_ids.append(id)

                # replace multiple tabs, new lines and multiple spaces with a single space
                text = clean_input_text(text)

                line = f"{id}\t{TASK}\t{text}\t{labelmap[str(label)]}\n"
                if random.random()<TRAIN_DEV_SPLIT:
                    f_dev.write(line)
                else:
                    f_train.write(line)


def decode():
    """
    Reads the file from 'out/multifakedetective/predictions.txt'. \\
    Creates a TSV file in 'out/multifakedetective/predictions.tsv' \\
        to submit in the form 'id, label' with no header.
    """

    label_map = {
        "certamente falso": 0,
        "probabilmente falso": 1,
        "probabilmente vero": 2,
        "certamente vero": 3
    }
    
    ids, labels = [], []
    ids_additional_data, labels_additional_data = [], []
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f, \
         open(f"out/{TASK}/test_additional_data_preds.txt", "r", encoding="utf-8") as f_additional_data:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            id = line_split[0]
            ids.append(id)
            label = clean_input_text(line_split[3]).lower()
            # if the label is well formatted take its numerical counterpart
            if label in label_map:
                labels.append(label_map[label])
            # else append most frequent class
            else:
                labels.append(2)

        lines_additional_data = f_additional_data.readlines()
        for line_additional_data in lines_additional_data:
            line_split = line_additional_data.split("\t")
            id = line_split[0]
            ids_additional_data.append(id)
            label = clean_input_text(line_split[3]).lower()
            # if the label is well formatted take its numerical counterpart
            if label in label_map:
                labels_additional_data.append(label_map[label])
            # else append most frequent class
            else:
                labels_additional_data.append(2)

    data = pd.DataFrame({
        "ID": ids,
        "label": labels
    })
    data.to_csv(f"out/{TASK}/predictions.tsv", sep="\t", header=False, index=False)
    
    data_additional_data = pd.DataFrame({
        "ID": ids_additional_data,
        "label": labels_additional_data
    })
    data_additional_data.to_csv(f"out/{TASK}/predictions_additional_data.tsv", sep="\t", header=False, index=False)

