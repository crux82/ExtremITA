import csv
import random
import pandas as pd
import json
import re
from os.path import isdir
from os import mkdir

random.seed(23)
TRAIN_DEV_SPLIT = 0.05

TASK = "hodi"

def clean_input_text(text):
    #avoid removing multiple space as it would lose alignment
    text = re.sub(r'\t', ' ', re.sub(r'\n', ' ', text)) 
    text = text.rstrip()
    return text

def encode():
    data = dict()
    
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
        
    with open("data/HODI/HODI_2023_train_subtaskA.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = clean_input_text(row['text'])
            label = "omotransfobico" if row['homotransphobic'] == "1" else "non_omotransfobico"
            
            data[row["id"]] = {
                "text": text,
                "label": label
            }
                        
    with open("data/HODI/HODI_2023_train_subtaskB.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = clean_input_text(row['text'])
            if row["rationales"] != "":
                indices = eval(row["rationales"])
                last_index = indices[0]
                label = ""
                for n, i in enumerate(indices):
                    if n > 0 and i != indices[n-1]+1:
                        label += " [gap] "
                    label += text[i]
                data[row["id"]]["rationale"] = clean_input_text(label)

    with open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:            

        for id, features in data.items():
            if "rationale" in features:
                output = f"{id}\thodi_a\t{features['text']}\t{features['label']}\n{id}\thodi_b\t{features['text']}\t{features['rationale']}\n"
            else:
                output = f"{id}\thodi_a\t{features['text']}\t{features['label']}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(output)
            else:
                f_train.write(output)


def get_indexes_from_text(text, label):
    label_split = label.split(" [gap] ")
    matches = []
    for pattern in label_split:
        tokens = pattern.split()
        running_pattern = []
        final_pattern = []
        match = True
        while match:
            running_pattern.append(tokens.pop(0))
            
            match = " ".join(running_pattern) in text
            if match:
                final_pattern.append(running_pattern[-1])
            match = match and len(tokens) > 0

        found = " ".join(final_pattern)
        matches.append(re.search(re.escape(found), text))  
    indexes = []
    for m in matches:
        if m != None:
            l_index, r_index = m.span()
            j = l_index
            while j < r_index:
                indexes.append(j)
                j += 1
    return indexes

def decode():
    """
    Reads the file from 'out/hodi/predictions.txt'. \\
    Divides the examples based on task_a or task_b. \\
    Creates 2 TSV file:
        -'out/hodi/task_a_predictions.tsv' in the form 'id, homotransphobic' with no header. \\
        -'out/hodi/task_b_predictions.tsv' in the form 'id, rationales' with no header.
    """
    ids_task_a, labels_task_a, ids_task_b, labels_task_b = [], [], [], []
    ids_task_a_non_homotransphobic = [] # this examples are non homotransphobic => no spans to predict for task b
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            id = line_split[0]
            label = clean_input_text(line_split[3])
            task = line_split[1]
            if task == "hodi_a":
                ids_task_a.append(id)
                numerical_label = 0 if "non_omotransfobico" in label else 1
                if numerical_label == 0:
                    ids_task_a_non_homotransphobic.append(id)
                labels_task_a.append(numerical_label)
            else:
                ids_task_b.append(id)
                text = line_split[2]
                if id in ids_task_a_non_homotransphobic:
                    final_label = "[]"
                else:
                    indexes = list(set(get_indexes_from_text(text.lower(), label.lower())))
                    indexes.sort()
                    final_label = json.dumps(indexes) # converts array into stringified array with brackets
                labels_task_b.append(final_label)

    data_task_a = pd.DataFrame({
        "id": ids_task_a,
        "homotransphobic": labels_task_a
    })
    data_task_a.to_csv(f"out/{TASK}/extremITA.A.run1.tsv", index=False, sep="\t")
    
    data_task_b = pd.DataFrame({
        "id": ids_task_b,
        "rationales": labels_task_b
    })
    data_task_b.to_csv(f"out/{TASK}/extremITA.B.run1.tsv", index=False, sep="\t")

