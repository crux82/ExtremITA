import csv
import re
import random
import pandas as pd
from os.path import isdir
from os import mkdir

random.seed(23)

TASK = "acti"
TRAIN_DEV_SPLIT = 0.05

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
    
    data_task1 = dict()
    with open(f"data/{TASK}/acti-subtask-a/subtaskA_train.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = clean_input_text(row['comment_text'])
            label = "cospirazione" if row['conspiratorial'] == "1" else "non_cospirazione"
            
            data_task1[row["Id"]] = {
                "text": text,
                "label": label
            }
             
    data_task2 = dict()           
    with open(f"data/{TASK}/acti-subtask-b/subtaskB_train.csv", encoding="utf-8") as f2:
        reader2 = csv.DictReader(f2)
        for row in reader2:
            id = row['Id']
            text = clean_input_text(row['comment_text'])
            topic = row['topic']

            data_task2[id] = {
                "text": text,
                "topic": topic,
            }

            # check by text if example is already present in task 1
            # else add it as an example for task 1 with label "cospirazione" as well
            found_in_task1 = False
            for _, item in data_task1.items():
                if item['text'] == text:
                    found_in_task1 = True
            if not found_in_task1:
                data_task2[id]['task_1_label'] = "cospirazione"


    with open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:            

        for id, features in data_task1.items():
            output = f"{id}\tacti_a\t{features['text']}\t{features['label']}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(output)
            else:
                f_train.write(output)
            
        for id, features in data_task2.items():
            if "task_1_label" in features.keys():
                output = f"{id}\tacti_b\t{features['text']}\t{features['topic']}\n{id}\tacti_a\t{features['text']}\t{features['task_1_label']}\n"
            else:
                output = f"{id}\tacti_b\t{features['text']}\t{features['topic']}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(output)
            else:
                f_train.write(output)


def decode():
    """
    Reads the file from 'out/acti/predictions.txt'. \\
    Divides the examples based on task_a or task_b. \\
    Creates 2 CSV file in 'out/acti/task_a_predictions.csv' and 'out/acti/task_b_predictions.csv' \\
        to submit in the form 'id, label' with no header.
    """

    label_map = {
        "Covid": 0,
        "Qanon": 1,
        "Climate Change": 1,
        "Russia": 2,
        "Ucraina": 2,
        "Terra Piatta": 3,
    }
    
    ids_task_a, labels_task_a, ids_task_b, labels_task_b = [], [], [], []
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            id = line_split[0]
            task = line_split[1]
            if task == "acti_a":
                label = 0 if "non_cospirazione" in clean_input_text(line_split[3]) else 1
                ids_task_a.append(id)
                labels_task_a.append(label)
            else:
                if clean_input_text(line_split[3]) in label_map:
                    label = label_map[clean_input_text(line_split[3])]
                else:
                    label = 0 # if the label is not well formatedd add the most frequent class (0=Covid)
                ids_task_b.append(id)
                labels_task_b.append(label)

    data_task_a = pd.DataFrame({
        "Id": ids_task_a,
        "Expected": labels_task_a
    })
    data_task_a.to_csv(f"out/{TASK}/extremITA_subtaskA.csv", index=False)
    
    data_task_b = pd.DataFrame({
        "Id": ids_task_b,
        "Expected": labels_task_b
    })
    data_task_b.to_csv(f"out/{TASK}/extremITA_subtaskB.csv", index=False)

