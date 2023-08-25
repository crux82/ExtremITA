import csv
import random
import pandas as pd
import re
from os.path import isdir
from os import mkdir

TASK = "discotex"
RUN_NUMBER = 1

random.seed(23)
TRAIN_DEV_SPLIT = 0.05

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")

    data_task1 = dict()
    for source in ["ted", "wiki"]:
        with open(f"data/{TASK}/task_1/discotex_1_{source}_train.tsv", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = clean_input_text(row['PROMPT'])+" [SEP] "+clean_input_text(row['TARGET'])
                label = "coerente" if row['CLASS'] == "1" else "non_coerente"
                data_task1[row["ID"]] = {
                    "text": text,
                    "label": label
                }

    data_task2 = dict()
    with open(f"data/{TASK}/task_2/discotex_2_train.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            score = eval(row['MEAN'])
            text = clean_input_text(row['TEXT'])
            data_task2[row["ID"]] = {
                "text": text,
                "score": score
            }

    with open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:

        for id, item in data_task1.items():
            line = f"{id}\tdiscotex_1\t{item['text']}\t{item['label']}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(line)
            else:
                f_train.write(line)


        for id, item in data_task2.items():
            line = f"{id}\tdiscotex_2\t{item['text']}\t{item['score']}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(line)
            else:
                f_train.write(line)

             
def decode():
    """
    Reads the file from 'out/discotex/predictions.txt'. \\
    Divides the examples based on task_a or task_b. \\
    Creates 2 TSV file:
        - 'out/discotex/task_a_predictions.tsv' in the form 'id, label' with no header;
        - 'out/discotex/task_b_predictions.tsv' in the form 'id, score' with no header;
    """
    
    ids_task_a, labels_task_a, ids_task_b, labels_task_b = [], [], [], []
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            id = line_split[0]
            task = line_split[1]
            if task == "discotex_1":
                label = 0 if "non_coerente" in line_split[3] else 1
                ids_task_a.append(id)
                labels_task_a.append(label)
            else:
                label = clean_input_text(line_split[3])
                ids_task_b.append(id)
                labels_task_b.append(label)

    data_task_a = pd.DataFrame({
        "ID": ids_task_a,
        "CLASS": labels_task_a
    })
    data_task_a.to_csv(f"out/{TASK}/extremITA.subtask1.run{RUN_NUMBER}.tsv", header=False, index=False, encoding="utf-8", sep="\t")
    
    data_task_b = pd.DataFrame({
        "ID": ids_task_b,
        "SCORE": labels_task_b
    })
    data_task_b.to_csv(f"out/{TASK}/extremITA.subtask2.run{RUN_NUMBER}.tsv", header=False, index=False, encoding="utf-8", sep="\t")

