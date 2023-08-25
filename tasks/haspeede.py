import csv
import random
import pandas as pd
import re
from os.path import isdir
from os import mkdir

random.seed(23)
TRAIN_DEV_SPLIT = 0.05

TASK = "haspeede"

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
        
    with open("data/HaSpeeDe3/development/training_textual.csv", encoding="utf-8") as f, \
         open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:
        reader = csv.DictReader(f)
        for row in reader:
            text = clean_input_text(row['anonymized_text'])
            label = "odio" if row['label'] == "1" else "non_odio"
            output = f"{row['anonymized_tweet_id']}\t{TASK}\t{text}\t{label}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(output)
            else:
                f_train.write(output)
            


def decode():
    
    subtasks = ["A", "B_religious"]
    for subtask in subtasks:
        anonymized_tweet_ids, labels = [], []
        with open(f"out/{TASK}/test_subtask{subtask}_preds.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()

            for line in lines:
                line_split = line.split("\t")
                id = line_split[0]
                anonymized_tweet_ids.append(id)
                label = 0 if "non_odio" in line_split[-1] else 1
                labels.append(label)

        df = pd.DataFrame({
            "anonymized_tweet_id": anonymized_tweet_ids,
            "label": labels
        })
        type = "textual" if subtask == "A" else ("XPoliticalHate" if subtask == "B_political" else "XReligiousHate")
        subtask = subtask.replace("_religious", "")
        run = "2"
        df.to_csv(f"out/{TASK}/extremITA_task{subtask}_{type}_run_{run}.csv", index=False)

        # create subtaskB_XPoliticalHate file from subtaskA file since the test set and the predictions are the same
        if subtask == "A":
            df.to_csv(f"out/{TASK}/extremITA_taskB_XPoliticalHate_run_{run}.csv", index=False)


