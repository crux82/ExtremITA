import csv
import random
import re
import emoji
import pandas as pd
from os.path import isdir
from os import mkdir

random.seed(23)
TRAIN_DEV_SPLIT = 0.05

TASK = "emit"
SUBMISSION_TYPE = "in-domain"

emo_map = {
    "Anger": "rabbia",
    "Anticipation": "anticipazione",
    "Disgust": "disgusto",
    "Fear": "paura",
    "Joy": "gioia",
    "Love": "amore",
    "Neutral": "neutro",
    "Sadness": "tristezza",
    "Surprise": "sorpresa",
    "Trust": "fiducia"
}

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")

    data = dict()
    with open(f"data/{TASK}/emit_train_A.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = clean_input_text(row['text'])
            text = emoji.demojize(text, language="it")
            labels = []
            for emotion in "Anger,Anticipation,Disgust,Fear,Joy,Love,Neutral,Sadness,Surprise,Trust".split(","):
                if row[emotion] == "1":
                    labels.append(emo_map[emotion])
            labels.sort()
            label = " ".join(labels)
            data[row['id']] = {
                "text": text,
                "label": label
            }

    with open(f"data/{TASK}/emit_train_B.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Direction'] == "1" and row['Topic'] == "1":
                data[row['id']]["topic"] = "direzione_argomento"
            elif row['Direction'] == "1":
                data[row['id']]["topic"] = "direzione"
            elif row['Topic'] == "1":
                data[row['id']]["topic"] = "argomento"
            else:
                data[row['id']]["topic"] = "non_specificato"
            
    with open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:
        for id, item in data.items():
            line = f"{id}\temit_a\t{item['text']}\t{item['label']}\n{id}\temit_b\t{item['text']}\t{item['topic']}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(line)
            else:
                f_train.write(line)



def decode():
    """
    Reads the file from 'out/emit/predictions.txt'. \\
    Divides the examples based on task_a or task_b. \\
    Creates a CSV with the name reporting: 
        - the name of the team, 
        - the addressed subtask (A or B), 
        - the ID of the run (1 or 2) and
        - the genre of the test set (in-domain or out-of-domain)\\
    Creates 2 CSV file in 
        - 'out/emit/extremITA_SubtaskA_in-domain_run1.csv'
        - 'out/emit/extremITA_SubtaskB_in-domain_run1.csv.csv'
    """

    # mapping from challenge labels to model-specific generations
    subtask_a_mapping = {
        "Anger": "rabbia",
        "Anticipation": "anticipazione",
        "Disgust": "disgusto",
        "Fear": "paura",
        "Joy": "gioia",
        "Love": "amore",
        "Neutral": "neutro",
        "Sadness": "tristezza",
        "Surprise": "sorpresa",
        "Trust": "fiducia"
    }
    # mapping from challenge labels to model-specific generations
    # need an array for the mappings as for both classes the model will generate "entrambi"
    subtask_b_mapping = {
        "Direction": ["direzione", "entrambi"],
        "Topic": ["argomento", "entrambi"]
    }
    
    task_a_ids, anger_list, anticipation_list, disgust_list, fear_list, joy_list, love_list, neutral_list, sadness_list, surprise_list, trust_list = [], [], [], [], [], [], [], [], [], [], []
    task_b_ids, direction_list, topic_list = [], [], []
    with open(f"out/{TASK}/{SUBMISSION_TYPE}_test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            id = line_split[0]
            label = clean_input_text(line_split[3])
            task = line_split[1]
            if task == "emit_a":
                task_a_ids.append(id)

                # check if model-specific labels are contained in the model generated text and append 1
                # else append 0
                anger_list.append(1 if subtask_a_mapping['Anger'] in label else 0)
                anticipation_list.append(1 if subtask_a_mapping['Anticipation'] in label else 0)
                disgust_list.append(1 if subtask_a_mapping['Disgust'] in label else 0)
                fear_list.append(1 if subtask_a_mapping['Fear'] in label else 0)
                joy_list.append(1 if subtask_a_mapping['Joy'] in label else 0)
                love_list.append(1 if subtask_a_mapping['Love'] in label else 0)
                neutral_list.append(1 if subtask_a_mapping['Neutral'] in label else 0)
                sadness_list.append(1 if subtask_a_mapping['Sadness'] in label else 0)
                surprise_list.append(1 if subtask_a_mapping['Surprise'] in label else 0)
                trust_list.append(1 if subtask_a_mapping['Trust'] in label else 0)
            else:
                task_b_ids.append(id)
                
                # there are 2 ways of defining the "direction" and the "topic"
                # need to loop through the array and find the first match
                # append 1 if model-specific labels are contained in the model generated text
                # else append 0
                direction_value = 0
                for direction_mapping in subtask_b_mapping["Direction"]:
                    if direction_mapping in label:
                        direction_value = 1
                direction_list.append(direction_value)

                topic_value = 0
                for topic_mapping in subtask_b_mapping["Topic"]:
                    if topic_mapping in label:
                        topic_value = 1
                topic_list.append(topic_value)

    # create dataframes for both the subtasks with the specified headers
    # and write them to file
    data_task_a = pd.DataFrame({
        "id": task_a_ids,
        "Anger": anger_list,
        "Anticipation": anticipation_list,
        "Disgust": disgust_list,
        "Fear": fear_list,
        "Joy": joy_list,
        "Love": love_list,
        "Neutral": neutral_list,
        "Sadness": sadness_list,
        "Surprise": surprise_list,
        "Trust": trust_list
    })
    data_task_a.to_csv(f'out/{TASK}/extremITA_SubtaskA_{SUBMISSION_TYPE}_run2.csv', index=False)
    
    data_task_b = pd.DataFrame({
        "id": task_b_ids,
        "Direction": direction_list,
        "Topic": topic_list
    })
    data_task_b.to_csv(f'out/{TASK}/extremITA_SubtaskB_{SUBMISSION_TYPE}_run2.csv', index=False)

