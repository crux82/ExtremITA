import csv
import random
import math
import re
from os.path import isdir
from os import mkdir
import pandas as pd

random.seed(23)
TRAIN_DEV_SPLIT = 0.05

TASK = "emotivita"

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
    with open(f"data/{TASK}/Development set.csv", encoding="utf-8") as f, \
         open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:
        reader = csv.DictReader(f)
        for row in reader:
            text = clean_input_text(row['text'])
            v = math.floor(eval(row['V'])*10)/10.
            a = math.floor(eval(row['A'])*10)/10.
            d = math.floor(eval(row['D'])*10)/10.
            line = f"{row['id']}\t{TASK}\t{text}\t{v} {a} {d}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(line)
            else:
                f_train.write(line)   



def decode():
    ids, Vs, As, Ds, = [], [], [], []
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            id = line_split[0]
            label = line_split[3].split(" ")
            if len(label) != 3:
                print(line)
            
            try:
                V = clean_input_text(label[0])
            except IndexError:
                V = "3.5"
            try:
                A = clean_input_text(label[1])
            except IndexError:
                A = "3.5"
            try:
                D = clean_input_text(label[2])
            except IndexError:
                D = "3.5"
            
            ids.append(id)
            Vs.append(V)
            As.append(A)
            Ds.append(D)

    data = pd.DataFrame({
        "id": ids,
        "V": Vs,
        "A": As,
        "D": Ds
    })
    run = 2
    data.to_csv(f"out/{TASK}/extremITA_SubtaskB_unconstrained_run{run}.csv", index=False)
    print("REMEMBER TO ZIP THE FILES BEFORE SUBMITTING!")

