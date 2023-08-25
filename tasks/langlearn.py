import csv
from os.path import isdir
from os import mkdir
import xml.etree.ElementTree as ET
import random
import spacy
import re
from tqdm import tqdm
import pandas as pd

random.seed(23)
MAX_TEXT_LENGTH = 100 # it is 100 per sentence
TRAIN_DEV_SPLIT = 0.05

nlp = spacy.load("it_core_news_sm", disable=["lemmatizer", "tagger"])

TASK = "langlearn"

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def shorten(text, tokens=MAX_TEXT_LENGTH):
    doc = nlp(text)
    output = ""
    n_tokens = 0
    for sent in doc.sents:
        if n_tokens + len(sent) > tokens:
            break
        output += " "+sent.text
        n_tokens += len(sent)
    return output.lstrip()

def encode():
    data = dict()
    
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
        
    data = dict()
    
    for corpus in ['CItA', 'COWS-L2H']:
        texts = dict()
        
        tree = ET.parse(f'data/LangLearn/{corpus}/Essays_{corpus}.xml')
        root= tree.getroot()
        for content in root.iter('doc'):
            texts[content.get('id')] = content.text

        filename = "Training" if corpus == "CItA" else "Train"
        with open(f"data/LangLearn/{corpus}/{filename}_{corpus}.tsv", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in tqdm(reader):
                id = f"{row['Essay_1']}-{row['Essay_2']}"
                text1 = clean_input_text(texts[row["Essay_1"]])
                text2 = clean_input_text(texts[row["Essay_2"]])
                
                text1 = shorten(text1)
                text2 = shorten(text2)
                
                text = f"{text1} [SEP] {text2}"
                data[id] = {
                    "text": text,
                    "label": "corretto"
                }

                id_reverse = f"{row['Essay_2']}-{row['Essay_1']}"
                text_reverse = f"{text2} [SEP] {text1}"
                data[id_reverse] = {
                    "text": text_reverse,
                    "label": "incorretto"
                }

    with open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:
        for id, item in data.items():
            line = f"{id}\t{TASK}\t{item['text']}\t{item['label']}\n"
            if random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(line)
            else:
                f_train.write(line)        



def decode():
    """
    2 files (CItA and COWS-L2H) with the following header \\
    Essay_1 Essay_2   Order_1 Order_2   Pred\\
    123     321         1_2     1_3     0       => in ordine corretto\\
    321     123         1_2     1_3     1       => in ordine non corretto
    """

    cita_df = pd.read_csv(f"data/LangLearn/CItA/Test_CItA.tsv", sep="\t", encoding="utf-8")
    cita_dict = dict()
    for id1, id2, order1, order2 in zip(cita_df['Essay_1'], cita_df['Essay_2'], cita_df['Order_1'], cita_df['Order_2']):
        cita_dict[f'{id1}-{id2}'] = {
            "order1": order1,
            "order2": order2
        }

    cows_l2h_df = pd.read_csv(f"data/LangLearn/COWS-L2H/Test_COWS-L2H.tsv", sep="\t", encoding="utf-8")
    cows_l2h_dict = dict()
    for id1, id2, order1, order2 in zip(cows_l2h_df['Essay_1'], cows_l2h_df['Essay_2'], cows_l2h_df['Order_1'], cows_l2h_df['Order_2']):
        cows_l2h_dict[f'{id1}-{id2}'] = {
            "order1": order1,
            "order2": order2
        }

    ids1_cita, ids2_cita, order1_cita, order2_cita, labels_cita = [], [], [], [], []
    ids1_cows_l2h, ids2_cows_l2h, order1_cows_l2h, order2_cows_l2h, labels_cows_l2h = [], [], [], [], []
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            id = line_split[0]
            id1, id2 = id.split("-")
            label = 1 if "incorretto" in clean_input_text(line_split[3]) else 0
            if id in cita_dict:
                ids1_cita.append(id1)
                ids2_cita.append(id2)
                order1_cita.append(cita_dict[id]['order1'])
                order2_cita.append(cita_dict[id]['order2'])
                labels_cita.append(label)
            elif id in cows_l2h_dict:
                ids1_cows_l2h.append(id1)
                ids2_cows_l2h.append(id2)
                order1_cows_l2h.append(cows_l2h_dict[id]['order1'])
                order2_cows_l2h.append(cows_l2h_dict[id]['order2'])
                labels_cows_l2h.append(label)
            else:
                print(50*"*")
                print("ERROR")
                print(f"{id1}-{id2} not in any dict")
                print(50*"*")
            
    assert len(ids1_cita) == len(ids2_cita) == len(order1_cita) == len(order2_cita) == len(labels_cita), \
        f"len is {len(ids1_cita)} != {len(ids2_cita)} != {len(order1_cita)} != {len(order2_cita)} != {len(labels_cita)}"
    assert len(ids1_cows_l2h) == len(ids2_cows_l2h) == len(order1_cows_l2h) == len(order2_cows_l2h) == len(labels_cows_l2h), \
        f"len is {len(ids1_cows_l2h)} != {len(ids2_cows_l2h)} != {len(order1_cows_l2h)} != {len(order2_cows_l2h)} != {len(labels_cows_l2h)}"
    
    data_cita = pd.DataFrame({
        "Essay_1": ids1_cita,
        "Essay_2": ids2_cita,
        "Order_1": order1_cita,
        "Order_2": order2_cita,
        "Label": labels_cita
    })
    data_cita.to_csv(f"out/{TASK}/Test_CItA.tsv", index=False, sep="\t")
    
    data_cows_l2h = pd.DataFrame({
        "Essay_1": ids1_cows_l2h,
        "Essay_2": ids2_cows_l2h,
        "Order_1": order1_cows_l2h,
        "Order_2": order2_cows_l2h,
        "Label": labels_cows_l2h
    })
    data_cows_l2h.to_csv(f"out/{TASK}/Test_COWS-L2H.tsv", index=False, sep="\t")


