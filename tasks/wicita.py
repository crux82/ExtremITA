import json
from os.path import isdir
from os import mkdir
import pandas as pd

TASK = "wicita"
LAN = "it" # "en"

def highligths(string, start, end, id):
    byte_range = [start, end]  # bytes 12-16, inclusive

    # get the substrings
    substr = string[byte_range[0]:byte_range[1]+1]  # +1 to include the last byte

    # add tags to the substrings
    tagged_substr = f"[BT{id}] {substr} [ET{id}]"

    return string[:byte_range[0]] + tagged_substr + string[byte_range[1]:]


def encode():
    data = dict()
    
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
    
    for split in ["dev", "train"]:
        data = dict()
                                        
        with open(f"data/WiC-ITA/binary/{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                label = "differente" if item['label']==0 else "uguale"
                

                start1 = item['start1']
                end1 = item['end1']
                start2 = item['start2']
                end2 = item['end2']

                sent1 = highligths(item['sentence1'], start1, end1, 1)
                sent2 = highligths(item['sentence2'], start2, end2, 2)

                text = f"{sent1} [SEP] {sent2}"

                data[item['id']] = {
                    "text": text,
                    "label": label
                }

                if split == "train":
                    text = f"{sent2} [SEP] {sent1}"

                    data[item['id']+ "_inv"] = {
                        "text": text,
                        "label": label
                    }
         
        with open(f"out/{TASK}/{split}.txt", "w", encoding="utf-8") as f_o:           
            for id, features in data.items():
                output = f"{id}\t{TASK}\t{features['text']}\t{features['label']}\n"           
                f_o.write(output)


            
def decode():
    """
    Reads the file from 'out/wicita/predictions.txt'. \\
    Creates a jsonl file in 'out/wicita/predictions.tsv' \\
        to submit in the form 'id, label' with no header.
    """
    
    ids, labels = [], []
    lan = "" if LAN == "it" else "_eng"
    with open(f"out/{TASK}/test{lan}_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            id = line_split[0]
            label = 1 if "uguale" in line_split[3] else 0
            ids.append(id)
            labels.append(label)

    data = pd.DataFrame({
        "id": ids,
        "label": labels
    })
    data.to_json(f"out/{TASK}/binary{lan}.jsonl", orient="records", lines=True)

