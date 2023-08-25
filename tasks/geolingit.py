import csv
import math
import re
from os.path import isdir
from os import mkdir
import pandas as pd

TASK = "geolingit"

def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
        
    for split in ['dev', 'train']:
        data = dict()
        
        with open(f"data/GeoLingIt/subtask_a/{split}_a.tsv", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                text = clean_input_text(row['text'])
                label = row['region']
                data[row['id']] = {
                    'text': text,
                    'label': label,
                }
                
        with open(f"data/GeoLingIt/subtask_b/{split}_b.tsv", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                latitude = math.floor(eval(row['latitude'])*100)/100.
                longitude = math.floor(eval(row['longitude'])*100)/100.
                data[row['id']]['latitude'] = latitude
                data[row['id']]['longitude'] = longitude

        with open(f"out/geolingit/{split}.txt", "w", encoding="utf-8") as f_o:
            for id, features in data.items():
                f_o.write(f"{id}\tgeolingit\t{features['text']}\t[regione] {features['label']} [geo] {features['latitude']} {features['longitude']}\n")




def decode():
    """
    subtask A\\
    id, text, region\\
    12, In.., Lazio

    subtask B\\
    id, text, lat, lon\\
    12, In.., 41.8984164, 12.54514535
    """


    ids, texts, regions, lats, lons = [], [], [], [], []
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")

            id = line_split[0]
            text = line_split[2]

            label_raw = line_split[3] # [regione] Lombardia [geo] 45.46 9.15
            try:
                region = label_raw.split("[geo]")[0].split("regione]")[1].strip()
            except IndexError:
                region = "Lazio"

            try:
                coordinates = label_raw.split("[geo]")[1].strip()
                lat, lon = coordinates.split(" ")
            except IndexError:
                lat, lon = "41.8984164", "12.54514535"

            # Una nota: nel dataset ci sono il 30/40% dei tweet scritti in 2 punti
            # 1607 [regione] Campania [geo] 40.8541123 14.24345155 (Napoli centro?)
            # 4359 [regione] Lazio [geo] 41.8984164 12.54514535 (Roma centro?)
            if region == "Campania" and lat == "40.85" and lon == "14.24":
                lat, lon = "40.8541123", "14.24345155"
            elif region == "Lazio" and lat == "41.89" and lon == "12.54":
                lat, lon = "41.8984164", "12.54514535"
            
            ids.append(id)
            texts.append(text)
            regions.append(region)
            lats.append(lat)
            lons.append(lon)

    data_subtaskA = pd.DataFrame({
        "id": ids, 
        "text": texts, 
        "region": regions
    })
    data_subtaskA.to_csv(f"out/{TASK}/extremITA.standard.a.2.tsv", sep="\t", index=False, encoding="utf-8")

    data_subtaskB = pd.DataFrame({
        "id": ids, 
        "text": texts, 
        "latitude": lats,
        "longitude": lons
    })
    data_subtaskB.to_csv(f"out/{TASK}/extremITA.standard.b.2.tsv", sep="\t", index=False, encoding="utf-8")
    print(50*"*")
    print("REMEMBER TO ZIP THE FILES BEFORE SUBMITTING AND BEFORE GENERATING ANOTHER ONE!")
    print(50*"*")

