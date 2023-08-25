from os.path import isdir
from os import mkdir
import re
import pandas as pd

TASK = "nermud"

GLOBAL_NOT_WELL_FORMATTED_COUNTER = 0


def clean_input_text(text):
    text = re.sub(r'\t+', ' ', re.sub(r'\n+', ' ', re.sub(r'\s+', " ", text)))
    text = text.rstrip()
    return text

def encode():
    data = dict()
    
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")
    
    for split in ['train', 'test']:
        data = dict()
        id = 0
        for corpus in ['degasperi', 'fiction', 'moro', 'wikinews']:
            with open(f"data/NERMuD/{corpus}_{split}.tsv", encoding="utf-8") as f:
                tokens = []
                ne_labels = []
                ne_mentions = []
                prev_label = "O"
                for line in f:
                    if line.strip() != "":
                        word, label = line.strip().split("\t")
                        tokens.append(word)
                        if label != "O":
                            if prev_label != label:
                                ne_labels.append(label)
                                ne_mentions.append([])
                            ne_mentions[-1].append(word)
                        prev_label = label
                    else:
                        sentence = " ".join(tokens)
                        ne = ""
                        for i in range(len(ne_labels)):
                            ne += f"[{ne_labels[i]}] {' '.join(ne_mentions[i])} "
                        ne = ne.strip()

                        data[id] = {
                            "text": sentence,
                            "label": ne
                            }
                        id += 1
                        tokens = []
                        prev_label = "O"
                        ne_labels = []
                        ne_mentions = []
                        print(id)

        outfile = "train" if split == "train" else "dev"
        with open(f"out/{TASK}/{outfile}.txt", "w", encoding="utf-8") as f_o:         
            for id, features in data.items():
                if features['label'] == "":
                    output = f"{outfile}_{id}\t{TASK}\t{features['text']}\tnessuna\n"
                else:
                    output = f"{outfile}_{id}\t{TASK}\t{features['text']}\t{features['label']}\n"
                f_o.write(output)



def extract_entities(label):
    # input example: [ORG] Partito popolare italiano [LOC] Italia
    # output example: [ (ORG, Partito popolare italiano), (LOC, Italia) ]
    entities = []
    while label != "" and label != " ":
        entity_label = ""

        try:
            # extract the label
            start_i = label.index("[")
            end_i = label.index("]", start_i)+1
            entity_label = label[start_i:end_i]
            # consume the label
            label = label.replace(entity_label, "", 1).strip()
            entity_label = entity_label.replace("[", "").replace("]", "")
        except:
            print("EXCEPTION not well formatted")
            print(label)
        
        # extract the span
        if "[" in label:
            span = label.split(" [")[0]
        else:
            span = label
        # consume the span
        label = label.replace(span, "", 1).strip()

        # add touple to list
        if entity_label != "":
            entities.append((entity_label, span))
    
    return entities



def get_words_labels(text, label):
    global GLOBAL_NOT_WELL_FORMATTED_COUNTER
    words, words_lower, labels = [], [], []
    # add O for every word
    for text_word in text.split(" "):
        words.append(text_word)
        words_lower.append(text_word.lower())
        labels.append("O")
    # if there are entities predicted, extract them
    if label != "nessuna" and label != "nessun" and label != "nessuno":
        entities_list = extract_entities(label.lower())
        for entity_type, span in entities_list:
            for i, span_word in enumerate(span.split()):
                try:
                    if i == 0:
                        labels[words_lower.index(span_word)] = f"B-{entity_type}"
                    else:
                        labels[words_lower.index(span_word)] = f"I-{entity_type}"
                except:
                    GLOBAL_NOT_WELL_FORMATTED_COUNTER += 1
    words.append("")
    labels.append("")

    return words, labels

def decode():
    
    words_ADG, labels_ADG = [], []
    words_FIC, labels_FIC = [], []
    words_WN, labels_WN = [], []
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split("\t")
            _, filetype = line_split[0].split("_")
            text = line_split[2]
            label = clean_input_text(line_split[3])
            text_words, text_labels = get_words_labels(text, label)

            if "ADG" in filetype:
                words_ADG.extend(text_words)
                labels_ADG.extend(text_labels)
            elif "FIC" in filetype:
                words_FIC.extend(text_words)
                labels_FIC.extend(text_labels)
            elif "WN" in filetype:
                words_WN.extend(text_words)
                labels_WN.extend(text_labels)
            else:
                print(f"NO FILETYPE FOUND\t{filetype}")
                quit()

    run = 2
    data_ADG = pd.DataFrame({
        "words": words_ADG,
        "labels": labels_ADG
    })
    data_ADG.to_csv(f"out/{TASK}/DAC_ADG_extremITA_{run}.tsv", sep="\t", header=False, index=False)
    data_FIC = pd.DataFrame({
        "words": words_FIC,
        "labels": labels_FIC
    })
    data_FIC.to_csv(f"out/{TASK}/DAC_FIC_extremITA_{run}.tsv", sep="\t", header=False, index=False)
    data_WN = pd.DataFrame({
        "words": words_WN,
        "labels": labels_WN
    })
    data_WN.to_csv(f"out/{TASK}/DAC_WN_extremITA_{run}.tsv", sep="\t", header=False, index=False)
    global GLOBAL_NOT_WELL_FORMATTED_COUNTER
    print(GLOBAL_NOT_WELL_FORMATTED_COUNTER)


