import random
from os.path import isdir
from os import mkdir
import spacy
import re

nlp = spacy.load("it_core_news_sm", disable=["lemmatizer", "tagger"])

TASK = "clinkart"
TRAIN_DEV_SPLIT = 0.05

random.seed(23)

def get_offset_from_split(text, MIN_CHARS= 50, MAX_TOKENS = 30):
    res = []
    l = 0
    newsent = {
        "from": 0,
        "to": 0
    }

    doc = nlp(text)

    sentences = [sent for sent in doc.sents]

    for s_id, sent in enumerate(sentences):

        if newsent["to"] - newsent["from"] < MIN_CHARS or l + len(sent) < MAX_TOKENS:
            newsent = {
                "from": newsent["from"],
                "to": sent.end_char
            }
            l = l + len(sent)
        else:
            res.append(newsent)
            newsent = {
                "from": sent.start_char,
                "to": sent.end_char
            }
            l = len(sent)

        # if last sentence is alone, add it as it is
        if s_id == len(sentences) - 1:
            res.append(newsent)   

    return res

def encode():
    if not isdir(f"out/{TASK}"):
        mkdir(f"out/{TASK}")

    data = dict()

    files = dict()
    files[TASK] = "data/CLinkaRT/Clinkart_training_data/training.txt"
    files["testlinkes"] = "data/CLinkaRT/TESTLINK_ES_training_data_v1.1/TESTLINK_training_data/training.txt"
    files["testlinkeu"] = "data/CLinkaRT/TESTLINK_EU_training_data/training.txt"

    for filetype in files:
        file = files[filetype]
        with open(file, encoding="utf-8") as f:
            sentences = dict()
            for line in f:
                if line != "\n":
                    if "|t|" in line:
                        document_id = filetype + "_" +line.split("|")[0]
                        text = "|".join(line.split("|")[2:]).strip()                                
                        sentences[document_id] = get_offset_from_split(text)                        
                    else:
                        document_id, _, span1, span2, tokens1, tokens2 = line.strip().split("\t")  
                        document_id = filetype + "_" + document_id                  
                        from1, to1 = map(eval, span1.split("-"))
                        from2, to2 = map(eval, span2.split("-"))
                        for sent in sentences[document_id]:
                            # final sentence_id will be the document id concatenated with the starting and ending token of the sentence
                            sentence_id = f"{document_id}_{sent['from']}_{sent['to']}"
                            # initialize object for the sentence
                            if sentence_id not in data.keys():
                                data[sentence_id] = {
                                    'text': text[sent["from"]:sent["to"]],
                                    "relations": []
                                }
                            # if the span of the tokens is contained in this sentence, add the relation
                            if min(from1, from2) >= sent["from"] and max(to1, to2) <= sent["to"]:
                                data[sentence_id]['relations'].append((from1, tokens1, tokens2, "RML", "EVENT"))

    with open(f"out/{TASK}/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"out/{TASK}/dev.txt", "w", encoding="utf-8") as f_dev:
        for id, item in data.items():
            label = ""

            relations = item["relations"]
            relations = sorted(relations, key=lambda from_id: from_id[0])
            for relation in relations:
                _, tokens1, tokens2, _, _ = relation
                label += f" [BREL] {tokens1} [SEP] {tokens2} [EREL]"
            
            # if the relations field is empty the label is [NOREL]
            label = label.lstrip() if relations else "[NOREL]"
            line = f"{id}\t{TASK}\t{item['text']}\t{label}\n"

            if id.startswith(TASK) and random.random()<TRAIN_DEV_SPLIT:
                f_dev.write(line)
            else:
                f_train.write(line)


def decode():
    with open(f"out/{TASK}/test_preds.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        out = dict()
        texts = dict()
        for line in lines:
            relations = []
            line_split = line.split("\t")

            id = line_split[0]
            text = line_split[2]
            prediction = line_split[3]
           
            doc_id, char_from, char_to = id.split("_")
            if not doc_id in out:
                out[doc_id] = []
            if not doc_id in texts:
                texts[doc_id] = ""
            texts[doc_id] += text
                
            char_from = eval(char_from)
            char_to = eval(char_to)
            regex = re.compile(r"\[BREL\].*?\[SEP\].*?\[EREL\]")
            
            matched_list = re.findall(regex, prediction)
            for matched in matched_list:
                valid = True
                try:
                    tmp = re.sub("^\[BREL\] ", "", matched)
                    tmp = re.sub(" \[EREL\]$", "", tmp)
                    brel, erel = tmp.split(" [SEP] ")
                except:
                    valid = False
                
                if "[BREL]" in brel or "[SEP]" in brel or "[EREL]" in brel or "[BREL]" in erel or "[SEP]" in erel or "[EREL]" in erel:
                    valid = False
                if valid:
                    relations.append((brel, erel))

            for brel, erel in relations:
                try:
                    m_from_brel, m_to_brel = re.search(r"\b{}\b".format(brel), text).span()
                    
                    # cerca l'erel più vicino al brel
                    min_dist = 10000000
                    for m in re.finditer(r"\b{}\b".format(erel), text):
                        f, _ = m.span()
                        if abs(f-m_from_brel)<min_dist:
                            min_dist = abs(f-m_from_brel)
                            m_from_erel, m_to_erel = m.span()
                except:
                    continue # questo può succedere se il transformer allucina il testo
                
                # corregge l'offset rispetto alla frase
                m_from_brel += char_from
                m_from_erel += char_from
                m_to_brel += char_from
                m_to_erel += char_from
                
                obj = (brel, erel, m_from_brel, m_to_brel, m_from_erel, m_to_erel)
                if obj not in out[doc_id]:
                    out[doc_id].append(obj)

    with open(f"out/{TASK}/ExtremITA.txt", "w", encoding="utf-8") as fo:
        for doc_id, sentences in out.items():
            text = texts[doc_id]
            fo.write(f"{doc_id}|t|{text}\n")
            for brel, erel, m_from_brel, m_to_brel, m_from_erel, m_to_erel in sentences:
                fo.write(f"{doc_id}\tREL\t{m_from_brel}-{m_to_brel}\t{m_from_erel}-{m_to_erel}\t{brel}\t{erel}\n")
            fo.write(f"\n")

