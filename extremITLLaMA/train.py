import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
import csv
import json

from utils import *
 
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
 
import torch
from datasets import load_dataset
import pandas as pd

#============================================
#               PARAMETERS
#============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKENIZER_MODEL = "yahma/llama-7b-hf"
BASE_MODEL = "sag-uniroma2/extremITA-Camoscio-7b"

input_train_path = "data/train.txt"
input_dev_path = "data/dev.txt"
OUTPUT_DIR = "models/extremITLLaMA"

CUTOFF_LEN = 512
CUT_EXTREMITA_INPUT_CHAR_LENGTH = 1200

task = "*"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]
 
EPOCHS = 2 #3
BATCH_SIZE = 32 #128
MICRO_BATCH_SIZE = 16 #32 #4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.1

tmp_train_file_name = "tmp_train.json"
tmp_dev_file_name = "tmp_dev.json"

#============================================
#               FUNCTIONS
#============================================

#LOAD INPUT TSV files in the extremITA format 
def load(input_file_path):
    dataset_df = pd.read_csv(input_file_path, header=None, usecols=[0,1, 2, 3], names=['0', '1', '2', '3'], \
                             sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8').astype(str)
    dataset_df = dataset_df.rename(
        columns={"0": "id", "1": "prefix", "2": "input_text", "3": "target_text"}
    )
    dataset_df = dataset_df[["id", "input_text", "target_text", "prefix"]]
    return dataset_df

 
# Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
def tokenize(prompt, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result
 
# Notice: result["labels"] is rewritten so that only the output is considered
def generate_and_tokenize_prompt(data_point, add_eos_token=True):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, CUTOFF_LEN)

    user_prompt = generate_prompt_str(
        data_point["instruction"], data_point["input"]
    )
    tokenized_user_prompt = tokenize(
        user_prompt, CUTOFF_LEN, add_eos_token=add_eos_token
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably
    return tokenized_full_prompt

    

def load_and_prepare_data(input_file_path: str, tasks):

    df = load(input_file_path)

    if isinstance(tasks, str):
        if(tasks != "*"):
            df = df[df["prefix"]==tasks]
    elif isinstance(tasks, list):
        tmp = None
        for task in tasks:
            if tmp == None:
                tmp = df[df["prefix"]==task]      
            else:
                tmp += df[df["prefix"]==task]
        df = tmp

    print(df.target_text.value_counts())

    dataset_data = [
        {
            "instruction": task_to_prompt(row_dict["prefix"]),
            "input": row_dict["input_text"],
            "output": target_text_to_answer(row_dict["target_text"], row_dict["prefix"])
        }
        for row_dict in df.to_dict(orient="records")
    ]

    return dataset_data

def trim_long_input(json_input, cutoff_len=10000000):
    for json_data in json_input:
        json_data["input"] = json_data["input"][:cutoff_len]
    return json_input

#============================================
#                   MAIN
#============================================

#-------------------
#    LOAD DATA 
#-------------------
train_data = load_and_prepare_data(input_train_path, task)
dev_data = load_and_prepare_data(input_dev_path, task)


with open(tmp_train_file_name, "w") as f:
   json.dump(train_data, f)
with open(tmp_dev_file_name, "w") as f:
   json.dump(dev_data, f)

json_train = load_dataset("json", data_files=tmp_train_file_name)
json_dev = load_dataset("json", data_files=tmp_dev_file_name)

# TRIM LONG INPUT
json_train["train"] = trim_long_input(json_train["train"], CUT_EXTREMITA_INPUT_CHAR_LENGTH)
json_dev["train"] = trim_long_input(json_dev["train"], CUT_EXTREMITA_INPUT_CHAR_LENGTH)


#-------------------
#    LOAD MODEL
#-------------------
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_MODEL)
 
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

tokenizer.padding_side = "left"


# PREPARE DATA
train_data = ( json_train["train"].shuffle().map(generate_and_tokenize_prompt) )
val_data = ( json_dev["train"].shuffle().map(generate_and_tokenize_prompt) )

# PREPARE MODEL
model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_strategy = "steps",
    logging_steps=1,
    optim="adamw_torch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    output_dir=OUTPUT_DIR,
    save_total_limit=1,
    load_best_model_at_end=True,
    label_names=["labels"]
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

model.config.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if torch.__version__ >= "2":
    model = torch.compile(model)

#-------------------
#    TRAIN & SAVE
#-------------------

trainer.train()

model.save_pretrained(OUTPUT_DIR)