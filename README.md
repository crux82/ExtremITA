
# Introduction

This repository contains the code to reproduce the ExtremITA architectures that participated in all of the [EVALITA 2023](https://www.evalita.it/campaigns/evalita-2023/) challenges. We evaluated two independent monolithic models: \texttt{extremIT5}, an Encoder-Decoder model, and \texttt{extremITLLaMA} an instruction-tuned Decoder-only Large Language Model, specifically designed for handling Italian instructions. Our approach revolves around representing tasks in natural language, where we provide instructions to the model using prompts that define the expected responses.

Remarkably, our best-performing model (\texttt{extremITLLaMA}) achieved first place in 41% of the subtasks (9 out of a total of 22) and showcased top-three performance in 64% (14 out of 22). These subtasks encompass various semantic dimensions, including Affect Detection, Authorship Analysis, Computational Ethics, Named Entity Recognition, Information Extraction, and Discourse Coherence.


# Set up environment

Create the environment and install the requirements:

```
# create conda env
conda create env -n extremITA python=3.9.10 -y
conda activate extremITA
# install IT5 and LLaMA requirements
cd model_scripts
pip install -r requirements.txt
# install other dependencies
cd ../
pip install -r requirements.txt
```

# How to generate the dataset

Data directory is empty as we cannot share the EVALITA data for fine-tuning. You should access each individual site from the [task page](https://www.evalita.it/campaigns/evalita-2023/tasks/) of EVALITA and request to download the data. After downloading it, put them in the associated folder: e.g., you should put the "ACTI" data in `/data/acti` divided by subtask. Once you collected all the data you can encode them into the dataset format for our models:

```
python encode.py
```

This command will generate a file for each task in the `out` directory. In order to fine-tune our models you should merge them in one single file and split into `train.txt` and `dev.txt`. In our experimentations we split with ratio 95/5.


# How to train the ExtremITA architectures

How to train


# Citation
TODO

