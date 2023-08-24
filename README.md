
# ExtremITA at EVALITA 2023: Multi-Task Sustainable Scaling to Large Language Models at its Extreme ![logo](./docs/logo.png)

This repository contains the code to reproduce the ExtremITA architectures that participated in all of the [EVALITA 2023](https://www.evalita.it/campaigns/evalita-2023/) challenges. We evaluated an independent monolithic model: **extremITLLaMA** an instruction-tuned Decoder-only Large Language Model, specifically designed for handling Italian instructions. Our approach revolves around representing tasks in natural language, where we provide instructions to the model using prompts that define the expected responses.

Remarkably, **extremITLLaMA** achieved first place in **41%** of the subtasks (9 out of a total of 22) and showcased top-three performance in **64%** (14 out of 22). These subtasks encompass various semantic dimensions, including Affect Detection, Authorship Analysis, Computational Ethics, Named Entity Recognition, Information Extraction, and Discourse Coherence.


# Set up the environment

Create the environment and install the requirements:

```
# create conda env
conda create env -n extremITA python=3.9.10 -y
conda activate extremITA
# install general dependencies
pip install -r requirements.txt
# install LLaMA requirements
cd extremITLLaMA
pip install -r requirements.txt
```

# How to generate the dataset

Data directory is empty as we cannot share the EVALITA data for fine-tuning. You should access each individual site from the [task page](https://www.evalita.it/campaigns/evalita-2023/tasks/) of EVALITA and request to download the data. After downloading it, put them in the associated folder: e.g., you should put the "ACTI" data in `/data/ACTI` divided by subtask.  
Once you collected all the data you can encode them into the dataset format for our models:

```
python encode.py
```

This command will generate a file for each task in the `out` directory. In order to fine-tune our models you should merge them in one single file and split into `train.txt` and `dev.txt`. In our experimentations we split with ratio 95/5.  
These files are made of 4 columns without any header:
- id
- task name, from which the natural language task description is generated
- input text
- expected output


# How to train extremITLLaMA

Be sure to have the 2 aforementioned files `train.txt` and `dev.txt` in the data folder, then run the following command:

```
nohup python -u extremITLLaMA/train.py > logs/training_extremITLLaMA.out &
```

By default the script will train the extremITLLaMA for 2 epochs on the whole dataset you provided. For more details please consult the official paper. In the end, the model will be saved in the `models` directory.


# Inference

To test the model we provide 2 dummy invented examples in `/data/test.txt` for the inference. Run this command, it will create a file `/data/predictions.txt` with the resulting predictions in the same 4 columns format and will print some information during inference.

```
nohup python -u extremITLLaMA/inference.py > logs/inference.out &
```


# Citation
To appear in:
```
@inproceedings{hromei2023extremita,
  author       = {Claudiu Daniel Hromei and
                  Danilo Croce and
                  Valerio Basile and
                  Roberto Basili},
  title        = {ExtremITA at EVALITA 2023: Multi-Task Sustainable Scaling to Large Language Models at its Extreme},
  booktitle    = {Proceedings of the Eighth Evaluation Campaign of Natural Language
                  Processing and Speech Tools for Italian. Final Workshop (EVALITA 2023)},
  publisher    = {CEUR.org},
  year         = {2023},
  month        = {September},
  address      = {Parma, Italy}
}
```
