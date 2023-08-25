
# ExtremITA at EVALITA 2023: Multi-Task Sustainable Scaling to Large Language Models at its Extreme ![logo](./docs/logo.png)

This repository contains the code to reproduce the ExtremITA architectures that participated in all of the [EVALITA 2023](https://www.evalita.it/campaigns/evalita-2023/) challenges. We evaluated an independent monolithic model: **extremITLLaMA** an instruction-tuned Decoder-only Large Language Model, specifically designed for handling Italian instructions. Our approach revolves around representing tasks in natural language, where we provide instructions to the model using prompts that define the expected responses.

Remarkably, **extremITLLaMA** achieved first place in **41%** of the subtasks (9 out of a total of 22) and showcased top-three performance in **64%** (14 out of 22). These subtasks encompass various semantic dimensions, including Affect Detection, Authorship Analysis, Computational Ethics, Named Entity Recognition, Information Extraction, and Discourse Coherence.


# Set up the environment

Create the environment and install the requirements:

```
# create conda env
conda create -n extremITA python=3.9 -y
conda activate extremITA
# install general dependencies
pip install -r requirements.txt
# install LLaMA requirements
cd extremITLLaMA
pip install -r requirements.txt
# install cudatoolkit
conda install cudatoolkit
```

# How to generate the dataset

The data directory is empty as we cannot share the EVALITA data for fine-tuning. You should access each individual site from the [task page](https://www.evalita.it/campaigns/evalita-2023/tasks/) of EVALITA and request to download the data. After downloading it, put them in the associated folder: e.g., you should put the "ACTI" data in `/data/ACTI` divided by subtask.  
Once you have collected all the data you can encode them into the dataset format for our model:

```
python encode.py
```

This command will generate a file for each task in the `out` directory. In order to fine-tune our models you should merge them into one single file and split them into `train.txt` and `dev.txt`. In our experimentations, we split with a ratio of 95/5.  
These files are made of 4 columns (with a tab character as a delimiter) without any header:
- id
- task name, from which the natural language task description is generated
- input text
- expected output


# How to train extremITLLaMA

Be sure to have the 2 files above `train.txt` and `dev.txt` in the data folder, then go back to the root folder and run the following command:

```
nohup python -u extremITLLaMA/train.py > training_extremITLLaMA.out &
```

By default, the script will train the extremITLLaMA for 2 epochs on the dataset you provided. For more details please consult the official paper. In the end, the model will be saved in the `models` directory.


# Inference

To test the model we provide 1 dummy example (`task_name`, `input_text`, `expected_output`) for each task:

```python
inputs = [
    ["emit_a", "Ora siamo tutti sollevati #IMedici", "fiducia"],
    ["emotivita", "Non pretendo che la nostra fosse stata una relazione perfetta, solo meravigliosa.", "4.0 4.0 3.3"],
    ["politicit", "[POLITICIAN] Riforma [POLITICIAN] catasto è più tasse sulla casa per tutti. Evitiamo gli alibi delle case “fantasma” da accatastare e [POLITICIAN] quelle in centro a valore [POLITICIAN] periferia perché si possono sistemare già con [POLITICIAN] normativa ATTUALE! @politician @politician @user @politician @politician State penalizzando un settore che è trainante [POLITICIAN] PIL ed è sempre cresciuto dal dopoguerra ad oggi! stai sereno informati almeno sui lavori parlamentari: è il Governo che ha minacciato [POLITICIAN] dimettersi se non si approvava [POLITICIAN] riforma [POLITICIAN] catasto, giusto perché è solo una mappatura! Solo chi non ha Onesta intellettuale non vede MAGGIORI TASSE PER TUTTI! cioè stiamo penalizzando l’unico settore che dal dopoguerra ad oggi ha visto non solo una crescita continua ma essere trainante [POLITICIAN] PIL italiano…brillanti davvero brillanti!", "uomo destra destra"],
    ["geolingit", "[USER] [USER] Si amico mio. Te accolli. Altro che sette. Saranno minimo na ventina a esse boni.", "[regione] Lazio [geo] 41.89 12.54"],
    ["langlearn", "Mi chiamo Francesco ho dieci anni e frequento la scuola Sant’Agata e sto in 1 a.Sono alto circa centoquaranta centimetri. Ho capelli neri, orecchie un po a sventola ciglie lunghe, sopracciglia corte, viso ovale, un po ciccione e dei occhi neri. Quando mi arrabbio lancio tutto per aria e se proprio mi arrabbio faccio la mia casa “sottosopra” cosa che accade raramente. Io di solito mi vesto con i jeans e una camicia. [SEP] Ciao io sono Francesca vi devo raccontare un fatto strano, ora ve lo racconto. Ero a Anzio con i miei amici naturalmente c’era anche l’amico che odiavo perché a mia mamma era simpatico. Un giorno eravamo andati al mare e stavamo nuotando, a un certo punto viene un onda gigante anzi tre, perché lontano stava passando una crociera che stava andando a Ponza.", "sì"],
    ["haspeede", "Fondo assunzioni straordinarie ha una dotazione finanziaria rilevante #leggedibilancio", "no"],
    ["hodi_a", "@user_abcdefghij Speriamo che si caghino sotto", "no"],
    ["multifakedetective", "#flowers #lovers Fate l'amore non fate la guerra. Marc Chagall, 'Il mazzo di fiori degli amanti', 1926. #art #painting #marcchagall #NoWars https://t.co/XahcLle4SK", "probabilmente vero"],
    ["acti_a", "Flat tax all'ungherese", "no"],
    ["nermud", "infatti impegna per il futuro , essendo il suo vero proposito quello di stabilire chi delle persone ora lì presenti appartiene alla Venezia Giulia e chi no.", "[LOC] Venezia Giulia"],
    ["clinkart", "Veniva documentato, inoltre, il rialzo della troponina TnT-hs (289;", "[BREL] 289 [SEP] troponina [EREL]"],
    ["wicita", "La [BT1] faccia  [ET1] dura verso gli abusi insanabili , che questo « lifting » al condono si picca di mostrare , ha infatti dei precedenti tali da far nascere qualche diffidenza . [SEP] Si potrebbe leggere tanto dinamismo , volontà ed attivismo in questo film , ma ci sono le [BT2] facce  [ET2] , queste non mentono , anzi smascherano ogni ipocrisia .", "sì"],
    ["discotex_1", "Potete aggiungere solo un gene? No, potete aggiungere addirittura intere vie metaboliche. Tutto questo è possibile grazie a due brillantissime genetiste. [SEP] Vorrei rassicurarvi, la maggior parte degli scienziati non vuole fare esseri umani geneticamente modificati.", "no"]
]
```

Go back to the root folder and run this command for the inference: it will loop through this list (already defined in the file), generate the linguistic description of the task and it will print the output generated by the **ExtremITLLaMA** model.

```
nohup python -u extremITLLaMA/inference.py > inference.out &
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
