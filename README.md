
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

To test the model we provide 1 dummy examples for each task:

```python
inputs = [
    ["emit_a", "Caspita che meraviglia #LamicaGeniale", "gioia fiducia"],
    ["emotivita", "Sono pazzo di te", "4.15 4.00 2.53"],
    ["politicit", "Neanche il tempo [POLITICIAN] meravigliarsi dell'ultima assurdità, che [POLITICIAN] #[POLITICAL_PARTY] sforna un'altra proposta inutile e bislacca. L'ultima quella [POLITICIAN] bonus per i matrimoni in chiesa. Propaganda a cui le persone, impegnate con ben altre priorità, non abboccheranno. 30 anni dopo ancora in piazza Tienanmen. @user Ironia, questa sconosciuta. . Si occupi [POLITICIAN] cose più serie, Direttore. . Io mi curerò [POLITICIAN] utilizzare le virgolette [POLITICIAN] prossima volta onde evitarLe l'incombenza [POLITICIAN] commentare quel che scrive una deputata dell'opposizione mentre il governo massacra il Paese. E Lei tace. Saluti. @user Cercherò [POLITICIAN] farla uscire meglio [POLITICIAN] prossima volta. Grazie!", "donna sinistra centrosinistra"],
    ["geolingit", "Daje", "[regione] Lazio [geo] 41.89 12.54"],
    ["langlearn", "Io mi chiamo Silvia e ho dieci anni. Ho un fratello di quattordici anni; se pensate sia bello averne uno più grande vi sbagliate. Vorrei che il mio fratello andasse via, però non so cosa farei senza di lui. Gli voglio bene e so, anche se in realtà non lo so, che anche lui mi vuole “bene”. [SEP] I miti di ieri erano rappresentati da una favola con eroi rimasti ancora oggi famosi, mentre i miti di oggi sono persone che quando finisce la loro carriera vengono dimenticati da quasi tutti. Il mio mito in canzone e spettacolo è Selena Gomez. Anche se ora, tutti la odiano perché si è fidanzata con Justin Biber a me piace comunque e non mi importa di tutto quello che pensano gli altri. Le sue canzoni sono bellissime!", "corretto"],
    ["haspeede", "Fondo assunzioni straordinarie ha una dotazione finanziaria rilevante #leggedibilancio", "non_odio"],
    ["hodi_a", "@user_abcdefghij Speriamo che si caghino sotto", "non_omotransfobico"],
    ["multifakedetective", "#flowers #lovers Fate l'amore non fate la guerra. Marc Chagall, 'Il mazzo di fiori degli amanti', 1926. #art #painting #marcchagall #NoWars https://t.co/XahcLle4SK", "probabilmente vero"],
    ["acti_a", "Flat tax all'ungherese", "non_cospirazione"],
    ["nermud", "Il ministro degli Esteri al commissario capo della Commissione alleata", "[ORG] Commissione alleata"],
    ["clinkart", "Presenza nel siero di anticorpi antimembrana basale glomerulare (anti MBG); negativa la ricerca di anticorpi anti citoplasma dei neutrofili (ANCA).", "[BREL] negativa [SEP] anticorpi [EREL] [BREL] negativa [SEP] ANCA [EREL]"],
    ["wicita", "Non sente ancora il ' peso ' della gravidanza perché l' aumento dell' addome è contenuto e i timori dei primi mesi sono ormai [BT1] superati  [ET1] . [SEP] I provvedimenti di utilizzazione possono essere adottati soltanto nei riguardi di personale che abbia [BT2] superato  [ET2] il periodo di prova.", "uguale"],
    ["discotex_1", "In alcune persone, i sintomi possono continuare per anni. Nella maggior parte dei pazienti, questi sintomi sono seguiti da movimenti involontari e dalla comparsa di un elettroencefalogramma atipico. La maggior parte dei pazienti muore a sei mesi dall'esordio, spesso a causa di infezioni intercorrenti quali polmoniti dovute al deterioramento del riflesso della tosse. [SEP] La prima, allo stato nativo, è solubile in acqua ed è presente nelle cellule sane.", "non_coerente"]
]
```

Run this command for the inference: it will loop through this list, will generate the linguistic description of the task and it will print the output generated by the **ExtremITLLaMA** model.

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
