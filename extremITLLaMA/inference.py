import torch
import argparse

from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM

from utils import *

parser = argparse.ArgumentParser(description='Alpaca Tagger.')
parser.add_argument('-tokenizer','--tokenizer', help='Tokenizer',default='yahma/llama-7b-hf')
parser.add_argument('-base_model','--base_model', help='Base Model',default='sag-uniroma2/extremITA-Camoscio-7b')
parser.add_argument('-adapters_model','--adapters_model', help='Adapters Model',default='sag-uniroma2/extremITA-Camoscio-7b-adapters')
parser.add_argument('-o','--output_file_path', help='Output predicted data file path',default='models/extremITLLaMA/predictions.txt')

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()

output_file_path = args.output_file_path

TOKENIZER_MODEL=args.tokenizer
BASE_MODEL=args.base_model
ADAPTERS_MODEL=args.adapters_model

BS = 1

CUTOFF_LEN = 512
CUT_EXTREMITA_INPUT_CHAR_LENGTH = 1200
MAX_NEW_TOKENS = 256

# ====================================
# LOADING RESOURCES
# ====================================
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_MODEL)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = (0)


if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if ADAPTERS_MODEL != '':
        model = PeftModel.from_pretrained(
            model,
            ADAPTERS_MODEL,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    if ADAPTERS_MODEL != '':
        model = PeftModel.from_pretrained(
            model,
            ADAPTERS_MODEL,
            device_map={"": device},
        )

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = tokenizer.bos_token_id = 1
model.config.eos_token_id = tokenizer.eos_token_id = 2

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

# ====================================
# LET'S DO OUR WORK
# ====================================

# INPUTS examples
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
    ["discotex_1", "In alcune persone, i sintomi possono continuare per anni. Nella maggior parte dei pazienti, questi sintomi sono seguiti da movimenti involontari e dalla comparsa di un elettroencefalogramma atipico. La maggior parte dei pazienti muore a sei mesi dall'esordio, spesso a causa di infezioni intercorrenti quali polmoniti dovute al deterioramento del riflesso della tosse. [SEP] La prima, allo stato nativo, è solubile in acqua ed è presente nelle cellule sane.", "0"]
]

# generate prompts based on task and text
prompts = []
for input in inputs:
    task = input[0]
    text = input[1]
    expected_output = input[2]

    instruction = task_to_prompt(task)
    prompt = generate_prompt_pred(instruction, text[:CUT_EXTREMITA_INPUT_CHAR_LENGTH])
    prompts.append(prompt)

# tokenization
tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

# inference
with torch.no_grad():
    gen_outputs = model.generate(
        **tokenized_inputs,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # decoding and printing
    for i in range(len(gen_outputs)):
        output = tokenizer.decode(gen_outputs[i], skip_special_tokens=True)
        print(output)


