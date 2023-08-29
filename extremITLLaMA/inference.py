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

# generate prompts based on task and text
for input in inputs:
    task = input[0]
    text = input[1]
    expected_output = input[2]

    instruction = task_to_prompt(task)
    prompt = generate_prompt_pred(instruction, text[:CUT_EXTREMITA_INPUT_CHAR_LENGTH])

    # tokenization
    tokenized_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # inference
    with torch.no_grad():
        gen_outputs = model.generate(
            **tokenized_inputs,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=MAX_NEW_TOKENS
        )

        # decoding and printing
        for i in range(len(gen_outputs[0])):
            output = tokenizer.decode(gen_outputs[0][i], skip_special_tokens=True)
            if "### Risposta:" in output:
                response = output.split("### Risposta:")[1].rstrip().lstrip()
            else:
                response = "UNK"
            
            print(f"\t {expected_output} \t {response}")



