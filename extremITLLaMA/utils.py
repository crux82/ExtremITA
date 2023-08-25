def target_answer_to_text(target_text: str, task: str):
    if task == "acti_a":
        if target_text == "sì" or target_text == "si":
            return "cospirazione"
        else:
            return "non_cospirazione"
    elif task == "acti_b":
        return target_text
    elif task == "clinkart":
        return target_text
    elif task == "discotex_1":
        if target_text == "sì" or target_text == "si":
            return "coerente"
        else:
            return "non_coerente"
    elif task == "discotex_2":
        return target_text
    elif task == "emit_a":
        return target_text
    elif task == "emit_b":
        return target_text
    elif task == "emotivita":
        return target_text
    elif task == "geolingit":
        return target_text
    elif task == "haspeede":
        if target_text == "sì" or target_text == "si":
            return "odio"
        else:
            return "non_odio"
    elif task == "hodi_a":
        if target_text == "sì" or target_text == "si":
            return "omotransfobico"
        else:
            return "non_omotransfobico"
    elif task == "hodi_b":
        return target_text
    elif task == "langlearn":
        if target_text == "sì" or target_text == "si":
            return "corretto"
        else:
            return "incorretto"
    elif task == "multifakedetective":
        return target_text
    elif task == "nermud":
        return target_text
    elif task == "politicit":
        return target_text
    elif task == "wicita":
        if target_text == "sì" or target_text == "si":
            return "uguale"
        else:
            return "differente"
    else:
        return "task sconosciuto"

def target_text_to_answer(target_text: str, task: str):
    if task == "acti_a":
        if target_text == "cospirazione":
            return "sì"
        else:
            return "no"
    elif task == "acti_b":
        return target_text
    elif task == "clinkart":
        return target_text
    elif task == "discotex_1":
        if target_text == "coerente":
            return "sì"
        else:
            return "no"
    elif task == "discotex_2":
        return target_text
    elif task == "emit_a":
        return target_text
    elif task == "emit_b":
        return target_text
    elif task == "emotivita":
        return target_text
    elif task == "geolingit":
        return target_text
    elif task == "haspeede":
        if target_text == "odio":
            return "sì"
        else:
            return "no"
    elif task == "hodi_a":
        if target_text == "omotransfobico":
            return "sì"
        else:
            return "no"
    elif task == "hodi_b":
        return target_text
    elif task == "langlearn":
        if target_text == "corretto":
            return "sì"
        else:
            return "no"
    elif task == "multifakedetective":
        return target_text
    elif task == "nermud":
        return target_text
    elif task == "politicit":
        return target_text
    elif task == "wicita":
        if target_text == "uguale":
            return "sì"
        else:
            return "no"
    else:
        return "task sconosciuto"

def task_to_prompt(task: str):
    if task == "acti_a":
        return "In questo testo si parla di una cospirazione? Rispondi sì o no."
    elif task == "acti_b":
        return "Di quale teoria cospirazionista parla questo testo, tra \"Covid\", \"Qanon\", \"Terrapiattista\", \"Russia\"?"
    elif task == "clinkart":
        return "Trova i risultati dei test e delle misurazioni nel testo. Per ogni risultato, scrivi \"[BREL]\", seguito dal risultato seguito da \"[SEP]\", seguito dal test, seguito da \"[EREL]\". Se non trovi nessun risultato, scrivi \"[NOREL]\"."
    elif task == "discotex_1":
        return "Le due frasi seguenti, separate da \"[SEP]\", sono coerenti tra loro? Rispondi sì o no."
    elif task == "discotex_2":
        return "Quanto è coerente questa frase, su una scala da 0 a 5?"
    elif task == "emit_a":
        return "Quali emozioni sono espresse in questo testo? Puoi scegliere una o più emozioni tra \"rabbia\", \"anticipazione\", \"disgusto\", \"paura\", \"gioia\", \"amore\", \"tristezza\", \"sorpresa\", \"fiducia\", o \"neutro\"."
    elif task == "emit_b":
        return "Di cosa parla il testo, tra \"direzione\", \"argomento\", \"entrambi\", \"non specificato\"?"
    elif task == "emotivita":
        return "Scrivi quanta valenza è espressa in questo testo su una scala da 1 a 5, seguito da quanto stimolo è espresso in questo testo su una scala da 1 a 5, seguito da quanto controllo è espresso in questo testo su una scala da 1 a 5."
    elif task == "geolingit":
        return "Scrivi la regione di appartenenza di chi ha scritto questo testo, seguito dalla latitudine, seguita dalla longitudine."
    elif task == "haspeede":
        return "In questo testo si esprime odio? Rispondi sì o no."
    elif task == "hodi_a":
        return "In questo testo si esprime odio omotransfobico? Rispondi sì o no."
    elif task == "hodi_b":
        return "Con quali parole l'autore del testo seguente esprime odio omotransfobico? Separa le sequenze di parole con [gap]."
    elif task == "langlearn":
        return "Questi due testi separati da [SEP] sono presentati nell'ordine in cui sono stati scritti? Rispondi sì o no."
    elif task == "multifakedetective":
        return "L'evento riportato nel testo è \"certamente vero\", \"probabilmente vero\", \"probabilmente falso\", o \"certamente falso\"?"
    elif task == "nermud":
        return "Scrivi le menzioni di entità nel testo, indicandone il tipo: [PER] (persona), [LOC] (luogo), [ORG] (organizzazione)."
    elif task == "politicit":
        return "Scrivi se l'autore del testo è \"uomo\" o \"donna\", seguito dalla sua appartenenza politica tra \"destra\", \"sinistra\", \"centrodestra\", \"centrosinistra\"."
    elif task == "wicita":
        return "La parola compresa tra [TGTS] e [TGTE] ha lo stesso significato in entrambe le frasi? Rispondi sì o no."
    else:
        return "task sconosciuto"


 ################ GENERATE METHODS ################
def generate_prompt_pred(instruction, input_):
    return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{instruction}
### Input:
{input_}
### Risposta:"""

def generate_prompt_str(instruction, input_):
    return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{instruction}
### Input:
{input_}
### Risposta:"""

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Di seguito è riportata un'istruzione che descrive un task, insieme ad un input che fornisce un contesto più ampio. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Risposta:
{data_point["output"]}"""
    else:
        return f"""Di seguito è riportata un'istruzione che descrive un task. Scrivete una risposta che completi adeguatamente la richiesta.
### Istruzione:
{data_point["instruction"]}
### Risposta:
{data_point["output"]}"""

    