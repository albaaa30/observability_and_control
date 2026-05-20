import logging
from pathlib import Path
import toml


def read_general_config(config_file):
    if not config_file.is_file():
        raise FileNotFoundError(f"El archivo de configuración {config_file} no existe.")
    config = toml.load(config_file)
    agent_answers_path = Path(config['general']['agent_answers_file'])
    offensive_dict_path = Path(config['general']['offensive_dict_file'])
    if not agent_answers_path.is_file():
        raise FileNotFoundError(f"El archivo de respuestas no existe: {agent_answers_path}")

    with open(agent_answers_path, 'r', encoding='utf-8') as f:
        agent_answers = [line.strip() for line in f if line.strip()]

    try:
        with open(offensive_dict_path, 'r', encoding='utf-8') as f:
            dictionary_offensive_words = set([line.strip().lower() for line in f if line.strip()])
    except FileNotFoundError:
        dictionary_offensive_words = set()
        logging.warning(f"Diccionario de palabras ofensivas no encontrado: {offensive_dict_path}")

    mode = config['general'].get('mode', 'use_api_diccionary')
    languages = config['general'].get('languages', ['es'])

    return mode, languages, agent_answers, dictionary_offensive_words

def read_api_config(config_file):
    if not config_file.is_file():
        raise FileNotFoundError(f"El archivo de configuración {config_file} no existe.")
    config = toml.load(config_file)
    api_key = config['api'].get('api_key')
    if api_key is None:
        raise ValueError("Es necesario tener una API KEY para utilizar los modos use_api y use_api_diccionary.")
    attributes_list = config['api'].get('attributes', ['TOXICITY', 'INSULT', 'IDENTITY_ATTACK', 'PROFANITY'])
    attributes = {attr: {} for attr in attributes_list}

    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"

    return url, attributes


def read_risk_config(config_file):
    if not config_file.is_file():
        raise FileNotFoundError(f"El archivo de configuración {config_file} no existe.")
    config = toml.load(config_file)
    risk_threshold = config['risk'].get('risk_threshold', 0.8)
    max_risk_rate = config['risk'].get('max_risk_rate', 0.2)

    return risk_threshold, max_risk_rate
