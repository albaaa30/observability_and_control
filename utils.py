import logging
import os
from pathlib import Path
import toml
from datetime import datetime
from datasets import load_dataset
import shutil
import json
import sqlite3
from langdetect import detect


def read_general_config(config_file):
    if not config_file.is_file():
        raise FileNotFoundError(f"El archivo de configuración {config_file} no existe.")
    config = toml.load(config_file)
    mode = config['general'].get('mode', 'use_api_diccionary')
    dictionary_offensive_words = None
    if mode in ('use_api_diccionary', 'use_diccionary'):
        offensive_dict_path = Path(config['general']['offensive_dict_file'])
        try:
            with open(offensive_dict_path, 'r', encoding='utf-8') as f:
                dictionary_offensive_words = set([line.strip().lower() for line in f if line.strip()])
        except FileNotFoundError:
            dictionary_offensive_words = set()
            logging.warning(f"Diccionario de palabras ofensivas no encontrado: {offensive_dict_path}")

    languages = config['general'].get('languages', ['en'])
    if languages in ('en', 'English'):
        languages = ['en']
    evaluate = config['general'].get('evaluate', False)

    return mode, evaluate, languages, dictionary_offensive_words


def read_api_config(config_file):
    if not config_file.is_file():
        raise FileNotFoundError(f"El archivo de configuración {config_file} no existe.")
    config = toml.load(config_file)
    api_key = config['api'].get('api_key')
    if api_key is None:
        raise ValueError("Es necesario tener una API KEY para utilizar los modos use_api y use_api_diccionary.")
    attributes_list = config['api'].get('requested_attributes', ['TOXICITY', 'INSULT', 'IDENTITY_ATTACK', 'PROFANITY'])
    requested_attributes = {attr: {} for attr in attributes_list}
    attributes_list = config['api'].get('evaluation_attributes', ['TOXICITY', 'INSULT', 'IDENTITY_ATTACK', 'PROFANITY'])
    evaluation_attributes = {attr: {} for attr in attributes_list}

    url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"

    return url, requested_attributes, evaluation_attributes


def read_risk_config(config_file):
    if not config_file.is_file():
        raise FileNotFoundError(f"El archivo de configuración {config_file} no existe.")
    config = toml.load(config_file)
    risk_threshold = config['risk'].get('risk_threshold', 0.4)
    max_risk_rate = config['risk'].get('max_risk_rate', 0.2)

    return risk_threshold, max_risk_rate


def load_and_prepare_dataset(config_file):
    if not config_file.is_file():
        raise FileNotFoundError(f"El archivo de configuración {config_file} no existe.")
    config = toml.load(config_file)
    dataset_name = config['dataset'].get('dataset')
    filter_language = config['dataset'].get('filter_language', False)
    languages = config['general'].get('languages', 'en')
    labels = None

    if dataset_name is None:
        agent_answers_path = Path(config['general']['agent_answers_file'])
        if not agent_answers_path.is_file():
            raise FileNotFoundError(f"El archivo de respuestas no existe: {agent_answers_path}. "
                                    f"Es necesario escoger un dataset (chatbot_arena, toxic_chat0124 o toxic_chat112) o "
                                    f"aportar un archivo.")
        with open(agent_answers_path, 'r', encoding='utf-8') as f:
            agent_answers = [line.strip() for line in f if line.strip()]

    elif dataset_name == 'chatbot_arena':
        ds = load_dataset("lmsys/chatbot_arena_conversations")
        df = clean_dataset(ds, filter_language, languages)
        df = clean_dataframe(df, "response")
        agent_answers = df["response"].tolist()

    elif dataset_name in ('toxic_chat0124','toxic_chat1123'):
        subset = dataset_name.replace("_", "")
        ds = load_dataset("lmsys/toxic-chat",subset)
        df = ds["train"].to_pandas()
        if filter_language and languages in ('en', 'English', ['en']):
            df["detected_language"] = df["user_input"].apply(detect_language)
            df = df[df["detected_language"] == "en"]
        labels = df["toxicity"].astype(int).tolist()
        evaluate = config['general'].get('evaluate', False)
        if evaluate:
            df = clean_dataframe(df, "user_input")
            agent_answers = df["user_input"].tolist()
            labels = df["toxicity"].astype(int).tolist()
        else:
            df = clean_dataframe(df, "model_output")
            agent_answers = df["model_output"].tolist()

    else:
        raise ValueError(f"Dataset no implementado: {dataset_name}")

    return agent_answers, dataset_name, labels


def detect_language(text):
    if not isinstance(text, str):
        return "unknown"
    text = text.strip()
    if not text:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def clean_dataframe(df, text_column):
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip() != ""]
    return df


def clean_dataset(ds, filter_language, languages):
    df = ds["train"].select_columns([
            "question_id",
            "winner",
            "language",
            "turn",
            "conversation_a",
            "conversation_b",
            "openai_moderation",
            "toxic_chat_tag"
        ]).to_pandas()
    if filter_language and languages in ('en', 'English', ['en']):
        df = df[df["language"] == "English"]
    df = df[df["winner"] != "tie"]
    df = df.reset_index(drop=True)
    df["response"] = df.apply(get_winner_response, axis=1)
    df = df[df["response"].notna()]
    df = df[df["response"].str.strip() != ""]
    return df


def create_results_folder(dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = (Path("results")/dataset_name/timestamp)
    results_dir.mkdir(parents=True,exist_ok=True)

    return results_dir


def get_winner_response(row):
    if row["winner"] == "model_a":
        conversation = row["conversation_a"]
    elif row["winner"] == "model_b":
        conversation = row["conversation_b"]
    else:
        return None
    assistant_messages = [msg["content"] for msg in conversation if msg["role"] == "assistant"]
    if len(assistant_messages) == 0:
        return None

    return assistant_messages[-1]


def setup_logging(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers: #evitar duplicados
        file_handler = logging.FileHandler(
            os.path.join(results_dir, "app.log"),
            mode="a",
            encoding="utf-8"
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def save_config(config_file, results_dir):
    shutil.copy(config_file, results_dir/"config.toml")


def init_cache(db_path):
    cache_db = sqlite3.connect(db_path)
    cursor = cache_db.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS api_cache (text TEXT PRIMARY KEY, response TEXT)""")
    cache_db.commit()

    return cache_db


def get_cached_response(cache_db, text):
    cursor = cache_db.cursor()
    cursor.execute("SELECT response FROM api_cache WHERE text = ?", (text,))
    row = cursor.fetchone()
    if row is None:
        return None

    return json.loads(row[0])


def save_cached_response(cache_db, text, response):
    cursor = cache_db.cursor()
    cursor.execute("INSERT OR REPLACE INTO api_cache(text, response) VALUES (?, ?)", (text, json.dumps(response)))
    cache_db.commit()
