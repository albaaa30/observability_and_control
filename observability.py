import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from spellchecker import SpellChecker
import re
import requests
import logging
import csv
import time
from utils import get_cached_response, save_cached_response

last_call = 0


def use_api(text, api_url, attributes, languages, cache_db, interval=1.0, max_retries=5):
    global last_call
    cached = get_cached_response(cache_db, text)

    if cached is not None:
        return cached

    data = {
        "comment": {"text": text},
        "languages": languages,
        "requestedAttributes": attributes
    }

    for attempt in range(max_retries):
        elapsed = time.time() - last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        try:
            resp = requests.post(api_url, json=data, timeout=30)
            if resp.status_code == 200:
                last_call = time.time()
                result = resp.json()
                attr_scores = result.get("attributeScores", {})
                scores = {
                    attr: attr_scores.get(attr, {})
                    .get("summaryScore", {})
                    .get("value", 0.0)
                    for attr in attributes.keys()
                }
                save_cached_response(cache_db, text, scores)
                return scores
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = int(retry_after)
                else:
                    wait = 2 ** attempt
                logging.warning(f"429 rate limit → retry in {wait}s")
                time.sleep(wait)
            else:
                logging.warning(f"Error {resp.status_code}: {resp.text}")
                return {}

        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"Exception: {e} → re  try in {wait}s")
            time.sleep(wait)

    logging.error(f"Fallo tras {max_retries} intentos.")
    return {}


def compute_metrics(mode, languages, url, attributes, evaluation_attributes, agent_answers, dictionary_offensive_words,
                    risk_threshold, cache_db):
    spell = SpellChecker(language=languages[0])
    metrics = []
    print(f"Comenzando el procesamiento de {len(agent_answers)} respuestas.")
    logging.info(f"Comenzando el procesamiento de {len(agent_answers)} respuestas.")
    for i, r in enumerate(agent_answers, start=1):
        if i % 100 == 0:
            percentage = 100 * i / len(agent_answers)
            print(f"\rProcesadas {i}/{len(agent_answers)} ({percentage:.1f}%)",end="",flush=True)
            logging.info(f"{i}/{len(agent_answers)} {percentage:.1f}% respuestas procesadas.")
        palabras = re.findall(r"\w+", r.lower())
        total_palabras = len(palabras)
        palabras_unicas = len(set(palabras))
        diversidad = palabras_unicas / total_palabras if total_palabras > 0 else 0

        frases = re.split(r"[.¡!¿?,;:…\-\—\n]", r)
        frases = [f.strip() for f in frases if f.strip()]
        legibilidad = total_palabras / len(frases) if len(frases) > 0 else 0

        errores = spell.unknown(palabras)
        tasa_errores = len(errores) / total_palabras if total_palabras > 0 else 0

        ofensivas_detectadas = []
        scores_api = {}
        riesgo_flag = False

        if mode == "use_api":
            scores_api = use_api(r, url, attributes, languages, cache_db)
            riesgo_flag = any(scores_api.get(attr, 0) >= risk_threshold for attr in evaluation_attributes)

        elif mode == "use_diccionary":
            ofensivas_detectadas = [p for p in palabras if p in dictionary_offensive_words]
            riesgo_flag = bool(ofensivas_detectadas)

        elif mode == "use_api_diccionary":
            scores_api = use_api(r, url, attributes, languages, cache_db)
            ofensivas_detectadas = [p for p in palabras if p in dictionary_offensive_words]
            riesgo_flag = bool(ofensivas_detectadas) or any(scores_api.get(attr, 0) >= risk_threshold for attr in evaluation_attributes)

        if scores_api:
            score_riesgo = max(scores_api.get(attr, 0) for attr in evaluation_attributes)
        elif ofensivas_detectadas:
            score_riesgo = 1
        else:
            score_riesgo = 0

        metrics.append({
            "respuesta": r,
            "longitud": total_palabras,
            "diversidad": diversidad,
            "legibilidad": legibilidad,
            "tasa_errores": tasa_errores,
            "ofensivas": ofensivas_detectadas,
            "perspective": scores_api,
            "riesgo": riesgo_flag,
            "score_riesgo": score_riesgo
        })

    print()
    logging.info(f"Procesamiento finalizado. {len(metrics)} respuestas analizadas.")

    return metrics


def visualize_metrics(metrics, agent_answers, results_dir):
    longitudes = [m["longitud"] for m in metrics]
    diversidades = [m["diversidad"] for m in metrics]
    legibilidades = [m["legibilidad"] for m in metrics]
    tasa_errores = [m["tasa_errores"] for m in metrics]

    plt.figure(figsize=(15, 10))

    # Serie temporal de longitudes
    plt.subplot(2, 3, 1)
    plt.plot(longitudes, marker='o')
    plt.title("Longitud de respuestas (palabras)")
    plt.xlabel("Nº de respuesta")
    plt.ylabel("Nº de palabras")

    # Histograma de diversidad
    plt.subplot(2, 3, 2)
    plt.hist(diversidades, bins=5, edgecolor='black', color='green')
    plt.title("Distribución de diversidad léxica")
    plt.xlabel("Diversidad")
    plt.ylabel("Frecuencia")

    # Evolución de legibilidad
    plt.subplot(2, 3, 3)
    plt.plot(legibilidades, marker='s', color='red')
    plt.title("Evolución de legibilidad (palabras/frase)")
    plt.xlabel("Nº de respuesta")
    plt.ylabel("Palabras promedio por frase")

    # Evolución de la tasa de errores
    plt.subplot(2, 3, 4)
    plt.plot(tasa_errores, marker='s', color='orange')
    plt.title("Tasa de errores")
    plt.xlabel("Nº de respuesta")
    plt.ylabel("Tasa de errores")

    # Nube de palabras
    texto_total_respuestas = " ".join(agent_answers)
    wordcloud = WordCloud(width=500, height=300, background_color="white", colormap="plasma").generate(texto_total_respuestas)
    plt.subplot(2, 3, 5)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nube de palabras")

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "metrics_plot.png")
    plt.savefig(plot_path)
    plt.show()

    csv_file = os.path.join(results_dir, "metrics_output.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    logging.info(f"Métricas guardadas en {csv_file}")
