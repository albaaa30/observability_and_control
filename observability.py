import matplotlib.pyplot as plt
from wordcloud import WordCloud
from spellchecker import SpellChecker
import re
import requests
import logging
from pathlib import Path
import csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def use_api(text, api_url, attributes, languages):
    data = {
        "comment": {"text": text},
        "languages": languages,
        "requestedAttributes": attributes
    }
    try:
        resp = requests.post(api_url, json=data)
        resp.raise_for_status()
        result = resp.json()
        attr_scores = result.get("attributeScores", {})
        scores = {attr: attr_scores.get(attr, {}).get("summaryScore", {}).get("value", 0.0)
                  for attr in attributes.keys()}
        return scores
    except Exception as e:
        logging.warning(f"Error al consultar API: {e}")
        return {a: 0.0 for a in attributes.keys()}

def compute_metrics(mode, languages, url, attributes, agent_answers, dictionary_offensive_words, risk_threshold):
    spell = SpellChecker(language=languages[0])
    metrics = []

    for r in agent_answers: #answer
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
            scores_api = use_api(r, url, attributes, languages)
            riesgo_flag = any(score >= risk_threshold for score in scores_api.values())

        elif mode == "use_diccionary":
            ofensivas_detectadas = [p for p in palabras if p in dictionary_offensive_words]
            riesgo_flag = bool(ofensivas_detectadas)

        elif mode == "use_api_diccionary":
            scores_api = use_api(r, url, attributes, languages)
            ofensivas_detectadas = [p for p in palabras if p in dictionary_offensive_words]
            riesgo_flag = bool(ofensivas_detectadas) or any(score >= risk_threshold for score in scores_api.values())

        metrics.append({
            "respuesta": r,
            "longitud": total_palabras,
            "diversidad": diversidad,
            "legibilidad": legibilidad,
            "tasa_errores": tasa_errores,
            "ofensivas": ofensivas_detectadas,
            "perspective": scores_api,
            "riesgo": riesgo_flag
        })

    return metrics

def visualize_metrics(metrics, agent_answers):
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
    plt.show()

    csv_file = Path("metrics_output.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    logging.info(f"Métricas guardadas en {csv_file}")
