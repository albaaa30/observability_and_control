import logging
import os


def control(metrics, max_risk_rate, results_dir):
    respuestas_totales = len(metrics)
    riesgos = [m for m in metrics if m["riesgo"]]
    tasa_riesgo = len(riesgos) / respuestas_totales if respuestas_totales > 0 else 0

    logging.info(f"Respuestas totales: {respuestas_totales}")
    logging.info(f"Respuestas de riesgo: {len(riesgos)} ({tasa_riesgo:.1%})")

    if tasa_riesgo > max_risk_rate:
        logging.warning("ALERTA: El agente está generando demasiadas respuestas de riesgo. Se procede a BLOQUEAR el sistema.")
        return False

    log_file_risk = os.path.join(results_dir, "risk_log.txt")
    log_file_safe = os.path.join(results_dir, "safe_log.txt")
    open(log_file_risk, "w").close()
    open(log_file_safe, "w").close()
    for i, m in enumerate(metrics, 1):
        logging.info(f"Respuesta {i}: {m['respuesta']}")
        if m["riesgo"]:
            texto_riesgo = []
            if m["ofensivas"]:
                texto_riesgo.append(f"Palabras ofensivas detectadas con diccionario: {m['ofensivas']}")
            if m["perspective"]:
                texto_riesgo.append(f"Riesgo detectado por la API Perspective. Scores: {m['perspective']}")

            mensaje = " | ".join(texto_riesgo)
            logging.warning(mensaje)

            with open(log_file_risk, "a", encoding="utf-8") as f:
                f.write(f"Respuesta {i}: {m['respuesta']} -> {mensaje}\n")
        else:
            with open(log_file_safe, "a", encoding="utf-8") as f:
                f.write(f"Respuesta {i}: {m['respuesta']} -> Sin riesgo\n")
    return True
