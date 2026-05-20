from pathlib import Path

from control import control
from observability import compute_metrics, visualize_metrics
from utils import read_general_config, read_api_config, read_risk_config

if __name__ == "__main__":
    config_file = Path("config.toml")
    mode, languages, agent_answers, dictionary_offensive_words = read_general_config(config_file)
    if mode == "use_api" or mode == "use_api_diccionary":
        url, attributes = read_api_config(config_file)
    else:
        url = None
        attributes = None
    risk_threshold, max_risk_rate = read_risk_config(config_file)
    metrics = compute_metrics(mode, languages, url, attributes, agent_answers, dictionary_offensive_words, risk_threshold)

    # Observabilidad
    visualize_metrics(metrics, agent_answers)

    # Control
    control(metrics, max_risk_rate)