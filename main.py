import logging
from pathlib import Path
from control import control
from evaluation import evaluate
from observability import compute_metrics, visualize_metrics
from utils import read_general_config, read_api_config, read_risk_config, create_results_folder, \
    setup_logging, load_and_prepare_dataset, save_config, init_cache


if __name__ == "__main__":
    config_file = Path("config.toml")
    mode, evaluation, languages, dictionary_offensive_words = read_general_config(config_file)
    if mode == "use_api" or mode == "use_api_diccionary":
        url, requested_attributes, evaluation_attributes = read_api_config(config_file)
    else:
        url = None
        requested_attributes = None
        evaluation_attributes = None

    agent_answers, dataset_name, y_true = load_and_prepare_dataset(config_file)
    results_dir = create_results_folder(dataset_name)
    save_config(config_file, results_dir)
    risk_threshold, max_risk_rate = read_risk_config(config_file)
    setup_logging(results_dir)
    if agent_answers is None:
        logging.warning("No se han podido cargar las respuestas del agente.")
        exit(0)

    cache_db = init_cache(results_dir / "api_cache.db")
    metrics = compute_metrics(mode, languages, url, requested_attributes, evaluation_attributes, agent_answers,
                              dictionary_offensive_words, risk_threshold, cache_db)

    # Observabilidad
    visualize_metrics(metrics, agent_answers, results_dir)

    # Control
    control(metrics, max_risk_rate, results_dir)

    #Evaluation
    if evaluation and y_true is not None:
        evaluate(metrics, y_true, results_dir)