import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score)


def evaluate(metrics, y_true, results_dir):
    y_pred = [int(m["riesgo"]) for m in metrics]

    if len(y_true) != len(y_pred):
        raise ValueError(f"Longitudes distintas: y_true={len(y_true)}, y_pred={len(y_pred)}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, target_names=["safe", "toxic"], zero_division=0)

    print("\nEVALUACIÓN:")
    print(report)

    with open(os.path.join(results_dir, "evaluation.txt"), "w", encoding="utf-8") as f:
        f.write("===== EVALUACIÓN =====\n\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["safe", "toxic"])
    disp.plot(ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    save_predictions(metrics, y_true, results_dir)
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    scores = [m["score_riesgo"] for m in metrics]
    roc_auc = save_roc_curve(y_true, scores, results_dir)
    pr_auc = save_precision_recall_curve(y_true, scores, results_dir)
    best = find_best_threshold(y_true, scores)
    save_summary(accuracy,precision,recall,f1,tn,fp,fn,tp,roc_auc,pr_auc,best,results_dir)
    save_errors(metrics, y_true, results_dir)


def save_predictions(metrics, y_true, results_dir):
    rows = []
    for i, m in enumerate(metrics):
        row = {
            "id": i,
            "y_pred": int(m["riesgo"]),
            "score_riesgo": m["score_riesgo"]
        }
        if y_true is not None:
            row["y_true"] = y_true[i]
        perspective = m.get("perspective", {})
        if perspective:
            for k, v in perspective.items():
                row[k] = v
        ofensivas = m.get("ofensivas", [])
        row["num_offensive_words"] = len(ofensivas)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "predictions.csv", index=False)


def save_summary(accuracy,precision,recall,f1,tn,fp,fn,tp,roc_auc,pr_auc,best,results_dir):
    best_threshold = best["threshold"]
    best_precision = best["precision"]
    best_recall = best["recall"]
    best_f1 = best["f1"]

    df = pd.DataFrame([{
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_threshold": best_threshold,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "best_f1": best_f1
    }])
    df.to_csv(results_dir / "summary.csv", index=False)


def save_errors(metrics, y_true, results_dir):
    rows_fp = []
    rows_fn = []
    for i, (m, yt) in enumerate(zip(metrics, y_true)):
        yp = int(m["riesgo"])
        row = {
            "id": i,
            "y_true": yt,
            "y_pred": yp,
            "score_riesgo": m["score_riesgo"],
            "respuesta": m["respuesta"]
        }
        if yt == 0 and yp == 1:
            rows_fp.append(row)
        elif yt == 1 and yp == 0:
            rows_fn.append(row)
    pd.DataFrame(rows_fp).to_csv(results_dir / "false_positives.csv", index=False)
    pd.DataFrame(rows_fn).to_csv(results_dir / "false_negatives.csv", index=False)


def save_roc_curve(y_true, scores, results_dir):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve.png")
    plt.close()

    return roc_auc


def save_precision_recall_curve(y_true, scores, results_dir):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "pr_curve.png")
    plt.close()

    return ap


def evaluate_thresholds(metrics, y_true, results_dir):
    scores = [m.get("score_riesgo", int(m["riesgo"])) for m in metrics]
    rows = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        y_pred = [int(score >= threshold) for score in scores]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "threshold_metrics.csv", index=False)

    best = df.loc[df["f1"].idxmax()]
    print("\nMEJOR UMBRAL:")
    print(
        f"threshold={best['threshold']:.2f} "
        f"precision={best['precision']:.3f} "
        f"recall={best['recall']:.3f} "
        f"f1={best['f1']:.3f}"
    )

    return df


def save_threshold_f1_plot(df, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["threshold"], df["f1"], marker="o")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score vs risk threshold")
    plt.tight_layout()
    fig.savefig(results_dir / "threshold_f1.png")
    plt.close()


def save_threshold_precision_recal_plot(df, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    ax.plot(df["threshold"], df["recall"], marker="o", label="Recall")
    ax.legend()
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall vs threshold")
    plt.tight_layout()
    fig.savefig(results_dir / "threshold_precision_recall.png")
    plt.close()


def find_best_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    idx = np.argmax(f1)

    return {"threshold": thresholds[idx], "precision": precision[idx], "recall": recall[idx], "f1": f1[idx]}
