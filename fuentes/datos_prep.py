"""Actividad 5: reproducible dataset acquisition, quality checks and EDA.
Usage: python fuentes/datos_prep.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer

BASE = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data():
    dataset = load_breast_cancer(as_frame=True)
    original = dataset.frame.copy().rename(columns={"target": "target_original"})
    original_missing = int(original.isna().sum().sum())
    original_duplicates = int(original.duplicated().sum())

    cleaned = original.drop_duplicates().copy()
    cleaned.columns = [c.strip().lower().replace(" ", "_") for c in cleaned.columns]
    # scikit-learn: 0=malignant, 1=benign. Positive label 1=malignant.
    cleaned["target_maligno"] = (cleaned.pop("target_original") == 0).astype(int)
    assert not cleaned.isna().any().any()
    assert set(cleaned["target_maligno"].unique()) == {0, 1}

    raw_path = BASE / "datos/datos_org/wdbc_original_v1.csv"
    clean_path = BASE / "datos/datos_limp/wdbc_clean_v1.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    original.to_csv(raw_path, index=False)
    cleaned.to_csv(clean_path, index=False)

    x = cleaned.drop(columns=["target_maligno"])
    q1, q3 = x.quantile(.25), x.quantile(.75)
    iqr = q3 - q1
    outliers = ((x < q1 - 1.5*iqr) | (x > q3 + 1.5*iqr)).sum().sort_values(ascending=False)
    report = {
        "dataset_source": "scikit-learn load_breast_cancer (UCI Wisconsin Diagnostic Breast Cancer)",
        "target_original": "0=malignant, 1=benign",
        "target_cleaned": "1=malignant, 0=benign",
        "original_rows": len(original),
        "cleaned_rows": len(cleaned),
        "predictor_columns": x.shape[1],
        "missing_values": original_missing,
        "duplicate_rows": original_duplicates,
        "class_counts": {str(k): int(v) for k,v in cleaned['target_maligno'].value_counts().sort_index().items()},
        "outliers_IQR_top_5": {k: int(v) for k,v in outliers.head().items()},
        "raw_sha256": sha256_file(raw_path),
        "clean_sha256": sha256_file(clean_path),
        "scaling": "Performed in the training-fold pipeline, not globally before splitting"
    }
    evidence = BASE / 'evidencias'
    evidence.mkdir(exist_ok=True)
    (evidence/'quality_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7,4))
    sns.countplot(data=cleaned, x='target_maligno', hue='target_maligno', palette='Blues', legend=False, ax=ax)
    ax.set_xticks([0,1], ['Benigno','Maligno'])
    ax.set(title='Distribución de clases', xlabel='Diagnóstico', ylabel='Observaciones')
    fig.tight_layout()
    fig.savefig(evidence/'class_distribution.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax, feature in zip(axes, ['mean_radius','mean_concave_points']):
        sns.histplot(data=cleaned, x=feature, hue='target_maligno', bins=25,
                     stat='density', common_norm=False, alpha=.5, ax=ax)
        ax.set(title=f'Distribución: {feature}')
    fig.tight_layout()
    fig.savefig(evidence/'feature_distributions.png', dpi=160)
    plt.close(fig)
    print('Preparación finalizada:', len(cleaned), 'observaciones;', x.shape[1], 'variables.')
    print('SHA-256 de datos limpios:', report['clean_sha256'])
    return report

if __name__ == '__main__':
    prepare_data()
