"""Actividad 5: two classifiers, five-fold CV, GridSearchCV, and MLflow tracking.
Usage: python fuentes/datos_prep.py && python fuentes/train.py
For a local metrics-only smoke test: python fuentes/train.py --no-mlflow
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay,
                             RocCurveDisplay)

BASE = Path(__file__).resolve().parents[1]


def run_training(log_to_mlflow=True):
    clean_path = BASE/'datos/datos_limp/wdbc_clean_v1.csv'
    if not clean_path.exists():
        raise FileNotFoundError('Run python fuentes/datos_prep.py first')
    df = pd.read_csv(clean_path)
    X = df.drop(columns=['target_maligno'])
    y = df['target_maligno']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    choices = {
        'Logistic Regression': (
            Pipeline([('scaler', StandardScaler()),
                      ('model', LogisticRegression(max_iter=3000, solver='liblinear',random_state=42))]),
            {'model__C':[.1, 1, 10], 'model__class_weight':[None,'balanced']}),
        'Random Forest': (
            Pipeline([('model', RandomForestClassifier(random_state=42,n_jobs=1))]),
            {'model__n_estimators':[100,200], 'model__max_depth':[None,8],
             'model__min_samples_leaf':[1,3]})
    }
    evidence = BASE/'evidencias'
    evidence.mkdir(exist_ok=True)
    dataset_sha = hashlib.sha256(clean_path.read_bytes()).hexdigest()

    if log_to_mlflow:
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature
        mlflow.set_tracking_uri(f"sqlite:///{BASE / 'mlflow.db'}")
        exp_name='Actividad5_Comparacion_Modelos'
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name, artifact_location=(BASE/'mlartifacts').resolve().as_uri())
        mlflow.set_experiment(exp_name)

    rows = []
    fitted = {}
    for name,(pipe,grid) in choices.items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True)
        search.fit(X_train,y_train)
        predictions=search.predict(X_test)
        probabilities=search.predict_proba(X_test)[:,1]
        vals={
            'accuracy':accuracy_score(y_test,predictions),
            'precision':precision_score(y_test,predictions),
            'recall':recall_score(y_test,predictions),
            'f1':f1_score(y_test,predictions),
            'roc_auc':roc_auc_score(y_test,probabilities),
            'cv_auc_mean':float(search.best_score_),
            'cv_auc_std':float(search.cv_results_['std_test_score'][search.best_index_])
        }
        rows.append({'Model':name,**vals})
        fitted[name]=search.best_estimator_
        print(name, 'params:',search.best_params_, 'metrics:',vals)

        if log_to_mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(search.best_params_)
                mlflow.log_params({'cv_folds':5,'test_size':.20,'random_state':42,'dataset_sha256':dataset_sha})
                mlflow.log_metrics(vals)
                mlflow.set_tags({'project':'Actividad 5','dataset':'Wisconsin Breast Cancer',
                                 'data_version':'v1','positive_class':'Malignant',
                                 'selection_metric':'ROC-AUC'})
                signature=infer_signature(X_train.head(5), search.predict(X_train.head(5)))
                mlflow.sklearn.log_model(
                    sk_model=search.best_estimator_,
                    name=name.lower().replace(' ','_'),
                    signature=signature, input_example=X_train.head(5),
                    serialization_format='skops',
                    skops_trusted_types=['sklearn.tree._tree.Tree'])

    result=pd.DataFrame(rows)
    result.to_csv(evidence/'model_comparison.csv',index=False)
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    for ax,(name,model) in zip(axes,fitted.items()):
        ConfusionMatrixDisplay.from_estimator(
            model,X_test,y_test,display_labels=['Benigno','Maligno'],
            cmap='Blues',colorbar=False,ax=ax)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(evidence/'confusion_matrices.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,model in fitted.items():
        RocCurveDisplay.from_estimator(model,X_test,y_test,name=name,ax=ax)
    ax.plot([0,1],[0,1],'k--',alpha=.4)
    ax.set(title='Curvas ROC: clasificación binaria')
    fig.tight_layout()
    fig.savefig(evidence/'roc_comparison.png',dpi=160)
    plt.close(fig)
    print('Entrenamiento finalizado. Resultados en evidencias/.')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-mlflow',action='store_true',help='Local test without tracking')
    args=parser.parse_args()
    run_training(log_to_mlflow=not args.no_mlflow)
