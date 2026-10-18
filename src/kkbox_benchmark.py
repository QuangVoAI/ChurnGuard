"""Reproducible KKBox benchmark from the cited public, pre-engineered feature matrices.

The source split is supplied as X_train/Y_train and X_val/Y_val.  The held-out
matrix is divided stratified 50:50 into calibration and final test because it
contains no timestamp.  Thus this is a labelled benchmark, not temporal proof.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw/kkbox_derived'; OUT=ROOT/'outputs'
Xtr=pd.read_csv(RAW/'X_train.csv'); ytr=pd.read_csv(RAW/'Y_train.csv').iloc[:,0].astype(int)
Xhold=pd.read_csv(RAW/'X_val.csv'); yhold=pd.read_csv(RAW/'Y_val.csv').iloc[:,0].astype(int)
Xcal,Xte,ycal,yte=train_test_split(Xhold,yhold,test_size=.5,random_state=42,stratify=yhold)

def score(name, model):
    model.fit(Xtr,ytr); raw=model.predict_proba(Xte)[:,1]
    # Calibration is fitted only on the calibration half, never on final test.
    cal=CalibratedClassifierCV(model, method='sigmoid', cv='prefit').fit(Xcal,ycal)
    p=cal.predict_proba(Xte)[:,1]; prev=yte.mean(); n=max(1,int(np.ceil(.1*len(yte))))
    top=np.argsort(p)[-n:]; precision=yte.iloc[top].mean(); b=brier_score_loss(yte,p)
    return {'dataset':'KKBox labelled benchmark','model':name,'n_test':len(yte),'churn_prevalence':prev,
            'roc_auc':roc_auc_score(yte,p),'pr_auc':average_precision_score(yte,p),'brier':b,
            'brier_baseline':prev*(1-prev),'brier_skill_score':1-b/(prev*(1-prev)),
            'precision_at_10':precision,'recall_at_10':yte.iloc[top].sum()/yte.sum(),'lift_at_10':precision/prev}

models=[('Logistic Regression baseline',make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=300,class_weight='balanced',n_jobs=-1))),
        ('HistGradientBoosting',make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(max_iter=180,learning_rate=.08,max_leaf_nodes=31,random_state=42)))]
res=pd.DataFrame([score(n,m) for n,m in models]); res.to_csv(OUT/'kkbox_benchmark_metrics.csv',index=False)
(OUT/'kkbox_benchmark_protocol.json').write_text(json.dumps({'source':'YLTsai0609 public pre-engineered KKBox feature matrices','raw_source':'KKBox WSDM 2018 competition','train_rows':len(Xtr),'calibration_rows':len(Xcal),'test_rows':len(Xte),'split':'Provided train matrix; stratified 50:50 calibration/final-test split of provided validation matrix, seed=42','feature_columns':Xtr.columns.tolist(),'limitation':'Derived public matrices; no raw event timestamps, so benchmark split is stratified user holdout rather than temporal.'},indent=2))
print(res.to_string(index=False))
