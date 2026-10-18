"""Generate decision-oriented UCI visuals beyond the minimum chart requirement."""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]; FIG=ROOT/'reports/figures'; PROC=ROOT/'data/processed'; OUT=ROOT/'outputs'
FIG.mkdir(parents=True,exist_ok=True); sns.set_theme(style='whitegrid',palette='colorblind')
orders=pd.read_parquet(PROC/'uci_orders.parquet'); snaps=pd.read_parquet(PROC/'uci_snapshots.parquet')
orders['month']=orders.event_date.dt.to_period('M'); orders['cohort']=orders.groupby('customer').event_date.transform('min').dt.to_period('M')
orders['month_index']=(orders.month.astype(int)-orders.cohort.astype(int)).astype(int)
cohort=orders.groupby(['cohort','month_index']).customer.nunique().unstack(fill_value=0)
retention=cohort.div(cohort[0],axis=0).iloc[:10,:9]
fig,ax=plt.subplots(figsize=(10,5)); sns.heatmap(retention*100,annot=True,fmt='.0f',cmap='YlGnBu',vmin=0,vmax=100,cbar_kws={'label':'Retention (%)'},ax=ax)
ax.set(title='Cohort retention: tỷ lệ khách còn mua theo tháng kể từ lần mua đầu',xlabel='Số tháng sau cohort',ylabel='Tháng mua đầu'); fig.tight_layout(); fig.savefig(FIG/'08_cohort_retention.png',dpi=180); plt.close(fig)

test=snaps[snaps.split.eq('test')].copy(); test['recency_band']=pd.qcut(test.recency_days,4,duplicates='drop'); test['frequency_band']=pd.qcut(test.frequency,4,duplicates='drop')
pivot=test.pivot_table(index='recency_band',columns='frequency_band',values='churn',aggfunc='mean',observed=False)
fig,ax=plt.subplots(figsize=(9,5)); sns.heatmap(pivot*100,annot=True,fmt='.1f',cmap='Reds',vmin=0,vmax=100,cbar_kws={'label':'Observed no-repeat (%)'},ax=ax)
ax.set(title='Risk map: churn tập trung ở khách recency cao và tần suất thấp',xlabel='Frequency quartile',ylabel='Recency quartile'); fig.tight_layout(); fig.savefig(FIG/'09_rfm_risk_heatmap.png',dpi=180); plt.close(fig)

clv=pd.read_csv(OUT/'bgnbd_gammagamma_clv.csv').rename(columns={'customer_id':'customer'})
clv['customer']=clv['customer'].astype(str); test['customer']=test['customer'].astype(str)
d=test.merge(clv[['customer','clv_90d']],on='customer',how='left').dropna(subset=['clv_90d']).copy()

# Use the fitted calibrated churn model—not recency alone—to define campaign risk.
bundle=joblib.load(ROOT/'models'/'portable_churn_model.joblib')
d['churn_risk']=bundle['model'].predict_proba(d[bundle['features']])[:,1]
risk_cut=float(d['churn_risk'].quantile(.80))  # Top 20% campaign-risk threshold
clv_cut=float(d['clv_90d'].quantile(.75))      # Top 25% value threshold
d['risk_tier']=np.where(d['churn_risk'] >= risk_cut, 'High churn risk (top 20%)', 'Lower churn risk')
d['clv_tier']=np.where(d['clv_90d'] >= clv_cut, 'High CLV (top 25%)', 'Lower CLV')
d['priority_segment']=d['risk_tier'] + ' / ' + d['clv_tier']
segment=(d.groupby('priority_segment', as_index=False)
           .agg(customers=('customer','size'), observed_churn=('churn','mean'), mean_clv_90d=('clv_90d','mean')))
segment.to_csv(OUT/'prescriptive_segments.csv',index=False)
pd.DataFrame([{
    'customers_with_clv':len(d), 'risk_cut_top20':risk_cut, 'clv_cut_top25':clv_cut,
    'high_risk_high_clv_customers':int(((d.churn_risk >= risk_cut) & (d.clv_90d >= clv_cut)).sum()),
    'high_risk_low_clv_customers':int(((d.churn_risk >= risk_cut) & (d.clv_90d < clv_cut)).sum())
}]).to_csv(OUT/'prescriptive_thresholds.csv',index=False)

palette={'High churn risk (top 20%) / High CLV (top 25%)':'#d73027',
         'High churn risk (top 20%) / Lower CLV':'#fc8d59',
         'Lower churn risk / High CLV (top 25%)':'#4575b4',
         'Lower churn risk / Lower CLV':'#91bfdb'}
fig,ax=plt.subplots(figsize=(9,5.5))
for group, part in d.groupby('priority_segment'):
    ax.scatter(part.churn_risk, np.log1p(part.clv_90d), s=18, alpha=.45, color=palette[group], label=group)
ax.axvline(risk_cut,color='#555',linestyle='--',linewidth=1.2,label=f'Top-20% risk cutoff = {risk_cut:.2f}')
ax.axhline(np.log1p(clv_cut),color='#555',linestyle=':',linewidth=1.2,label=f'Top-25% CLV cutoff = £{clv_cut:,.0f}')
ax.set(title='Retention prioritization: calibrated churn risk × estimated 90-day CLV', xlabel='Predicted churn probability', ylabel='log(1 + estimated 90-day CLV, GBP)')
ax.legend(title='Action segment',fontsize=8,loc='best'); fig.tight_layout(); fig.savefig(FIG/'10_risk_clv_priority.png',dpi=180); plt.close(fig)
