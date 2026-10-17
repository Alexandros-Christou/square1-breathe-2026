"""Reproduce Breathe predictions using only the three shipped CSVs."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

WEATHER=['temp_c','temp_max_c','temp_min_c','rh_pct','wind_ms','precip_mm','pressure_kpa','ghi_kwh_m2']
def features(frame, categories):
    dt=pd.to_datetime(frame['date'])
    x=frame[WEATHER+['lat_r','lon_r']].copy()
    for col in ['site','division']:
        x[col]=pd.Categorical(frame[col],categories=categories[col])
    x['month']=dt.dt.month
    x['doy_sin']=np.sin(2*np.pi*dt.dt.dayofyear/365.25)
    x['doy_cos']=np.cos(2*np.pi*dt.dt.dayofyear/365.25)
    x['weekday']=dt.dt.dayofweek
    x['temp_range']=frame['temp_max_c']-frame['temp_min_c']
    x['year']=dt.dt.year
    return x

def history(train,other,stat='mean'):
    a=train.assign(month=pd.to_datetime(train.date).dt.month)
    b=other.assign(month=pd.to_datetime(other.date).dt.month)
    lookup=a.groupby(['site','month']).pm25.agg(stat)
    out=np.array([lookup.get((s,m),np.nan) for s,m in zip(b.site,b.month)])
    fallback=a.groupby('site').pm25.agg(stat)
    return np.array([v if np.isfinite(v) else fallback.get(s,float(a.pm25.agg(stat))) for v,s in zip(out,b.site)])

def train_model(train,other,kind, categories):
    params=dict(random_state=1701,categorical_features='from_dtype',max_bins=255)
    if kind=='reference_analogue':
        params.update(loss='squared_error',max_iter=100)
    elif kind=='robust':
        params.update(loss='absolute_error',max_iter=260,learning_rate=.07,max_leaf_nodes=31,min_samples_leaf=65,l2_regularization=15)
    else: raise ValueError(kind)
    model=HistGradientBoostingRegressor(**params)
    with threadpool_limits(limits=4):
        model.fit(features(train,categories),train.pm25)
        out=model.predict(features(other,categories))
    return np.maximum(out,0)

def evaluate(train,valid,categories, label):
    y=valid.pm25.to_numpy();preds={}
    preds['site_month_mean']=history(train,valid,'mean')
    preds['site_month_median']=history(train,valid,'median')
    preds['reference_analogue']=train_model(train,valid,'reference_analogue',categories)
    preds['robust']=train_model(train,valid,'robust',categories)
    preds['robust_blend']=.85*preds['robust']+.15*preds['site_month_median']
    scores={k:float(mean_absolute_error(y,v)) for k,v in preds.items()}
    chosen=preds['robust']
    err=pd.DataFrame({'division':valid.division,'actual':y,'abs_error':abs(y-chosen)})
    return dict(label=label,train_rows=len(train),valid_rows=len(valid),train_end=train.date.max(),valid_start=valid.date.min(),valid_end=valid.date.max(),scores=scores,
      division_mae=err.groupby('division').abs_error.mean().to_dict(),
      tail_rows=int((y>=15).sum()),tail_mae=float(np.mean(abs(y[y>=15]-chosen[y>=15]))),
      ordinary_mae=float(np.mean(abs(y[y<15]-chosen[y<15]))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('data'));ap.add_argument('--out',type=Path,default=Path('output'));ap.add_argument('--predict-only',action='store_true');args=ap.parse_args()
    tr=pd.read_csv(args.data/'train.csv');te=pd.read_csv(args.data/'test.csv');sample=pd.read_csv(args.data/'sample_submission.csv')
    assert set(tr.columns)==set(te.columns)|{'pm25'}
    assert not tr.id.duplicated().any() and not te.id.duplicated().any()
    assert len(te)==66699 and tr.date.max()<te.date.min()
    categories={c:sorted(tr[c].unique()) for c in ['site','division']}
    args.out.mkdir(parents=True,exist_ok=True)
    if not args.predict_only:
        results=[]
        results.append(evaluate(tr[tr.date<'2024-04-01'],tr[(tr.date>='2024-04-01')&(tr.date<'2025-01-01')],categories,'Development: April–December2024 after earlier dates only'))
        results.append(evaluate(tr[tr.date<'2025-01-01'],tr[tr.date>='2025-01-01'],categories,'Confirmation: January–March2025 after earlier dates only'))
        print(json.dumps(results,ensure_ascii=True),flush=True)
        (args.out/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
    # Robust loss selected for MAE and smoke-tail resistance; no public-board tuning.
    pred=train_model(tr,te,'robust',categories)
    output=pd.DataFrame({'id':te.id,'pm25':pred}).set_index('id').loc[sample.id].reset_index()
    assert output.id.tolist()==sample.id.tolist() and np.isfinite(output.pm25).all()
    target=args.out/'submission.csv';output.to_csv(target,index=False,float_format='%.8f')
    summary=dict(rows=len(output),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),negative_train_values=int((tr.pm25<0).sum()),training_shape=list(tr.shape),test_shape=list(te.shape),train_date_range=[tr.date.min(),tr.date.max()],test_date_range=[te.date.min(),te.date.max()],features_used='Weather,roundedcoordinates,site,division,andcalendar only; ID excluded. No external files/targets.',model='HistGradientBoostingRegressor absolute_error seed1701',prediction_range=[float(pred.min()),float(pred.max())])
    (args.out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
