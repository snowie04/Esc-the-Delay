import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from raw_features import SCHEMA, build_features, read_uploads, template_zip

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='ESC the Delay | Delivery Risk', page_icon='📦', layout='wide')
st.html('''<style>
.stApp{background:#f5faf9;color:#153b3a} [data-testid="stHeader"]{background:#f5faf9dd}
.block-container{max-width:1320px;padding-top:2.5rem;padding-bottom:3rem}
h1,h2,h3{color:#123e3b!important;letter-spacing:-.035em}
[data-testid="stSidebar"]{background:#e9f5f1;border-right:1px solid #d1e9e2}
.brand{font-size:23px;font-weight:800;color:#126c61}.brand-sub{color:#5b7a73;font-size:11px;letter-spacing:.12em;text-transform:uppercase}
.eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:11px;font-weight:750;color:#248278}
.hero{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:38px 42px;background:linear-gradient(115deg,#064d49,#0a7c70 75%,#14947d);border-radius:24px;margin:15px 0 22px;overflow:hidden}
.hero h1{color:white!important;font-size:clamp(30px,3.5vw,48px);line-height:1.12;margin:12px 0 16px;letter-spacing:-.04em}
.hero p{color:#d6f3e9;font-size:15px;max-width:570px;line-height:1.75;margin:0}.hero .eyebrow{color:#b5f5d9}
.hero>div:first-child{flex:1;min-width:0}.hero-art{flex:0 0 220px;width:220px}.parcel-scene{position:relative;height:220px}.route-orbit{position:absolute;inset:12px;border:2px dashed #8ee4c080;border-radius:50%}.parcel{position:absolute;left:25px;top:16px;font-size:100px;transform:rotate(-10deg)}.truck{position:absolute;right:-8px;bottom:20px;font-size:96px}.delivery-check{position:absolute;right:8px;top:12px;background:#c9ffe1;color:#087b64;border-radius:50%;width:42px;height:42px;text-align:center;line-height:42px;font-size:26px;font-weight:800}.delivery-tag{position:absolute;bottom:-8px;left:9px;font-size:9px;letter-spacing:.1em;color:#c6f9e5}.eyebrow{opacity:1}.pill{display:inline-block;font-size:11px;padding:6px 11px;border-radius:30px;color:#d7ffee;background:#ffffff15;border:1px solid #ffffff30;margin-top:20px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px}.step{border:1px solid #dcebe6;background:white;border-radius:15px;padding:18px 20px}.step b{font-size:14px;color:#1e4e46}.step p{font-size:12px;line-height:1.6;color:#668079;margin:7px 0 0}
.num{background:#e3f5ee;color:#157664;padding:5px 8px;border-radius:7px;font-size:11px;margin-right:8px}
button[kind="primary"]{background:#087f78!important;border-color:#087f78!important;border-radius:10px}
[data-testid="stDownloadButton"] button{border-color:#b5d8cc;color:#116a5a;border-radius:10px}
[data-testid="stFileUploader"] section{background:#edf7f3;border:1px dashed #9dcdbd;border-radius:14px}
[data-testid="stMetric"]{background:white;border:1px solid #dcebe6;border-radius:14px;padding:18px}[data-testid="stMetricValue"]{color:#08766c}
button[data-baseweb="tab"]{color:#517970}button[data-baseweb="tab"][aria-selected="true"]{color:#087f78}[data-baseweb="tab-highlight"]{background-color:#087f78}
.footer{border-top:1px solid #dcebe6;padding-top:18px;margin-top:40px;color:#76928a;font-size:11px}
@media(max-width:850px){.hero-art{display:none}.hero{padding:28px}.steps{grid-template-columns:1fr}}
</style>''')

@st.cache_resource
def load_assets():
    with (ROOT/'config/model_config.json').open(encoding='utf-8') as f:
        config=json.load(f)
    return config,joblib.load(ROOT/'models/delivery_pipeline.joblib')

with st.sidebar:
    st.html('<div class="brand">📦 ESC the Delay</div><div class="brand-sub">Delivery intelligence</div>')
    st.divider()
    st.markdown('**A head start on late deliveries.**')
    st.write('Turn everyday order data into a prioritized delivery watchlist. Spot risk at payment approval and help your team focus follow-ups.')
    st.divider()
    st.markdown('**Your workflow**\n\n1. Upload raw store data\n2. Build features automatically\n3. Download your risk watchlist')
    st.divider()
    st.caption('XGBoost · 37 model features\n\nBatch scoring at payment approval')

st.html('''<div class="eyebrow">ESC the Delay / Operations workspace</div>
<div class="hero"><div><div class="eyebrow">E-commerce delivery intelligence</div><h1>See delivery risks.<br>Act before the delay.</h1><p>Upload your raw order and store data. We connect the tables, build the model features, and highlight orders that deserve a closer look.</p><span class="pill">RAW DATA IN &nbsp; → &nbsp; ACTIONABLE WATCHLIST OUT</span></div>
<div class="hero-art parcel-scene" aria-label="Delivery parcels and a delivery truck"><div class="route-orbit"></div><div class="parcel">📦</div><div class="truck">🚚</div><div class="delivery-check">✓</div><div class="delivery-tag">READY FOR THE NEXT MILE</div></div></div>
<div class="steps"><div class="step"><b><span class="num">01</span>Connect your data</b><p>Orders, items, sellers, customers, products and locations.</p></div><div class="step"><b><span class="num">02</span>Let the model prepare</b><p>Automatic joins, distance calculations and historical signals.</p></div><div class="step"><b><span class="num">03</span>Focus your follow-ups</b><p>Rank orders by risk and export a ready-to-review watchlist.</p></div></div>''')
try:
    config,model=load_assets()
except Exception:
    st.error('The model or configuration could not be loaded. Check the models and config folders.')
    st.stop()
features=config['feature_names'];threshold=float(config['threshold'])
prediction_tab,report_tab,guide_tab=st.tabs(['Predict deliveries','Model performance','Data guide'])

with prediction_tab:
    st.subheader('Start with your raw data')
    st.write('Select the six Olist-format CSV files together. Include historical orders and the new orders you want to score in the same tables.')
    st.caption('No manual cleaning or model feature preparation is required. Blank delivery dates are expected for new orders.')
    with st.expander('Which files do I need?'):
        st.dataframe(pd.DataFrame({'File':[f'olist_{k}_dataset.csv' for k in SCHEMA],'Purpose':['New orders + order history','Products and sellers in each order','Customer locations','Seller locations','Product categories, weights and dimensions','Postal code coordinates']}),hide_index=True,width='stretch')
        st.write('Payments, reviews and the category translation file are not needed by this model.')
        st.download_button('Download raw CSV templates',template_zip(),'raw_data_templates.zip','application/zip')
    uploads=st.file_uploader('Upload raw store tables',type='csv',accept_multiple_files=True)
    scope=st.radio('Choose orders to score',['Open orders','Historical replay'],horizontal=True,help='Open orders have no recorded delivery date. Historical replay scores approved orders as of their approval time, including delivered orders.')
    st.caption("Scoring always uses payment approval as the prediction time; this is not a live rescore using today's shipment status.")
    signature=(tuple((f.name,hashlib.sha256(f.getvalue()).hexdigest()) for f in uploads),scope)
    if st.session_state.get('source_signature')!=signature:
        st.session_state.pop('prediction_result',None);st.session_state.pop('prediction_audit',None)
        st.session_state['source_signature']=signature
    if uploads:
        filenames={Path(f.name.replace('\\','/')).name for f in uploads}
        missing=[f'olist_{k}_dataset.csv' for k in SCHEMA if f'olist_{k}_dataset.csv' not in filenames]
        if missing:st.info('Still needed: '+', '.join(missing))
        else:st.success('All six raw tables are ready for validation.')
        if st.button('Analyze delivery risk',type='primary',disabled=bool(missing)):
            try:
                with st.status('Preparing your delivery watchlist…',expanded=True) as status:
                    st.write('Reading raw tables and checking matching IDs…')
                    tables=read_uploads(uploads);orders=tables['orders']
                    required=['order_id','order_approved_at','order_purchase_timestamp','order_estimated_delivery_date']
                    absent=[c for c in required if c not in orders]
                    if absent:raise ValueError('orders: missing columns: '+', '.join(absent))
                    eligible=pd.Series(True,index=orders.index)
                    for col in required[1:]:eligible &= pd.to_datetime(orders[col],errors='coerce').notna()
                    if scope=='Open orders':
                        delivered=orders.get('order_delivered_customer_date',pd.Series(pd.NaT,index=orders.index))
                        eligible &= pd.to_datetime(delivered,errors='coerce').isna()
                    target_ids=set(orders.loc[eligible,'order_id'])
                    if not target_ids:raise ValueError('No eligible orders in this scope. Orders need purchase, approval and estimated delivery dates. Use Historical replay to test a delivered dataset.')
                    st.write('Joining orders and building approval-time historical features…')
                    built=build_features(tables);scored=built.loc[built['order_id'].isin(target_ids)].copy()
                    excluded=len(target_ids)-len(scored)
                    if scored.empty:raise ValueError('No selected orders have matching order items.')
                    if any(c not in scored for c in features):raise ValueError('The feature builder does not match the model configuration.')
                    st.write(f'Scoring {len(scored):,} orders with the saved model…')
                    matrix=scored[features].apply(pd.to_numeric,errors='coerce').astype('float64').replace([np.inf,-np.inf],np.nan)
                    probabilities=model.predict_proba(matrix)[:,1]
                    if not np.isfinite(probabilities).all():raise ValueError('The model returned invalid risk scores.')
                    result=scored[['order_id','order_approved_at']].copy();result['risk_score']=probabilities
                    result['predicted_late']=(probabilities>=threshold).astype(int)
                    result['risk_status']=np.where(result.predicted_late.eq(1),'Flagged for review','Below alert threshold')
                    result['recommended_action']=np.where(result.predicted_late.eq(1),'Review seller readiness and delivery plan','Continue routine monitoring')
                    history=scored[['seller_late_rate_30d','route_transit_mean_90d']].notna().all(axis=1)
                    st.session_state['prediction_result']=result.sort_values('risk_score',ascending=False)
                    st.session_state['prediction_audit']={'excluded':excluded,'history_coverage':float(history.mean())}
                    status.update(label='Your watchlist is ready',state='complete',expanded=False)
            except Exception as exc:st.error(f'Unable to score this upload: {exc}')
    if 'prediction_result' in st.session_state:
        result=st.session_state['prediction_result'];audit=st.session_state['prediction_audit']
        st.divider();st.subheader('Your delivery watchlist')
        a,b,c=st.columns(3);a.metric('Orders analyzed',f'{len(result):,}');b.metric('Flagged for review',f'{result.predicted_late.sum():,}');c.metric('Alert rate',f'{result.predicted_late.mean():.2%}')
        if audit['excluded']:st.warning(f"{audit['excluded']:,} selected orders were excluded because matching items were unavailable.")
        if audit['history_coverage']<.9:st.warning(f"Only {audit['history_coverage']:.1%} of scored orders have both recent seller outcomes and route transit history. Missing history uses the saved model imputer; limited history can reduce reliability.")
        st.caption(f'Alert threshold: {threshold:.10f}. Scores rank delivery risk; probability calibration has not been validated. Below-threshold orders may still arrive late.')
        flagged_only=st.checkbox('Show flagged orders only')
        visible=result.loc[result.predicted_late.eq(1)] if flagged_only else result
        st.dataframe(visible,hide_index=True,width='stretch',column_config={'risk_score':st.column_config.ProgressColumn('Risk score',min_value=0,max_value=1,format='%.3f')})
        st.download_button('Download full watchlist',result.to_csv(index=False).encode('utf-8-sig'),'delivery_watchlist.csv','text/csv')
        st.download_button('Download flagged orders',result.loc[result.predicted_late.eq(1)].to_csv(index=False).encode('utf-8-sig'),'flagged_orders.csv','text/csv')

with report_tab:
    st.subheader('Selected model — XGBoost');st.write('37 features · Untouched test split · Fixed alert threshold')
    m=config['test_metrics']
    for row in [[('PR-AUC / AP','pr_auc','.4f'),('ROC-AUC','roc_auc','.4f')],[('Precision','precision','.2%'),('Recall','recall','.2%')],[('F1-score','f1','.4f'),('F2-score','f2','.4f')]]:
        for column,(label,key,fmt) in zip(st.columns(2),row):column.metric(label,format(m[key],fmt))
    st.divider();a,b=st.columns(2);a.metric('Selected threshold',f'{threshold:.4f}');b.metric('Test alert rate',f"{m['predicted_positive_rate']:.2%}")
    st.caption('Historical test set: 19,299 orders. An 80/20 label-stratified split was made within each approval month. This is not a future-month holdout. These fixed test metrics do not change with uploads.')

with guide_tab:
    st.subheader('From store exports to delivery decisions')
    st.markdown('''**What does this app do?**  
It flags orders at elevated risk of arriving after their estimated delivery date, so operations teams can prioritize follow-ups.

**What should I upload?**  
Six raw Olist-format tables: orders, order items, customers, sellers, products and geolocation. Include historical and new orders together. Keep IDs consistent across tables. No 37-feature spreadsheet is needed.

**Why include history?**  
The model uses seller workload, recent seller delay outcomes and route transit times. Include at least the preceding 90 days where available, plus older orders that remain pending or have recent delivery outcomes. A single new order cannot provide these signals. Keep known carrier and delivery timestamps on historical orders; leave unknown future timestamps blank.

**When is risk calculated?**  
At each order's payment approval time. Historical events are counted only if their timestamps are strictly earlier than that approval time. A new order's actual delivery date and eventual late label are not required. Order status does not determine eligibility.

**Can I try the existing dataset?**  
Upload the six original raw CSVs and choose **Historical replay**. The app reconstructs approval-time features for eligible orders, including completed orders. This is a demonstration, not a new independent test.

**What happens automatically?**  
Table validation, order aggregation, distance and calendar features, seller and route history, missing-value imputation, model scoring and watchlist export.

**How should I use the scores?**  
Start with flagged orders and review seller readiness and delivery plans. Scores are risk indicators, not guaranteed outcomes. The model was trained on historical Olist data; performance on a different store or period needs validation.
''')
    with st.expander('Raw table columns'):
        for name,columns in SCHEMA.items():st.markdown(f'**olist_{name}_dataset.csv**');st.code(', '.join(columns),language=None)
        st.write('Optional on orders: order_status, order_delivered_carrier_date, order_delivered_customer_date. Include known delivery timestamps for history. Optional on order items: shipping_limit_date.')
st.html('<div class="footer">ESC the Delay · E-commerce delivery intelligence · Built for focused operational follow-ups</div>')
