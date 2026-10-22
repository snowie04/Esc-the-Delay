"""Approval-time inference features adapted from build_olist_features_v3_corrected.py.

Preserves training formulas, units, aggregation and strictly prior history.
Unlike the training builder, retains approved orders without delivery outcomes.
"""
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA = {
    "orders": ["order_id", "customer_id", "order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date"],
    "order_items": ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
    "customers": ["customer_id", "customer_zip_code_prefix", "customer_state"],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_state"],
    "products": ["product_id", "product_category_name", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
    "geolocation": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
}
OPTIONAL = {
    "orders": ["order_status", "order_delivered_carrier_date", "order_delivered_customer_date"],
    "order_items": ["shipping_limit_date"],
}
NUMERIC = {
    "order_items": ["order_item_id", "price", "freight_value"],
    "products": ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
    "geolocation": ["geolocation_lat", "geolocation_lng"],
}

def read_uploads(files):
    tables = {}
    for file in files:
        name = Path(file.name.replace("\\", "/")).name
        key = next((k for k in SCHEMA if name == f"olist_{k}_dataset.csv"), None)
        if key is None:
            continue
        if key in tables:
            raise ValueError(f"Duplicate file: {name}")
        identifiers = {c: "string" for c in SCHEMA[key] if c.endswith("_id") or "zip_code_prefix" in c}
        tables[key] = pd.read_csv(io.BytesIO(file.getvalue()), dtype=identifiers, low_memory=False)
    return tables


def template_zip():
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for key, columns in SCHEMA.items():
            archive.writestr(f"olist_{key}_dataset.csv", pd.DataFrame(columns=columns + OPTIONAL.get(key, [])).to_csv(index=False))
    return result.getvalue()


def validate_tables(tables):
    missing = [f"olist_{key}_dataset.csv" for key in SCHEMA if key not in tables]
    if missing:
        raise ValueError("Missing raw files: " + ", ".join(missing))
    clean = {}
    for key, columns in SCHEMA.items():
        frame = tables[key].copy()
        absent = [c for c in columns if c not in frame]
        if absent:
            raise ValueError(f"{key}: missing columns: " + ", ".join(absent))
        if frame.empty:
            raise ValueError(f"{key}: the table is empty.")
        for col in OPTIONAL.get(key, []):
            if col not in frame:
                frame[col] = np.nan
        for col in columns:
            if col.endswith("_id") or "zip_code_prefix" in col:
                frame[col] = frame[col].astype("string").str.strip().replace("", pd.NA)
        for col in NUMERIC.get(key, []):
            converted = pd.to_numeric(frame[col], errors="coerce")
            if (frame[col].notna() & (converted.isna() | ~np.isfinite(converted))).any():
                raise ValueError(f"{key}.{col}: contains invalid numeric values.")
            frame[col] = converted
        unique = {"orders": ["order_id"], "customers": ["customer_id"], "sellers": ["seller_id"], "products": ["product_id"], "order_items": ["order_id", "order_item_id"]}.get(key)
        if unique and (frame[unique].isna().any().any() or frame.duplicated(unique).any()):
            raise ValueError(f"{key}: IDs must be present and unique: {unique}")
        clean[key] = frame
    for source, col, target in [("orders", "customer_id", "customers"), ("order_items", "order_id", "orders"), ("order_items", "seller_id", "sellers"), ("order_items", "product_id", "products")]:
        if (~clean[source][col].isin(clean[target][col])).any():
            raise ValueError(f"{source}.{col}: some IDs are absent from {target}. Upload matching reference tables.")
    for col in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        values = clean["orders"][col]
        parsed = pd.to_datetime(values, errors="coerce")
        if (values.notna() & values.astype(str).str.strip().ne("") & parsed.isna()).any():
            raise ValueError(f"orders.{col}: invalid dates. Use YYYY-MM-DD HH:MM:SS.")
        clean["orders"][col] = parsed
    clean["order_items"]["shipping_limit_date"] = pd.to_datetime(clean["order_items"]["shipping_limit_date"], errors="coerce")
    return clean


V3 = ['seller_dispatch_late_rate_90d','seller_handling_std_90d','seller_order_growth_7d','route_transit_mean_90d','route_transit_std_90d','delivery_time_pressure','seller_state_order_growth_7d','max_item_weight_g','seller_late_rate_smoothed_90d','route_late_rate_smoothed_90d','seller_cold_start','route_cold_start','shipping_deadline_slack_days','total_item_price','max_item_volume_cm3']
BASE = ['purchase_month','purchase_weekday','purchase_is_weekend','after_5pm','estimated_delivery_days','item_count','seller_count','total_freight','total_weight_g','total_volume_cm3','category_count','seller_customer_distance_km','same_state','freight_price_ratio','distance_x_weight','distance_x_volume','distance_per_delivery_day']
NEW = ['purchase_year','approval_weekday','approval_hour','approval_month','estimated_delivery_slack','seller_order_count_7d','seller_order_count_30d','seller_late_rate_30d','seller_late_history_n_30d','seller_pending_orders','route_order_count_30d','route_late_rate_30d','route_late_history_n_30d','platform_order_count_7d','platform_order_count_30d','seller_avg_handling_days_90d','weight_vs_seller_avg_90d']

def history_counts(query_t, query_key, event_t, event_key, days, values=None):
    """Strictly prior events, grouped by key. For outcomes, event time is knowledge time."""
    ans = np.full(len(query_t), np.nan if values is not None else 0., dtype=float)
    q = pd.DataFrame({'key':pd.Series(query_key).fillna('__MISSING__').astype(str), 'time':pd.to_datetime(query_t), 'ix':np.arange(len(query_t))})
    e = pd.DataFrame({'key':pd.Series(event_key).fillna('__MISSING__').astype(str), 'time':pd.to_datetime(event_t)})
    if values is not None: e['val'] = np.asarray(values,dtype=float)
    e = e[e.time.notna()]
    if values is not None: e=e[e.val.notna()]
    groups={k:g for k,g in e.groupby('key',sort=False)}
    for key,g in q.groupby('key',sort=False):
        ev=groups.get(key)
        if ev is None: continue
        ts=ev.time.to_numpy(dtype='datetime64[ns]')
        ix=np.argsort(ts,kind='stable'); ts=ts[ix]
        qt=g.time.to_numpy(dtype='datetime64[ns]')
        right=np.searchsorted(ts,qt,side='left')
        left=np.searchsorted(ts,qt-np.timedelta64(days,'D'),side='left')
        cnt=right-left
        if values is None: ans[g.ix.to_numpy()]=cnt
        else:
            v=ev.val.to_numpy()[ix]
            cs=np.r_[0.,np.cumsum(v)]
            sums=cs[right]-cs[left]
            ans[g.ix.to_numpy()]=np.divide(sums,cnt,out=np.full(len(cnt),np.nan),where=cnt>0)
    return ans

def build_features(tables):
    tables = validate_tables(tables)
    orders=tables["orders"].copy(); items=tables["order_items"].copy()
    customers=tables["customers"].copy(); sellers=tables["sellers"].copy()
    products=tables["products"].copy(); geo=tables["geolocation"].copy()
    for c in ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']:
        orders[c]=pd.to_datetime(orders[c],errors='coerce')
    items['shipping_limit_date']=pd.to_datetime(items['shipping_limit_date'],errors='coerce')
    geo=geo.groupby('geolocation_zip_code_prefix',as_index=False)[['geolocation_lat','geolocation_lng']].median()
    sellers=sellers.merge(geo,left_on='seller_zip_code_prefix',right_on='geolocation_zip_code_prefix',how='left').rename(columns={'geolocation_lat':'seller_lat','geolocation_lng':'seller_lng'}).drop(columns='geolocation_zip_code_prefix')
    customers=customers.merge(geo,left_on='customer_zip_code_prefix',right_on='geolocation_zip_code_prefix',how='left').rename(columns={'geolocation_lat':'customer_lat','geolocation_lng':'customer_lng'}).drop(columns='geolocation_zip_code_prefix')
    products['volume_cm3']=products['product_length_cm']*products['product_height_cm']*products['product_width_cm']
    item=items.merge(products[['product_id','product_category_name','product_weight_g','volume_cm3']],on='product_id',how='left').merge(sellers[['seller_id','seller_state','seller_lat','seller_lng']],on='seller_id',how='left')
    item=item.merge(orders[['order_id','customer_id','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','order_purchase_timestamp']],on='order_id',how='left').merge(customers[['customer_id','customer_state','customer_lat','customer_lng']],on='customer_id',how='left')
    # Great-circle distances per item; aggregate to order-level mean.
    lat1=np.radians(item.seller_lat.to_numpy(float));lon1=np.radians(item.seller_lng.to_numpy(float));lat2=np.radians(item.customer_lat.to_numpy(float));lon2=np.radians(item.customer_lng.to_numpy(float))
    h=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    item['distance_km']=6371*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
    item['same_state_item']=(item.seller_state==item.customer_state).where(item.seller_state.notna()&item.customer_state.notna()).astype(float)
    item['route_key']=item.seller_state.fillna('UNK')+'>'+item.customer_state.fillna('UNK')
    # Historical seller events: distinct seller/order pair, so item_count does not inflate workload.
    seller_events=item.sort_values(['order_id','seller_id']).drop_duplicates(['order_id','seller_id']).copy()
    seller_events['shipping_limit_date']=items.groupby(['order_id','seller_id'])['shipping_limit_date'].min().reindex(pd.MultiIndex.from_frame(seller_events[['order_id','seller_id']])).to_numpy()
    seller_events['dispatch_late_known']=np.where(seller_events.order_delivered_carrier_date.notna() & seller_events.shipping_limit_date.notna(),(seller_events.order_delivered_carrier_date>seller_events.shipping_limit_date).astype(float),np.nan)
    seller_events['transit_days']=(seller_events.order_delivered_customer_date-seller_events.order_delivered_carrier_date).dt.total_seconds()/86400
    seller_events['transit_days']=seller_events.transit_days.where(seller_events.transit_days>=0)
    seller_events['late_known']=(seller_events.order_delivered_customer_date.dt.normalize()>seller_events.order_estimated_delivery_date.dt.normalize()).astype(float)
    seller_events.loc[seller_events.order_delivered_customer_date.isna()|seller_events.order_estimated_delivery_date.isna(),'late_known']=np.nan
    seller_events['handling_days']=(seller_events.order_delivered_carrier_date-seller_events.order_approved_at).dt.total_seconds()/86400
    # The target becomes known only when delivered, not at purchase/approval.
    seller_events['outcome_known_at']=seller_events.order_delivered_customer_date
    seller_events['weight_item']=seller_events.product_weight_g
    # Per-seller order weight (not individual item weight).
    seller_events=seller_events.drop(columns='weight_item')
    seller_weight=item.groupby(['order_id','seller_id'],as_index=False)['product_weight_g'].sum(min_count=1).rename(columns={'product_weight_g':'seller_order_weight_g'})
    seller_events=seller_events.merge(seller_weight,on=['order_id','seller_id'],how='left')
    valid=seller_events.order_approved_at.notna()&seller_events.seller_id.notna()
    se=seller_events.loc[valid].reset_index(drop=True)
    qtime=se.order_approved_at; skey=se.seller_id
    se['seller_order_count_7d']=history_counts(qtime,skey,qtime,skey,7) # strictly earlier approval times
    se['seller_order_count_30d']=history_counts(qtime,skey,qtime,skey,30)
    se['seller_late_rate_30d']=history_counts(qtime,skey,se.outcome_known_at,skey,30,se.late_known)
    se['seller_late_history_n_30d']=history_counts(qtime,skey,se.outcome_known_at.where(se.late_known.notna()),skey,30)
    # Only valid approval -> carrier events can close a pending order.
    # Exclude same-timestamp approvals and handovers at prediction time.
    se['seller_pending_orders']=np.nan
    for key,g in se.groupby('seller_id',sort=False):
        approved_t=g.order_approved_at.to_numpy(dtype='datetime64[ns]')
        carrier_t=g.order_delivered_carrier_date.to_numpy(dtype='datetime64[ns]')
        valid_handover=(~np.isnat(carrier_t)) & (carrier_t>=approved_t)
        # Count prior approved orders not yet handed over. A handover
        # cannot cancel an order before its own approval.
        starts=np.sort(approved_t)
        ends=np.sort(carrier_t[valid_handover])
        pending=np.searchsorted(starts,approved_t,side='left')-np.searchsorted(ends,approved_t,side='left')
        if (pending<0).any():
            raise ValueError(f'Negative pending count for seller {key}: check event chronology')
        se.loc[g.index,'seller_pending_orders']=pending
    # Handling time becomes observable at carrier handover.
    se['seller_avg_handling_days_90d']=history_counts(qtime,skey,se.order_delivered_carrier_date,skey,90,se.handling_days.clip(lower=0))
    # Historical dispatch outcomes become known only at carrier handover.
    se['seller_dispatch_late_rate_90d']=history_counts(qtime,skey,se.order_delivered_carrier_date,skey,90,se.dispatch_late_known)
    handling=se.handling_days.where(se.handling_days>=0)
    handling_mean=se.seller_avg_handling_days_90d
    handling_sq=history_counts(qtime,skey,se.order_delivered_carrier_date,skey,90,handling**2)
    se['seller_handling_std_90d']=np.sqrt(np.maximum(handling_sq-handling_mean**2,0))
    # Compare last 7 days with previous 30-day daily baseline.
    se['seller_order_growth_7d']=(se.seller_order_count_7d/7)/(se.seller_order_count_30d/30+0.1)
    se['route_transit_mean_90d']=history_counts(qtime,se.route_key,se.order_delivered_customer_date,se.route_key,90,se.transit_days)
    transit_sq=history_counts(qtime,se.route_key,se.order_delivered_customer_date,se.route_key,90,se.transit_days**2)
    se['route_transit_std_90d']=np.sqrt(np.maximum(transit_sq-se.route_transit_mean_90d**2,0))
    # Smoothing uses only earlier KNOWN delivery outcomes (not all future labels).
    known=se.outcome_known_at.where(se.late_known.notna())
    global_key=pd.Series('all',index=se.index)
    prior=history_counts(qtime,global_key,known,global_key,365,se.late_known)
    prior=np.where(np.isfinite(prior),prior,0.07) # cold-start fallback; no target-derived global constant
    seller_n=history_counts(qtime,skey,known,skey,90)
    seller_r=history_counts(qtime,skey,known,skey,90,se.late_known)
    route_n=history_counts(qtime,se.route_key,known,se.route_key,90)
    route_r=history_counts(qtime,se.route_key,known,se.route_key,90,se.late_known)
    alpha=20.0
    se['seller_late_rate_smoothed_90d']=(seller_r.fillna(0) if hasattr(seller_r,'fillna') else np.nan_to_num(seller_r))*seller_n/(seller_n+alpha)+alpha*prior/(seller_n+alpha)
    se['route_late_rate_smoothed_90d']=np.nan_to_num(route_r)*route_n/(route_n+alpha)+alpha*prior/(route_n+alpha)
    se['seller_cold_start']=(seller_n<5).astype(float)
    se['route_cold_start']=(route_n<5).astype(float)
    # State-specific volume growth (event = approval; no outcomes involved).
    state=se.seller_state.fillna('UNK')
    state7=history_counts(qtime,state,qtime,state,7)
    state30=history_counts(qtime,state,qtime,state,30)
    se['seller_state_order_growth_7d']=(state7/7)/(state30/30+0.1)
    # Pressure is calculated later using approval-time delivery promise.
    se['delivery_time_pressure_components']=se.seller_avg_handling_days_90d+se.route_transit_mean_90d
    se['seller_avg_weight_90d']=history_counts(qtime,skey,qtime,skey,90,se.seller_order_weight_g)
    se['weight_vs_seller_avg_90d']=se.seller_order_weight_g/se.seller_avg_weight_90d.replace(0,np.nan)
    # State-to-state historical route aggregates use only delivered known outcomes.
    rkey=se.route_key
    se['route_order_count_30d']=history_counts(qtime,rkey,qtime,rkey,30)
    se['route_late_rate_30d']=history_counts(qtime,rkey,se.outcome_known_at,rkey,30,se.late_known)
    se['route_late_history_n_30d']=history_counts(qtime,rkey,se.outcome_known_at.where(se.late_known.notna()),rkey,30)
    # Platform volume counts unique orders, not seller events.
    approved=orders.loc[orders.order_approved_at.notna(),['order_id','order_approved_at']].drop_duplicates('order_id')
    se['platform_order_count_7d']=history_counts(qtime,np.repeat('platform',len(se)),approved.order_approved_at,np.repeat('platform',len(approved)),7)
    se['platform_order_count_30d']=history_counts(qtime,np.repeat('platform',len(se)),approved.order_approved_at,np.repeat('platform',len(approved)),30)
    # Aggregating multi-seller orders: mean risk/volume, max backlog, total orders, etc.
    agg={'seller_order_count_7d':'mean','seller_order_count_30d':'mean','seller_late_rate_30d':'mean','seller_late_history_n_30d':'sum','seller_pending_orders':'max','route_order_count_30d':'mean','route_late_rate_30d':'mean','route_late_history_n_30d':'sum','platform_order_count_7d':'first','platform_order_count_30d':'first','seller_avg_handling_days_90d':'mean','weight_vs_seller_avg_90d':'mean',**{c:'mean' for c in V3 if c not in ['delivery_time_pressure','max_item_weight_g','max_item_volume_cm3','shipping_deadline_slack_days','total_item_price']},'delivery_time_pressure_components':'mean'}
    dynamic=se.groupby('order_id').agg(agg).reset_index()
    base=item.groupby('order_id').agg(item_count=('order_item_id','size'),seller_count=('seller_id','nunique'),total_freight=('freight_value','sum'),total_price=('price','sum'),total_weight_g=('product_weight_g','sum'),total_volume_cm3=('volume_cm3','sum'),category_count=('product_category_name','nunique'),seller_customer_distance_km=('distance_km','mean'),same_state=('same_state_item','mean')).reset_index()
    extra=item.groupby('order_id').agg(max_item_weight_g=('product_weight_g','max'),max_item_volume_cm3=('volume_cm3','max'),total_item_price=('price','sum'),shipping_deadline=('shipping_limit_date','min')).reset_index()
    out=orders.merge(extra,on='order_id',how='left').merge(base,on='order_id',how='left').merge(dynamic,on='order_id',how='left')
    purchase=out.order_purchase_timestamp;approval=out.order_approved_at
    out['purchase_month']=purchase.dt.month;out['purchase_year']=purchase.dt.year;out['purchase_weekday']=purchase.dt.dayofweek
    out['purchase_is_weekend']=(out.purchase_weekday>=5).astype(float);out['after_5pm']=(purchase.dt.hour>=17).astype(float)
    out['approval_weekday']=approval.dt.dayofweek;out['approval_hour']=approval.dt.hour;out['approval_month']=approval.dt.month
    out['estimated_delivery_days']=(out.order_estimated_delivery_date-purchase).dt.total_seconds()/86400
    out['estimated_delivery_slack']=(out.order_estimated_delivery_date-approval).dt.total_seconds()/86400
    out['shipping_deadline_slack_days']=(out.shipping_deadline-approval).dt.total_seconds()/86400
    # Extreme dates are treated as unknown, not as legitimate lead times.
    # Keep modest negative slack: approval may genuinely follow the deadline.
    deadline_bad=(out.shipping_deadline_slack_days < -30) | (out.shipping_deadline_slack_days > 365)
    out.loc[deadline_bad,'shipping_deadline_slack_days']=np.nan
    # Ratios are undefined when no positive delivery promise time remains.
    valid_slack=out.estimated_delivery_slack.where(out.estimated_delivery_slack>0)
    out['delivery_time_pressure']=out.delivery_time_pressure_components/valid_slack
    # Large but mathematically valid ratios are retained for diagnostics.
    out['freight_price_ratio']=out.total_freight/out.total_price.replace(0,np.nan)
    out['distance_x_weight']=out.seller_customer_distance_km*out.total_weight_g
    out['distance_x_volume']=out.seller_customer_distance_km*out.total_volume_cm3
    out['distance_per_delivery_day']=out.seller_customer_distance_km/out.estimated_delivery_days.replace(0,np.nan)
    out['late_delivery']=np.where(out.order_delivered_customer_date.notna()&out.order_estimated_delivery_date.notna(),(out.order_delivered_customer_date.dt.normalize()>out.order_estimated_delivery_date.dt.normalize()).astype(int),np.nan)
    # Inference must retain undelivered orders: their eventual label is unknown.
    out = out.loc[out.order_approved_at.notna() & out.item_count.notna()].copy()
    out = out.sort_values(["order_approved_at", "order_id"]).reset_index(drop=True)
    cols = ["order_id", "order_approved_at", "late_delivery"] + BASE + NEW + V3
    return out[cols].replace([np.inf, -np.inf], np.nan)
