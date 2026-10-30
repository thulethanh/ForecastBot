"""Read-only dimension catalog; no forecast data or scoring changes."""
import pandas as pd
from sqlalchemy import text
from forecastbot.db.postgres import get_dataset_engine

FIELDS={'M5':['state_id','store_id','item_id'],
        'HCMC_BUS':['route_id','operator','vehicle_group']}


def load_catalog(dataset):
    engine=get_dataset_engine(dataset)
    if dataset=='M5':
        sql='''SELECT DISTINCT state_id, store_id, item_id
               FROM m5_sales_train WHERE d = (SELECT MIN(d) FROM m5_sales_train)''' 
    elif dataset=='HCMC_BUS':
        sql='''SELECT DISTINCT route_id, operator, vehicle_group
               FROM daily_passenger_stats'''
    else: raise ValueError('Unsupported dataset')
    with engine.connect() as conn, conn.begin():
        conn.execute(text('SET TRANSACTION READ ONLY'))
        conn.execute(text("SET LOCAL statement_timeout = '8000ms'"))
        frame=pd.read_sql(text(sql),conn)
    return frame.fillna('').astype(str).apply(lambda col:col.str.strip())


def choices(frame,field,selected):
    subset=frame
    for k,v in selected.items():
        if v: subset=subset[subset[k].eq(v)]
    return sorted(v for v in subset[field].unique().tolist() if v)
