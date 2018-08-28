import os, pdb
import pandas as pd
import numpy as np
from utils import engine, thread
from sqlalchemy.dialects import postgresql as psql

SPAN = {'min': 475, 'of': 500}

with engine.connect() as conn:
    conn.execute('drop table if exists eod, eom, clusters')


def threaded(filename):
    try:
        days = pd.read_csv(f'tmp/Stocks/{filename}')
    except:
        return  # No rows, corrupt, whatever (there's a few)
    if days.shape[0] < SPAN['of']:
        return  # Require minimum n_rows

    # Cleanup
    # --------
    ticker = filename.split('.')[0]
    days.columns = map(str.lower, days.columns)
    days = days[['date', 'close']].rename(columns={'close': 'price'})  # we don't need OHLCV for this project
    days['date'] = pd.to_datetime(days.date)
    days = days.set_index('date').sort_index()

    # Impute, add change, determine eligibility
    # --------
    # Expand to "all days" ('D'), mark imputed & ffill(). Then downsample back to "business days" ('B')
    days = days.resample('D').asfreq()
    days['imputed'] = days.price.isnull()
    days = days.ffill()  # days.interpolate() ?
    days = days.resample('B').asfreq()

    change = days.price.pct_change()
    change[0] = 0  # first NaN pct_change
    days['change'] = change
    days['has_enough'] = (days.price > 1 & ~days.imputed)\
        .astype(int).rolling(SPAN['of'], closed='left').sum() >= SPAN['min']

    months = []
    clusters = []
    for month, group in days.groupby(pd.Grouper(freq='BM')):
        has_enough = days.loc[month.strftime('%Y-%m')].iloc[-1].has_enough  # last_day.has_enough
        next_month = month + 1
        if next_month in days.index:
            next_month = days.loc[(month + 1).strftime('%Y-%m')]
            mtd_1mf = (next_month.price.iloc[-1] / next_month.price.iloc[0]) - 1
        else:
            next_month = None
            mtd_1mf = (group.price.iloc[-1] / group.price.iloc[0]) - 1
        imputed = next_month is None

        # EOM summaries
        # --------
        month_data = dict(
            date=month,
            imputed=imputed,
            mtd_1mf=mtd_1mf
        )
        months.append(month_data)

        # Clusters
        # --------
        missing_days = group.imputed.astype(int).sum()
        v = group.change.values
        if len(v) > 22:
            v = np.concatenate([[v[0:2].mean()], v[2:]])
        elif len(v) < 22:
            v = np.concatenate([np.zeros(22 - len(v)), v])
        assert len(v) == 22
        clusters.append(dict(
            date=month,
            vals=v.tolist(),
            has_enough=has_enough,
            missing_days=missing_days,
            imputed=imputed,
            mtd_1mf=mtd_1mf
        ))
    months = pd.DataFrame(months).set_index('date').sort_index()
    clusters = pd.DataFrame(clusters).set_index('date').sort_index()

    # Determine clusters eligibility (need to collect them all first, above)
    # ------
    eligible = clusters.has_enough & (clusters.missing_days.rolling(window=12).min() < 3)
    clusters['eligible_12m'] = eligible
    # "reverse eligibility", or look_12m: is anything in my 12-future eligible?
    # two tricks here: reverse-check-reverse. [::-1] means reverse; max() checks for "any are true"?
    # Also bool(nan) = True
    clusters['look_12m'] = eligible[::-1].rolling(window=12).max()[::-1].fillna(False).astype(bool)

    # Only needed has_enough for intermediate
    del days['has_enough'], months['has_enough'], clusters['has_enough']

    # Save to DB
    # --------
    for table, df in [('eod', days), ('eom', months), ('clusters', clusters)]:
        dtype = {'date': psql.DATE}
        if table == 'clusters':
            dtype['vals'] = psql.ARRAY(psql.DOUBLE_PRECISION)
        df['ticker'] = ticker
        idx = ['date', 'ticker']
        # df = df['1995-01-01':'2017-12-31']
        df = df.reset_index().set_index(idx)
        with engine.connect() as conn:
            df.to_sql(table, conn, if_exists='append', index_label=idx, dtype=dtype)


thread(threaded, os.listdir('tmp/Stocks'))
