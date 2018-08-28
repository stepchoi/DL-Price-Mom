import pdb, os
import pandas as pd
import numpy as np
from box import Box
from utils import engine, thread, bcolors, tmp_path, pickle_load, pickle_dump
np.random.seed(42)  # ensure consistent results
import pandas.tseries.offsets as offsets


def concat_x_cols(df, X):
    """
    unpack vals from col of lists to their own cols
    Everything that's not 'x' is 'y' (hierarchical index). This stores columns like `mtd_1mf`, `bin_up_down`, etc.
    """
    X = pd.DataFrame(X, index=df.index)
    cols = [
        ['y']*df.shape[1] + ['x']*X.shape[1],
        df.columns.tolist() + list(range(X.shape[1]))
    ]
    df = df.join(X)
    df.columns = pd.MultiIndex.from_arrays(cols)
    return df


def get_clusters(origin):
    year = int(str(origin)[:4])
    with engine.connect() as conn:
        sql = f"select 1 from clusters where date_part('year', date)={year}"
        res = conn.execute(sql).fetchone()
        if res is None: return None, None

        # offset = BusinessMonthEnd()
        offset = offsets.MonthEnd()
        end = offset.rollforward(origin.to_timestamp()).strftime('%Y-%m-%d')

        sql = f"""
        select ticker, date, mtd_1mf, vals from clusters 
        where look_12m=TRUE and date <= '{end}'
        order by date asc -- if not for sequencing, something to ensure consistent (same) clusters used each time  
        """
        df = pd.read_sql(sql, conn, index_col=['date', 'ticker']).sort_index()

        # unpack vals from col of lists to their own cols
        X = [x for x in df.vals.values]
        df = concat_x_cols(df.drop('vals', 1), X)

    df['y', 'mtd_1mf'] = df.y.mtd_1mf
    return df


def argmax_matrix(mat):
    maxes = np.argmax(mat, 1)
    ohc = np.zeros_like(mat)
    ohc[np.arange(len(ohc)), maxes] = 1
    return ohc


def clusters2np(q, origin, reset=False):
    os.makedirs(tmp_path('embed_clust/clusters2np'), exist_ok=True)
    f = tmp_path(f'embed_clust/clusters2np/{origin}.pkl')
    if os.path.exists(f) and not reset:
        return pickle_load(f)

    # tmp: drop MultiIndex, then add back @end. Pandas has trouble w/ date-slicing MultiIndexes.
    # https://stackoverflow.com/questions/36621778/pandas-dataframe-datetime-slicing-with-index-vs-multiindex
    # q.index = q.index.set_levels(q.index.levels[0].to_period('M'), 0)  # this approach too slow
    q = q.reset_index().set_index('date')
    q.index = q.index.to_period('M')

    months = Box(
        train=q.loc['1995-12' : origin - 1].index.unique(),
        test=q.loc[origin].index.unique()
    )
    data = Box(
        train=[],
        test=[],
        train_start=months.train[0], train_end=months.train[-1],
        test_start=months.test[0], test_end=months.test[-1]
    )
    for set_, months_ in months.items():
        skipped = []
        ds = data[set_]
        def threaded(m):
            chunk = q.loc[m-12:m]
            for ticker, group in chunk.groupby('ticker'):
                if group.shape[0] != 13:
                    # skipped.append(ticker)
                    skipped.append(group.shape[0])  # FIXME find solution for this
                    continue

                # One-hot encode the arg-maxes
                vals = group.x.values
                ohc = argmax_matrix(vals)

                ds.append(pd.DataFrame({
                    'date': m,
                    'ticker': ticker,
                    'x': [ohc[:-1]],
                    'y': [ohc[-1]],
                    'mtd_1mf': group.y.mtd_1mf[-1]
                }))
        thread(threaded, months_)

        data[set_] = pd.concat(ds).set_index(['date', 'ticker'])
        ds = data[set_]  # re-assign

        # Tack on tercile in case we want that as our target instead of cluster
        ds['tercile'] = None
        for m, group in ds.groupby(level=0):
            ds.loc[m, 'tercile'] = pd.qcut(group.mtd_1mf, 3, labels=False)

        # Warn of data skipped
        msg = f"{set_}: {len(ds.x)} kept, {len(skipped)} skipped"
        if len(ds.x) < len(skipped): msg = f"{bcolors.WARNING}{msg}{bcolors.ENDC}"
        print(msg)
        # print(pd.Series(skipped).value_counts())
    pickle_dump(data, f)
    return data
