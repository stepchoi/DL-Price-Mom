"""
ae.py is does not completely converge - it will show slightly different answers (mostly due to the fact that the test_set will be different each run) but it is fairly stable
meaning that certain architectures such as roughly num_layers or learning rates will be _similar_ but we want to capture these tendencies, not focus on _*BEST*_
I think given the MSE - which are now more meaningful - the find two is important. We have a cutoff of .000275 and for each origin we want the lowest  z_dim that has at least two similar examples having low MSE reconstruction loss. This means that we need at least two similar hypers to confirm that it is a good one.
I think z_dim will be 14 or 15 and by running two different ae's, the num_layers differ each origin and the best can be different but are fairly similar within origin (even across both runs).
"""
import pdb
from utils import engine
import pandas as pd
from sqlalchemy import text


with engine.connect() as conn:
    # -- AE --
    # 1) using z_dim =14, 15, 16 find lowest z_dim with MSE threshold below 0.000275
    sql = f"""
    select id, hypers, mse, origin from ae
    where (hypers->>'z_dim')::int between 14 and 16 
    and mse < 0.000225
    """
    df = pd.read_sql(sql, conn)
    if df.shape[0] == 0:
        raise Exception("No data found (likely MSE criteria too low for current runs)")
    # expand out the hypers col of dicts into individual columns
    df = pd.concat([df, df.hypers.apply(pd.Series)], 1)
    df['n_layers'] = df.layers.apply(lambda l: l['n'])

    ids = []
    for origin, origin_group in df.groupby('origin'):
        # 2) look for at hyper set that has at least TWO examples of same num_layers
        # 3) if there are only one example of z_dim below threshold (e.g.z_dim = 14 with only one hyperset below 0.000275) then we ignore and use the best of z_dim =15 with best two num_layers
        valid_rows = []
        for idx, hypers_group in origin_group.groupby(['z_dim', 'n_layers'], sort=True):
            if hypers_group.shape[0] >= 2:
                valid_rows.append(hypers_group)
        ids.append(pd.concat(valid_rows).sort_values(['z_dim','mse']).iloc[0].id )
    
    if len(ids) == 0: raise Exception("No AE zdim/layer combos identified as winners")
    conn.execute(f"update ae set use=false")
    conn.execute(text(f"update ae set use=true where id in :ids"), ids=tuple(ids))