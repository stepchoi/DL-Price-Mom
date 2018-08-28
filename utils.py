import json, os, pickle, sys
from os import path
from time import sleep
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm


THREADS = cpu_count()
config_json = json.load(open(os.path.join(path.dirname(__file__), 'config.json')))
engine = create_engine(config_json['DB_URL'], pool_size=cpu_count(), max_overflow=-1)


def noise_idxs(vals):
    # Noise = summation of q*ln(q)) that is lower than ln(2) = 0.7
    # noise = `(vals * np.log(vals)).sum(1) < np.log(0.5)` OR <below>; log(1/n) = -log(n) ... roughly
    return -(vals * np.log(vals)).sum(1) > np.log(100)


def geometric(arr):
    mul = 1
    for x in arr: mul *= (1 + x/100)
    return (mul - 1)*100


def thread(fn, arr, threads=THREADS):
    with ThreadPool(threads) as pool:
        itr = pool.imap(fn, arr)
        list(tqdm(itr, total=len(arr)))


def pickle_load(f):
    if not os.path.exists(f):
        return None
    with open(f, 'rb') as f:
        return pickle.load(f)


def pickle_dump(data, f):
    with open(f, 'wb') as f:
        pickle.dump(data, f)


# https://stackoverflow.com/a/287944/362790
# colors in CLI print()
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def path_join(cur_file, *args):
    return path.abspath(path.join(path.dirname(cur_file), *args))


def tmp_path(path_):
    return path_join(__file__, 'tmp', path_)


def common_args(parser, keys):
    lookup = {
        'reset': dict(action='store_true', help='Reset results?'),
        'run-num': dict(type=int, default=1, help='Which run_num'),
        'origin': dict(nargs='*', help='List of origins to cover, eg --origin 2003 2004 2005'),
        'pickup': dict(action='store_true', help='Pick up next origin where you left off?'),
    }
    for k in keys:
        parser.add_argument(f'--{k}', **lookup[k])


def generate_origins(args_origin, n_runs=3):
    origins = None
    if not args_origin:
        origins = pd.period_range('2002-12', periods=180, freq='M').tolist()
    else:
        origins = [
            pd.period_range(f"{int(o) - 1}-12", periods=12, freq='M').tolist()
            for o in args_origin
        ]
        # flatten
        origins = [item for sublist in origins for item in sublist]

    n_origins = len(origins)
    return (origins * n_runs), n_origins, 0


# TODO remove before submission (origins sanity-check)
def check_origin_max_date_match(origin, df):
    max_date = str(df.index.get_level_values(0)[-1])[:7]
    origin = str(origin)
    if origin > max_date:
        print(f"x {bcolors.FAIL}origin={origin} max_date={max_date}{bcolors.ENDC}")
        return False
    else:
        print(f"√ {bcolors.OKGREEN}origin={origin} max_date={max_date}{bcolors.ENDC}")
        return True
