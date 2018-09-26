import split_gpu

import pdb, uuid, pickle
from keras.models import Sequential
from keras.layers import Dense, GRU
import keras.backend as K
from keras import callbacks, optimizers
import numpy as np
import pandas as pd
from utils import engine, common_args, bcolors, generate_origins, check_origin_max_date_match
from data import clusters2np
from sqlalchemy.dialects import postgresql as psql
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope


EARLY_STOPPING = callbacks.EarlyStopping(min_delta=.0001, patience=5)
REDUCE_LR_PLATEAU = callbacks.ReduceLROnPlateau(patience=2)
HYPEROPT_EVALS = 100


def col2np(col):
    # Unpack a dataframe column of np.arrays into the np.array version
    # https://stackoverflow.com/a/45548507/362790
    return np.array(col.values.tolist()).reshape((-1, *col.iloc[0].shape))


class RNN(object):
    def __init__(self, args, data):
        self.id = uuid.uuid4()
        self.args = args
        self.data = data

    def compile(self, hypers):
        self.hypers = hypers
        timesteps = 12
        n_clust = q.shape[1] - 1

        # See https://keras.io/getting-started/sequential-model-guide/#getting-started-with-the-keras-sequential-model
        # section 'Same stacked LSTM model, rendered "stateful"' for chaining sequences over long time horizon

        # expected input data shape: (batch_size, timesteps, data_dim)
        model = Sequential()
        layers = hypers['layers']
        d_n = hypers['d_layers']
        n = int(layers['n'])

        for i in range(n):
            extra = dict(return_sequences=True)
            if i == 0: extra.update(input_shape=(timesteps, n_clust))
            if i == n - 1: del extra['return_sequences']
            model.add(GRU(layers[f'{n}-{i}'], **extra))
        for j in range(d_n):
            model.add(Dense(n_clust, activation='tanh'))

        target_dim = args.bins
        model.add(Dense(target_dim, activation='softmax'))

        adam = optimizers.Adam(lr=10 ** -hypers['lr'])
        model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self):
        batch_size = 2 ** int(self.hypers['batch_size'])
        d = self.data.train
        x = col2np(d.x)
        for m, group in d.groupby(level=0):
            d.loc[m, 'quant'] = pd.qcut(group.mtd_1mf, args.bins, labels=False)
            
        y = pd.get_dummies(d.quant).values
        history = self.model.fit(
            x, y,
            validation_split=.3,
            batch_size=batch_size,
            epochs=1000,
            callbacks=[EARLY_STOPPING, REDUCE_LR_PLATEAU]
        ).history
        return history['val_loss'][-1], history['val_acc'][-1]

    def test(self):
        d = self.data.test
        d['quant_T'] = pd.qcut(d.mtd_1mf, args.bins, labels=False)
        loss_test, acc_test = self.model.evaluate(col2np(d.x), pd.get_dummies(d.quant_T))

        pct_covered = [0.] * args.bins
        preds = self.model.predict(col2np(d.x)).argmax(1)
        df = pd.DataFrame({
            'quant': preds,
            'mtd_1mf': d.mtd_1mf
        })
        #df['ranked'] = df.quant.argsort()
        df['binned'] = pd.cut(df.quant, args.bins, labels=False)
        quants = df.groupby('binned', sort=True)
        p = quants.mtd_1mf.mean()
        p = [p[i] if i in p else 0 for i in range(args.bins)]  # fill in missing holes with 0

        # p = p.sort_index()  # handled oin groupby(sort=True) above?
        pct_covered = df.groupby('binned').size()/df.shape[0]
        hml = p[args.bins - 1] - p[0]
        bnh = d.mtd_1mf.mean()

        return {
            'loss_test': loss_test,
            'acc_test': acc_test,
            'hml': hml,  # high minus low
            'bnh': bnh,  # buy and hold
            'pct_cov': pct_covered.tolist(),
            **{f'p{i}': v for i, v in enumerate(p)}
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='tercile', help='(tercile|quintile|decile)')
    parser.add_argument('--reset-pkl', action='store_true', help='Reset only the clusters2np pkl files (--reset alone will do this too)')
    common_args(parser, ['reset', 'origin', 'pickup'])

    args = parser.parse_args()

    if args.reset:
        with engine.connect() as conn:
            conn.execute(f'drop table if exists rnn_{args.target}')

    args.bins = {'quintile': 5, 'tercile': 3, 'decile': 10}[args.target]

    origins, n_origins, n_origins_done = generate_origins(args.origin, 3)
    args.origin = origins.pop(0)

    while True:
        with engine.connect() as conn:
            # Pick up where you left off.
            sql = f"select count(*) as ct from rnn_{args.target} where origin='{args.origin}'"
            if args.pickup and conn.execute(sql).fetchone().ct >= HYPEROPT_EVALS:
                print(f'{bcolors.WARNING}skip origin={args.origin}{bcolors.ENDC}')
                args.origin = origins.pop(0)
                continue

            sql = f"select id from embed_clust where origin='{args.origin}' and use=true limit 1"
            embed_clust = conn.execute(sql).fetchone()
            if embed_clust is None:
                raise Exception(f"No embed_clust row for origin={args.origin} with `use` column checked.")
            sql = f"select q from embed_clust_q where id='{embed_clust.id}' limit 1"
            embed_clust = conn.execute(sql).fetchone()
            if embed_clust is None:
                raise Exception(f"No full clusters for this origin, run embed_clust.py with --origin {args.origin}")
            q = pickle.loads(embed_clust.q)

            check_origin_max_date_match(args.origin, q)

        data = clusters2np(q, args.origin, reset=(args.reset or args.reset_pkl))


        print(f"{bcolors.OKBLUE}Train: {data.train_start}-{data.train_end}")
        print(f"Test: {data.test_start}-{data.test_end}{bcolors.ENDC}")

        def run_model(hypers):
            K.clear_session()

            rnn = RNN(args, data)
            rnn.compile(hypers)
            loss_val, acc_val = rnn.train()
            res = rnn.test()
            with engine.connect() as conn:
                dtype = {'hypers': psql.JSONB}
                df = pd.DataFrame([{
                    'id': rnn.id,
                    'origin': str(args.origin),
                    'hypers': hypers,
                    'loss_val': loss_val,
                    'acc_val': acc_val,
                    **res
                }]).set_index('id')
                df.to_sql(f'rnn_{args.target}', conn, index_label='id', if_exists='append', dtype=dtype)

            return acc_val

        def unit_space(n):
            obj = {'n': n}
            for i in range(n):
                obj[f'{n}-{i}'] = scope.int(hp.quniform(f'{n}-{i}', 10, 65, 5))
            return obj
        layer_space = hp.choice('layers', [unit_space(i) for i in [2, 3, 4, 5]])
        space = {
            'lr': hp.quniform('lr', 1.5, 5, .01),  # => 1e-x
            # 'opt': hp.choice('opt', ['adam', 'rmsprop']),  # winner=adam
            'batch_size': scope.int(hp.quniform('batch_size', 6, 11, 1)),  # => 2**x
            'layers': layer_space,
            'd_layers': hp.choice('d_layers', [1, 2, 3]),
        }
        trials = Trials()
        best = fmin(run_model, space=space, algo=tpe.suggest, max_evals=HYPEROPT_EVALS, trials=trials)

        n_origins_done += 1
        if n_origins_done == n_origins:
            args.run_num += 1
        args.origin = origins.pop(0)
