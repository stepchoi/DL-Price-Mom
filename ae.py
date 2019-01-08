'''
Keras implementation of deep embedder to improve clustering, inspired by:
"Unsupervised Deep Embedding for Clustering Analysis" (Xie et al, ICML 2016)

Definition can accept somewhat custom neural networks. Defaults are from paper.
'''
import split_gpu

import uuid
import numpy as np
import pandas as pd
from utils import engine, common_args, generate_origins, bcolors, check_origin_max_date_match
from data import get_clusters
from sqlalchemy.dialects import postgresql
from keras import backend as K
from keras import callbacks
from keras.models import Model
from keras.optimizers import Adam
from keras import layers as L
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


TEST_SIZE = .30
EARLY_STOPPING = callbacks.EarlyStopping()
REDUCE_LR_PLATEAU = callbacks.ReduceLROnPlateau()
HYPEROPT_EVALS = 50


class AE(object):
    def __init__(self, input_dim=22):
        self.input_dim = input_dim
        self.autoencoder = None
        self.encoder = None

    @staticmethod
    def best_hypers(origin):
        with engine.connect() as conn:
            sql = f"select hypers from ae where origin='{origin}' and use=true limit 1"
            h = conn.execute(sql).fetchone()
            if h is None:
                print("No ae.use for this forecast origin; go into database and check-box some `use` column")
                return None
            return h.hypers

    def compile(self, hypers):
        self.hypers = hypers
        self.batch_size = 2**int(hypers['batch_size'])
        z_dim = int(hypers['z_dim'])
        self.z_dim = z_dim
        input_dim = self.input_dim

        # see https://blog.keras.io/building-autoencoders-in-keras.html for functional API autoencoder example
        input = L.Input(shape=(input_dim,))

        layers = hypers['layers']
        n = int(layers['n'])
        if n == -1:
            self.autoencoder = self.encoder = self.decoder = L.Lambda(lambda x: x)(input)
            return

        dims = [int(layers[f'{n}-{i}']) for i in range(n)]

        def dense(model, d, last=False, extra={}):
            act = 'linear' if last else 'tanh'
            init = {
                'tanh': 'glorot_normal',
                'linear': 'glorot_uniform',
            }[act]
            args_ = {'kernel_initializer': init}
            model = L.Dense(d, **args_, **extra)(model)
            if not last:
                model = L.Activation(act)(model)
            return model

        if len(dims) == 0:
            encoded = dense(input, z_dim, extra={'name': 'embedding'})
            decoded = dense(encoded, input_dim, last=True)
        else:
            encoded = dense(input, dims[0])
            for d in dims[1:]: encoded = dense(encoded, d)
            encoded = dense(encoded, z_dim, extra={'name': 'embedding'})
            decoded = encoded
            for i, d in enumerate(dims[::-1]):
                extra = {'name': 'decoder_start'} if i == 0 else {}
                decoded = dense(decoded, d, extra=extra)
            decoded = dense(decoded, input_dim, last=True)

        autoencoder = Model(input, decoded)
        encoder = Model(input, encoded)

        # Decoder a tad more complex
        encoded_input = L.Input(shape=(z_dim,))
        decoder, building_decoder = encoded_input, False
        for l in autoencoder.layers:
            if l.name == 'decoder_start': building_decoder = True
            if building_decoder: decoder = l(decoder)
        decoder = Model(encoded_input, decoder)

        autoencoder.compile(optimizer=Adam(lr=10 ** -hypers['lr']), loss='mse')
        self.autoencoder, self.encoder, self.decoder = autoencoder, encoder, decoder

    def train(self, X):
        X_train, X_test = train_test_split(X, test_size=TEST_SIZE)
        res = self.autoencoder.fit(
            X_train, X_train,
            epochs=500,  # the callbacks will stop the model
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(X_test, X_test),
            verbose=0,
            callbacks=[REDUCE_LR_PLATEAU, EARLY_STOPPING]
        )
        return res.history['val_loss'][-1]

def run_model(X, z_dim, origin):
    i = 0
    def inner_fn(hypers):
        nonlocal i
        hypers['z_dim'] = z_dim
        K.clear_session()  # Important! Else code hangs
        ae = AE()
        ae.compile(hypers)
        mse = ae.train(X)
        id = uuid.uuid4()
        with engine.connect() as conn:
            dtype = {'hypers': postgresql.JSONB}
            df = pd.DataFrame([{
                'itr': i,
                'id': id,
                'mse': mse,
                'hypers': hypers,
                'use': False,
                'origin': str(origin)
            }]).set_index('id')
            df.to_sql('ae', conn, if_exists='append', index_label='id', dtype=dtype)
        i += 1
        return mse
    return inner_fn


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    common_args(parser, ['reset', 'origin', 'pickup'])
    args = parser.parse_args()

    if args.reset:
        with engine.connect() as c:
            c.execute("drop table if exists ae")

    origins, n_origins, n_origins_done = generate_origins(args.origin, 3)
    args.origin = origins.pop(0)

    while True:
        # Pick up where left off
        with engine.connect() as conn:
            sql = f"select count(*) as ct from ae where origin='{args.origin}'"
            if args.pickup and conn.execute(sql).fetchone().ct >= HYPEROPT_EVALS:
                print(f'{bcolors.WARNING}skip origin={args.origin}{bcolors.ENDC}')
                args.origin = origins.pop(0)
                continue

        # print(f"{bcolors.OKBLUE}origin={args.origin} {bcolors.ENDC}")

        # Defining space variables needs to be _inside_ this for-loop. HyperOpt does something weird where it
        # marks spaces as "complete" or something, and every iteration past the first z_dim just skips; so this way we
        # re-init the space variables
        clust = get_clusters(origin=args.origin)
        if clust is None: break
        X = clust.x.values

        check_origin_max_date_match(args.origin, clust)

        for z_dim in [14, 15, 16]:
            def unit_space(n):
                obj = {'n': n}
                for i in range(n):
                    obj[f'{n}-{i}'] = hp.quniform(f'{n}-{i}', 15, 80, 5)
                return obj

            # old note: fix: these are independent, give TPE shared knowledge of the layers + their best units somehow
            # new thinking: no, a 1-layer network will want different #units for its 1st (and only) layer than a 3-layer net.
            # Keep it as-is (hp.choice)
            layer_space = hp.choice('layers', [unit_space(i) for i in [1, 2, 3]])

            space = {
                'lr': hp.uniform('lr', 1.5, 6),  # => 1e-x
                'batch_size':  hp.quniform('batch_size', 5, 13, 1),  # => 2**x
                'layers': layer_space,
                # 'act': hp.choice('act', ['relu', 'tanh'])
            }

            trials = Trials()
            fmin(run_model(X, z_dim, args.origin), space=space, algo=tpe.suggest, max_evals=HYPEROPT_EVALS, trials=trials)

        n_origins_done += 1
        if n_origins_done == n_origins:
            args.run_num += 1
        args.origin = origins.pop(0)
