"""
Addapted from Xifeng Guo. 2017.1.30 (https://github.com/XifengGuo/DEC-keras)
Keras implementation for Deep Embedded Clustering (DEC) algorithm:
        Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.
"""

import pdb, uuid, pickle
import pandas as pd
from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from box import Box
from sqlalchemy.dialects import postgresql as psql

from utils import engine, common_args, bcolors, noise_idxs, generate_origins, check_origin_max_date_match
from data import get_clusters, concat_x_cols
from ae import AE
from S_Dbw import S_Dbw


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_spec = InputSpec(ndim=2)
        self.reason_stop = 'unknown'

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EmbedClust(object):
    def __init__(self, ae, args, n_clusters=10, alpha=1.0):
        super(EmbedClust, self).__init__()
        self.id = uuid.uuid4()  # used for ref'ing in database, connecting to s_dbw, etc
        self.args = args
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.ae = ae
        self.reason_stop = 'unknown'
        self.noise_pct = 0

        # prepare EmbedClust model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(ae.encoder.output)
        self.model = Model(inputs=self.ae.encoder.input, outputs=[clustering_layer, ae.autoencoder.output])

    def extract_features(self, x):
        return self.ae.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, tol=1e-3):
        args = self.args
        z_dim, batch_size = self.ae.z_dim, self.ae.batch_size

        # max_iter should be a function of num_clust as well as sample size 
        maxiter = self.n_clusters * 2 * max(1e5, x.shape[0])
        maxiter = int(maxiter)

        update_interval = int(x.shape[0] / batch_size) * 2  # 2 epochs
        save_interval = int(maxiter / 8)

        esp_delta = 5

        if args.wCVI:
            maxiter = maxiter * 2
            update_interval = update_interval * 2
            save_interval = int(maxiter / 5)

        print('Update interval', update_interval)
        print('Save interval', save_interval)

        early_stop_start_after = save_interval

        assert save_interval >= update_interval

        # Step 1: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.z = self.ae.encoder.predict(x)
        y_pred = kmeans.fit_predict(self.z)
        y_pred_last = np.copy(y_pred)
        self.cluster_centers = kmeans.cluster_centers_
        self.model.get_layer(name='clustering').set_weights([self.cluster_centers])

        # Step 2: deep clustering
        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            # Early stops
            if ite == maxiter-1:
                self.reason_stop = 'max iterations'
                break

            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                self.q, self.p = q, p

                # evaluate the clustering performance
                y_pred = q.argmax(1)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)

                if ite > early_stop_start_after and delta_label < tol:
                    if esp_delta == 0:
                        print('delta_label ', delta_label, '< tol ', tol)
                        self.reason_stop = 'delta trigger'
                        print('Reached tolerance threshold. Stopping training.')
                        break
                    esp_delta -= 1

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            self.model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        # Add noise cluster
        noise = noise_idxs(self.q)
        q_w_noise = np.c_[self.q, np.zeros(self.q.shape[0])]
        q_w_noise[noise], q_w_noise[noise, -1] = 0., 1.
        self.q_w_noise = q_w_noise

        # Save metrics to DB
        with engine.connect() as conn:
            dtype = {
                'origin': psql.VARCHAR(64),
                's_dbw': psql.DOUBLE_PRECISION,  # interpreted as int if first val is nan
            }
            df = pd.DataFrame([dict(
                id=self.id,
                run_num=args.run_num,
                origin=str(args.origin),

                z_dim=z_dim,
                n_clusters=self.n_clusters,
                last_itr=ite,
                stop_reason=self.reason_stop,

                delta_label=delta_label,
                s_dbw=self.s_dbw(),

                use=False
            )]).set_index('id')
            df.to_sql('embed_clust', conn, index_label='id', if_exists='append', dtype=dtype)

        return y_pred

    # ----- CVI methods --------
    # --------------------------

    def s_dbw(self):
        """
        Calculate S_Dbw using iphysresearch/S_Dbw_validity_index (separate file); we need to massage our data
        to pass in a format that project expects
        """
        # Remove noise rows from data (create filtered sets, *_fs)
        noise = noise_idxs(self.q)
        z_fs, q_fs = self.z[~noise], self.q[~noise]
        self.noise_pct = len(self.z[noise])/len(self.z)
        if z_fs.shape[0] == 0: return 0.  # assert z_fs.shape[0] > 0, "Everything was noise"

        # Vector of cluster assignments. Numerical, starting with 0. Same row length as X
        data_cluster = q_fs.argmax(1)

        # Cluster centroids
        cluster_centers = self.cluster_centers

        # Scale x anc cluster centroids for S_Dbw 
        z_fs = np.r_[z_fs, cluster_centers]
        z_fs = scale(z_fs)
        nc = cluster_centers.shape[0]
        z_fs, cluster_centers = z_fs[:-nc], z_fs[-nc:]

        s_dbw = S_Dbw(z_fs, data_cluster, cluster_centers)
        return s_dbw.S_Dbw_result()


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--maxiter', default=1e6, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--nc', nargs='*', help='List of n_clusters to cover, eg --nc 9 10 11')
    parser.add_argument('--wCVI', action='store_true', help='Finds and redo the weak CVI origins')
    common_args(parser, ['reset', 'run-num', 'origin', 'pickup'])
    args = parser.parse_args()

    origins, n_origins, n_origins_done = generate_origins(args.origin, 3)
    args.origin = origins.pop(0)

    args.nc = args.nc or list(range(6,20))
    nc_range = [int(nc) for nc in args.nc]
    if args.reset:
        with engine.connect() as conn:
            conn.execute("drop table if exists embed_clust, embed_clust_q")

    while True:
        if args.wCVI:
            while True:
                cont_loop = False
                with engine.connect() as conn:
                    sql = f"select s_dbw from embed_clust where origin='{args.origin}'"
                    temp_CVI = conn.execute(sql).fetchall()

                for sdbw in temp_CVI:
                    if sdbw < 0.75:
                        print(f'{bcolors.WARNING}skip origin={args.origin} {bcolors.ENDC}')
                        n_origins_done += 1
                        args.origin = origins.pop(0)
                        cont_loop = True
                        break
                if cont_loop: continue
                break

        clusters = get_clusters(args.origin)
        x = clusters.x.values

        check_origin_max_date_match(args.origin, clusters)

        for nc in nc_range:
            # Pick up where you left off. EmbedClust is very computationally expensive, so you don't want to
            # re-run unnecessarily. Also allows parallization (since other nodes will pick up unfinished origins)
            with engine.connect() as conn:
                sql = f"select 1 from embed_clust where origin='{args.origin}' and n_clusters={nc} limit 1"
                if args.pickup and conn.execute(sql).fetchone():
                    print(f'{bcolors.WARNING}skip origin={args.origin} nc={nc}{bcolors.ENDC}')
                    continue
            print(f"{bcolors.OKBLUE}origin={args.origin} nc={nc}{bcolors.ENDC}")

            K.clear_session()  # hyperopt creates many graphs, will max memory fast if not cleared
            ae = AE()
            hypers = AE.best_hypers(args.origin)
            if hypers is None:
                print("No embed_clust.use for this forecast origin; go into database and check-box some `use` column")
                break
            ae.compile(hypers)
            embed_clust = EmbedClust(ae, args, nc)

            print('...Pretraining...')
            embed_clust.ae.train(x)

            embed_clust.model.summary()
            embed_clust.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam')
            y_pred = embed_clust.fit(x, tol=args.tol)

            # Save for use by RNN. See https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch08s08.html
            q_ = concat_x_cols(clusters.y, embed_clust.q_w_noise)
            dtype = {'q': psql.BYTEA, 'cluster_prices': psql.BYTEA}
            df = pd.DataFrame([{
                'id': embed_clust.id,
                'q': pickle.dumps(q_),
            }]).set_index('id')
            with engine.connect() as conn:
                df.to_sql('embed_clust_q', conn, if_exists='append', index_label='id', dtype=dtype)

        n_origins_done += 1
        if n_origins_done == n_origins:
            args.run_num += 1
        args.origin = origins.pop(0)

