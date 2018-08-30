## Dependencies
- Python 3.6+, Postgres, Unix (tested on Ubuntu)
- `pip install -r requirements.txt`
- Install hyperopt [from source](http://hyperopt.github.io/hyperopt/#installation)

## Setup
1. `createdb dl_price_mom`
1. `cp config.example.json config.json` then modify `config.json`
1. Sign into [Kaggle](https://www.kaggle.com), download [this dataset](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/home)
1. Unzip
    ```
    unzip price-volume-data-for-all-us-stocks-etfs.zip -d tmp
    pushd tmp
    unzip Data.zip
    popd
    ```

The important additions to your dir structure should be:

```
- config.json (modified)
- tmp
  - Stocks 
```

#### Import
`python import.py`

This will populate your database from the Kaggle dataset, it will take a few hours.

#### Note re: dataset
We have arrange the code to work with Kaggle dataset but there are few important provisions:
1. We did not use Kaggle data for our analysis. We used proprietary licensed data from FTSE Russell which we cannot provide by the terms of the agreement.
2. Kaggle does not specify the investment universe, which was sourced from FTSE Rusell database for the Russell 1000 based investment universe.

## Run
You'll train three separate components to completion (they depend on each other sequentially):
1. Autoencoder
2. EmbedClust
3. Recurrent Neural Network (GRU/FFN)

Each component will take a day or so; EmbedClust in particular takes multiple days. Run each step in a tmux session, check back in 24h. Between each step (after completion), you'll choose hyperparameter winners.

1. `python ae.py`
2. `python select_winners.py` - selects optimal hyperparameters for autoencoder.
3. `python embed_clust.py`
4. Manually select embed_clust clustering outputs for each origin (set `use` to `TRUE`), based on normalized X_B or S_Dbw scores.
5. `python rnn.py`

## Credit
- Deep Clustering with Convolutional Autoencoders (DCEC)
  - Code: [XifengGuo/DCEC](https://github.com/XifengGuo/DCEC)
  - Paper: [Unsupervised Deep Embedding for Clustering Analysis](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)
- S_Dbw
  - Code: [iphysresearch/S_Dbw_validity_index](https://github.com/iphysresearch/S_Dbw_validity_index)
  - Paper: [Clustering Validity Assessment: Finding the optimal partitioning of a data set ](https://pdfs.semanticscholar.org/dc44/df745fbf5794066557e52074d127b31248b2.pdf)
- Xie-Beni index (XB)
  -  Paper: [Improved Validation Index for Fuzzy Clustering ](http://folk.ntnu.no/skoge/prost/proceedings/acc05/PDFs/Papers/0203_WeB17_6.pdf)

