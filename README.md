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

This will populate your database from the Kaggle dataset, it will take some hours.


#### Note re: dataset
We have arrange the code to work with Kaggle dataset but there are few caveats:
1. We did not use Kaggle data for our analysis. We used proprietary licensed data from FTSE Russell we which can not offer by the terms of the agreement.
1. Kaggle does not specify the investment universe, which we sourced from FTSE Rusell database for Russell 1000 data.

## Run
You'll train three separate components to completion (they depend on each other sequentially):
1. Autoencoder
1. EmbedClust
1. Recurrent Neural Network

Each component will take a day or so; EmbedClust in particular takes many days. Run each step in a tmux session, check back in 24h. Between each step (after completion), you'll choose hyperparameter winners.

1. `python ae.py`
1. `python select_winners.py`
1. `python embed_clust.py`
1. Manually select some embed_clust winners (set `use` to `TRUE`), based on XB or S_Dbw scores.
1. `python rnn.py`

## Credit
- Deep Clustering with Convolutional Autoencoders (DCEC)
  - Code: [XifengGuo/DCEC](https://github.com/XifengGuo/DCEC)
  - Paper: [Unsupervised Deep Embedding for Clustering Analysis](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)
- S_Dbw
  - Code: [iphysresearch/S_Dbw_validity_index](https://github.com/iphysresearch/S_Dbw_validity_index)
  - Paper: [Clustering Validity Assessment: Finding the optimal partitioning of a data set ](https://pdfs.semanticscholar.org/dc44/df745fbf5794066557e52074d127b31248b2.pdf)
- Xie-Beni index (XB)
  -  Paper: [Improved Validation Index for Fuzzy Clustering ](http://folk.ntnu.no/skoge/prost/proceedings/acc05/PDFs/Papers/0203_WeB17_6.pdf)

