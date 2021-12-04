# Conclusions
The following research aimed to roughly follow the outline of previous innovation and advances in recommender systems while simultaneously producing varying recommender models built on top of the [HETREC 2011](https://grouplens.org/datasets/hetrec-2011/) {cite}`cantador2011second` 2nd International Workshop on Information Heterogeneity and Fusion in Recommender Systems [Last.FM](https://www.last.fm/) dataset. 

We began by incorporating a Matrix Factorisation model, which calculates the product between learned user and item embeddings to produce a score for that particular user-item combination. This type of model proved very intuitive, but lacks in some predictive power and tends to be biased towards recommending popular items.

We then moved on to developing deep learning recommender systems using Keras on top of Tensorflow, specifically using their TFRS package. This package makes development of recommender systems an easy and fluid process. We only scratched the surface of what is capable with this package. In the future, it would be interesting to explore the more complex offerings of TFRS with this relatively simple dataset.
