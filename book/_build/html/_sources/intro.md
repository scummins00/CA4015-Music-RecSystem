# Introduction

Computer-based systems provide us with the opportunity of mining user history to reveal trends and personal preferences that they, and those around them, may not have previously realised {cite}`ekstrand2011collaborative`. A vast amount of research has been conducted on how to best recommend content to users and many methods have been proposed {cite}`marko1997, pazzani1999framework,guttman1998agent`.

In this research, we will be focusing on Collaborative Filtering {cite}`thorat2015survey,su2009survey`, an approach that decides what to recommend to a user based on the preferences of similar users {cite}`Elahi_Ricci_Rubens_2016,su2009survey`. Similarity between users is estimated by comparing user-interactions and ratings with products or services {cite:}`Elahi_Ricci_Rubens_2016, horat2015survey`. The table below highlights some popular services currently using recommender systems.

| Service         | Recommendations|
| :-------------: |:--------------:|
| Amazon          | Retail Products|
| Facebook (Meta) | Friends        |
| Netflix         | Movies / Shows |
| Spotify         | Music          |
| Youtube         | Videos         |
| Google News     | News           |

Recommending something to someone else is a natural process that aims to help somebody sift through a large corpus of information efficiently to find only the items of most interest to them. Recommender systems augment that process, and as such have become a vital tool for information-availability in our digital lifestyle {cite}`su2009survey`.

In the following Jupyter Book, we will be using data acquired from the [HETREC 2011](https://grouplens.org/datasets/hetrec-2011/){cite}`cantador2011second` 2nd International Workshop on Information Heterogeneity and Fusion in Recommender Systems. In particular, we will be focusing on the *artist listening records* dataset published by [Last.FM](https://www.last.fm/). This dataset consists of 92,800 artists listening records from 1,892 users. We will explore this data in more detail in {doc}`./Preliminary_Data_Analysis`. In the following section, we will discuss in-depth two approaches to Collaborative Filtering, namely; *Matrix Factorisation*, and *Recommendation with Deep Neural Networks*.

## Collaborative Filtering In-Depth
Collaborative Filtering (CF) aims to solve the problem of recommending $m$ items, by $n$ users where the data is often represented by a $n \times m$ matrix.

### Matrix Factorisation

### Deep Neural Networks