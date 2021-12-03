import tensorflow.compat.v1 as tf
from CFModel import CFModel
import pandas as pd
import collections
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from IPython import display
import sklearn
import sklearn.manifold
from matplotlib import pyplot as plt
import altair as alt
alt.data_transformers.enable('default', max_rows=None)
alt.renderers.enable('default')

#Function to build sparse representation of rating matrix
def build_rating_sparse_tensor(ratings_df, num_queries, num_items):
    """
    Args:
    ratings_df: a pd.DataFrame with `userID`, `artistID` and `listened` columns.
    Returns:
    a tf.SparseTensor representing the ratings matrix.
    """
    indices = ratings_df[['userID', 'artistID']].values
    values = ratings_df['listened'].values
    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[num_queries, num_items])

# Utility to split the data into training and test sets.
def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
        df: a dataframe.
        holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
        train: dataframe for training
        test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test

def sparse_mean_square_error(sparse_ratings, query_embeddings, item_embeddings):
    """
    Args:
        sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
        query_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
        dimension, such that U_i is the embedding of user i.
        item_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
        dimension, such that V_j is the embedding of movie j.
    Returns:
        A scalar Tensor representing the MSE between the true ratings and the
        model's predictions.
    """
    predictions = tf.gather_nd(
        tf.matmul(query_embeddings, item_embeddings, transpose_b=True),
        sparse_ratings.indices)
    loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
    return loss

def build_model(ratings, num_queries, num_items, embedding_dim=3, init_stddev=1.):
    """
    Args:
        ratings: a DataFrame of the ratings
        embedding_dim: the dimension of the embedding vectors.
        init_stddev: float, the standard deviation of the random initial embeddings.
    Returns:
        model: a CFModel.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(train_ratings, num_queries, num_items)
    A_test = build_rating_sparse_tensor(test_ratings, num_queries, num_items)
    
    # Initialize the embeddings using a normal distribution.
    U = tf.Variable(tf.random_normal(
        [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal(
        [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
    train_loss = sparse_mean_square_error(A_train, U, V)
    test_loss = sparse_mean_square_error(A_test, U, V)
    metrics = {
        'train_error': train_loss,
        'test_error': test_loss
    }
    embeddings = {
        "userID": U,
        "artistID": V
    }
    return CFModel(embeddings, train_loss, [metrics])

def compute_scores(query_embedding, item_embeddings, measure):
    """Computes the scores of the candidates given a query.
    Args:
        query_embedding: a vector of shape [k], representing the query embedding.
        item_embeddings: a matrix of shape [N, k], such that row i is the embedding
            of item i.
        measure: a string specifying the similarity measure to be used. Can be
            either DOT or COSINE.
    Returns:
        scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    u = query_embedding
    V = item_embeddings
    if measure == 'cosine':
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    scores = u.dot(V.T)
    return scores

# @title User recommendations and nearest neighbors (run this cell)
def user_recommendations(model, measure, exclude_rated=False, k=6):
  if USER_RATINGS:
    scores = compute_scores(
        model.embeddings["userID"][1892], model.embeddings["artistID"], measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
        score_key: list(scores),
        'artistID': artists['id'],
        'name': artists['name']
    })
    if exclude_rated:
      # remove movies that are already rated
      rated_movies = ratings[ratings.userID == "1892"]["artistID"].values
      df = df[df.artistID.apply(lambda artistID: artistID not in rated_movies)]
    display.display(df.sort_values([score_key], ascending=False).head(k))  

def item_neighbors(model, title_substring, measure, items, k=6):
  # Search for movie ids that match the given substring.
  ids =  items[items['name'].str.contains(title_substring)].index.values
  titles = items.iloc[ids]['name'].values
  if len(titles) == 0:
    raise ValueError("Found no artists with the name %s" % title_substring)
  print("Nearest neighbors of : %s." % titles[0])
  if len(titles) > 1:
    print("[Found more than one matching artist. Other candidates: {}]".format(
        ", ".join(titles[1:])))
  artistID = ids[0]
  scores = compute_scores(
      model.embeddings["artistID"][artistID], model.embeddings["artistID"],
      measure)
  score_key = measure + ' score'
  df = pd.DataFrame({
      score_key: list(scores),
      'name': items['name']
  })
  display.display(df.sort_values([score_key], ascending=False).head(k))

  # @title Embedding Visualization code (run this cell)

import seaborn as sns
def item_embedding_norm(models, artists, rating_matrix):
    """Visualizes the norm and number of ratings of the item embeddings.
    Args:
        model: A MFModel object.
        artists: A DF with total ratings for each artist.
        rating_matrix: Rating Matrix to count weights.
    """
    if not isinstance(models, list):
        models = [models]
    df = pd.DataFrame({
        'artistID': artists.sort_values(by='artistID')['artistID'].values,
        'name': artists.sort_values(by='artistID')['name'].values,
        'rating': rating_matrix[['artistID', 'userID']].sort_values(by='artistID').groupby('artistID').count()['userID'].values,
    })
    charts = []
    for i, model in enumerate(models):
        norm_key = 'norm'+str(i)
        df[norm_key] = np.linalg.norm(model.embeddings["artistID"], axis=1)
        
        plt.figure(figsize=(10,10))
        plt.title('Total Listens by Norm of Embeddings')
        sns.scatterplot(x='rating', y='norm0', data=df)
        plt.ylabel('Norm of Embedding')
        plt.xlabel('Total Listens (Millions)')
        plt.show()

def visualize_item_embeddings(data, x, y):
    nearest = alt.selection(
        type='single', encodings=['x', 'y'], on='mouseover', nearest=True,
        empty='none')
    base = alt.Chart().mark_circle().encode(
        x=x,
        y=y,
        color=alt.condition(genre_filter, "genre", alt.value("whitesmoke")),
    ).properties(
        width=600,
        height=600,
        selection=nearest)
    text = alt.Chart().mark_text(align='left', dx=5, dy=-5).encode(
        x=x,
        y=y,
        text=alt.condition(nearest, 'name', alt.value('')))
    return alt.hconcat(alt.layer(base, text), genre_chart, data=data)

def tsne_item_embeddings(model, artists):
    """Visualizes the movie embeddings, projected using t-SNE with Cosine measure.
    Args:
        model: A MFModel object.
    """
    tsne = sklearn.manifold.TSNE(
        n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
        init='pca', verbose=True, n_iter=400)

    print('Running t-SNE...')
    V_proj = tsne.fit_transform(model.embeddings["artistID"])
    artists.loc[:,'x'] = V_proj[:, 0]
    artists.loc[:,'y'] = V_proj[:, 1]
    return visualize_item_embeddings(artists, 'x', 'y')

# @title Solution
def gravity(U, V):
  """Creates a gravity loss given two embedding matrices."""
  return 1. / (U.shape[0]*V.shape[0]) * tf.reduce_sum(
      tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))

def build_regularized_model(
    ratings, num_queries, num_items, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1.,
    init_stddev=0.1):
  """
  Args:
    ratings: the DataFrame of movie ratings.
    embedding_dim: The dimension of the embedding space.
    regularization_coeff: The regularization coefficient lambda.
    gravity_coeff: The gravity regularization coefficient lambda_g.
  Returns:
    A CFModel object that uses a regularized loss.
  """
  # Split the ratings DataFrame into train and test.
  train_ratings, test_ratings = split_dataframe(ratings)
  # SparseTensor representation of the train and test datasets.
  A_train = build_rating_sparse_tensor(train_ratings, num_queries, num_items)
  A_test = build_rating_sparse_tensor(test_ratings, num_queries, num_items)
  U = tf.Variable(tf.random_normal(
      [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
  V = tf.Variable(tf.random_normal(
      [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))

  error_train = sparse_mean_square_error(A_train, U, V)
  error_test = sparse_mean_square_error(A_test, U, V)
  gravity_loss = gravity_coeff * gravity(U, V)
  regularization_loss = regularization_coeff * (
      tf.reduce_sum(U*U)/U.shape[0] + tf.reduce_sum(V*V)/V.shape[0])
  total_loss = error_train + regularization_loss + gravity_loss
  losses = {
      'train_error_observed': error_train,
      'test_error_observed': error_test,
  }
  loss_components = {
      'observed_loss': error_train,
      'regularization_loss': regularization_loss,
      'gravity_loss': gravity_loss,
  }
  embeddings = {"userID": U, "artistID": V}

  return CFModel(embeddings, total_loss, [losses, loss_components])