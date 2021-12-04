#!/usr/bin/env python
# coding: utf-8

# # Advanced Deep Learning Recommender System
# In the following notebook, we will be creating another Deep-Learning Recommender using Tensorflow V2. For this iteration, we will try to incorporate text and timestamp data available to us. As already stated multiple times, the tags in this data are user-generated. Therefore, they are messy, inconsistent, and may not be entirely accurate and or useful. 
# 
# The TFRS package is incredibly robust, and offers plenty of direction for expansion of recommender systems. The library can tokenize text and timestamps into features. It processes text into a 'bag-of-words' representation, which it can then use to find similarities. It will be interesting to see if this approach alters recommendations to be affected more by the genre or tags associated with artists.
# 
# Similarly, it will be interesting to see how the inclusion of temporal data changes recommendations. In our data, a timestamp is associated with a *user*, *artist*, and *tag*. It indicates the exact time that particular user gave that artist that tag. It is entirely possible that amongst the tag information users who do not like particular artists have left negative tags. I wonder if an association will be made between few listens (*low weight*) and particular tag tokens.

# In[1]:


import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow_recommenders as tfrs


# ### Accounting for Tag Information
# In the following cell, we will prove that in our data, users can tag the same artist multiple times. Preferably, we would like only 1 tag for any user-artist association. We will order a user-artist tags dataset by their creation time and use the most recent tag and timestamp for each user-artist combination.

# In[2]:


#Let's read in genres and tags
tags = pd.read_csv('../data/tags.dat', sep='\t', encoding='latin-1')
user_tagging = pd.read_csv('../data/user_taggedartists.dat', sep='\t', encoding='latin-1')
user_tagging_time = pd.read_csv('../data/user_taggedartists-timestamps.dat', sep='\t', encoding='latin-1')

#Check if duplicates are present
if True in user_tagging_time[['userID', 'artistID']].duplicated():
    print("Contains Duplicate user-artist combinations.")


# In[3]:


u_tag_a = user_tagging.merge(tags[['tagID', 'tagValue']], on='tagID')
u_tag_a = u_tag_a.merge(user_tagging_time, on=['userID', 'artistID', 'tagID'])
print("Displaying 3 random samples for tag data:")
u_tag_a.sample(3)


# In[4]:


#Group by user-artist combo, sort by timestamp and extract that tagValue
u_tag_a = u_tag_a.sort_values(by='timestamp', ascending=False).groupby(['userID', 'artistID']).first().reset_index()


# In[5]:


#Let's check if this dataset contains duplicate user-artist combinations
if True in np.unique(u_tag_a[['userID', 'artistID']].duplicated().values):
    print("Contains Duplicate user-artist combinations.")
else:
    print("Does not contain Duplicate user-artist combinations.")


# ### Data Preprocessing
# Our preprocessing steps are as before for the most part. For a final step, we will merge our tag information dataset with our ratings matrix. To start, we normalise our weight column as previous.
# 
# There will be many cases where a user listens to a particular artist, but never provides that artist with a tag. In those cases, we will let the tag value be `no tag`, and for the corresponding timestamp value, we will use a value corresponding to today. We obviously want our model to find associations between users and common tags. However, our model can also build associations in situations where a user has decided to not provide a tag.
# 
# The timestamps provided in the dataset do not correspond to the correct year and must have the final 3 digits removed. For this, we can just divide them all by 1,000. Using [this website](https://timestamp.online/), I entered some of the corrected timestamps to ensure they do indeed correspond to the appropriate year. All entries I checked returned values around 2008 to 2011, which makes sense for this dataset.

# In[6]:


import time
import math

#Correct timestamp data in u_tag_a
u_tag_a['timestamp'] = u_tag_a['timestamp'].apply(lambda x: math.floor(x))


# In[7]:


#Let's define our amount of users
rating_matrix = pd.read_csv('../data/user_artists.dat', sep='\t', encoding='latin-1')
num_users = len(rating_matrix.userID.unique())


# In[8]:


#Let's normalise our weight column per user
new_rating_matrix = pd.DataFrame(columns=['userID', 'artistID', 'weight'])
for user_id in rating_matrix.userID.unique():
    user_ratings = rating_matrix[rating_matrix.userID == user_id]
    ratings = np.array(user_ratings['weight'])
    user_ratings['weight'] = tf.keras.utils.normalize(ratings, axis=-1, order=2)[0]
    new_rating_matrix = new_rating_matrix.append(user_ratings)
rating_matrix = new_rating_matrix
rating_matrix.describe()


# In[9]:


#Let's use left merge to merge our tag data and rating matrix
rating_matrix = rating_matrix.merge(u_tag_a[['userID', 'artistID', 'tagValue', 'timestamp']],
                                    on=['userID', 'artistID'], how='left')

#Get today's timestamp
now = math.floor(time.time())

#Fill missing values as stated
rating_matrix.tagValue = rating_matrix['tagValue'].fillna('no tag')
rating_matrix.timestamp = rating_matrix['timestamp'].fillna(now)


# In[10]:


print("Displaying Sample of new Rating Matrix")
rating_matrix[rating_matrix.tagValue != 'no tag'].sample(5)


# The small sample above gives an indication for some of the values we can expect to find for tags. There is a large amount of distinct values in our tag data. It will be interesting to see how the recommender system interprets these.
# 
# The below pre-processing steps are as before in our other notebooks. We are correcting the scale of user and artist ID's, then ensuring their maximum values are appropriate before replacing the columns in our rating matrix.
# 
# ---

# ### Artist Preprocessing
# To make use of the tags generally associated with artists, we will calculate their most popular tag in our data. We will use this information later on when developing our candidate model. The function in the cell below performs as previous. Essentially, it finds the most popular tag for each artist and attaches it to their profile.
# 
# We will add this extra information to our ratings matrix, as well as the artists name. Using the artist name as an identifier will make more sense to us than an ID number.

# In[11]:


#Let's match artists to genres
artists = pd.read_csv('../data/artists.dat', sep='\t', encoding='latin-1')
artists_tagged = user_tagging.merge(tags[['tagID', 'tagValue']], on='tagID')
artists_tagged = (artists_tagged.groupby('artistID')['tagValue'].apply(lambda grp: list(grp))).reset_index()

#This function performs as previous.
for index, row in artists_tagged.iterrows():
    d = {}
    new_tags = []
    for val in row.tagValue:
        if val not in d:
            d[val] = 1
        else:
            d[val] += 1
    for key, value in d.items():
        if d[key] >=3:
            new_tags.append([key, value])
    new_tags.sort(key=lambda x:x[1], reverse=True)
    if new_tags:
        artists_tagged.at[index, "tagValue"] = [tag[0] for tag in new_tags]
        artists_tagged.at[index, 'genre'] = artists_tagged.at[index, 'tagValue'][0]
        
#Let's add these tags to our artists
artists.rename(columns={'id':'artistID'}, inplace=True)
artists = artists.join(artists_tagged, on='artistID', how='left', rsuffix='right')
artists.tagValue = artists.tagValue.fillna('No Tags')
artists.genre = artists.genre.fillna('No Tags')
artists.rename(columns={'tagValue': 'genres'}, inplace=True)


# In[12]:


#We add the extra info to our ratings matrix
rating_matrix = rating_matrix.merge(artists[['artistID', 'name', 'genre']], on='artistID')


# In[13]:


#Extract userID column
userids = np.asarray(rating_matrix.userID)

#Remap the column
u_mapper, u_ind = np.unique(userids, return_inverse=True)

#Let's define our amount of artists
artists = pd.read_csv('../data/artists.dat', sep='\t', encoding='latin-1')
artists.rename(columns={'id':'artistID'}, inplace=True)
num_artists = len(artists.artistID.unique())

#Extract artistID column
artistids = np.asarray(rating_matrix.artistID)

#Remap the column
a_mapper, a_ind = np.unique(artistids, return_inverse=True)


# In[14]:


# Let's replace old columns with new ind ones
rating_matrix.userID = u_ind
rating_matrix.artistID = a_ind

#Let's ensure the max value is approriate
assert(rating_matrix.userID.unique().max() == 1891)
assert(rating_matrix.artistID.unique().max() == 17631)


# In[15]:


#We convert the ID's to string so we can use the StringLookup function later
rating_matrix.userID = rating_matrix.userID.apply(str)
rating_matrix.artistID = rating_matrix.artistID.apply(str)

rating_matrix.timestamp = rating_matrix.timestamp.apply(int)


# In[16]:


rating_matrix.sample(5)


# ## Model Development
# We will perform the steps to developing a model as previous. However, we will now utilise our tags and timestamps. We do this by instantiating our interactions dictionary and including the extra features.
# 
# ### Instantiate Interaction Dictionary
# Our interactions dictionary is just our rating matrix. It contains the following features `userID`, `artistID`, `name`, `weight`, `tag`, `genre`, and `timestamp`. We create a mapping for our dictionary below. We also create seperate individual mappings for artist ID's, tags, and genres.

# In[17]:


#Let's build our interactions dictionary as previous
interactions_dict = {name: np.array(value) for name, value in rating_matrix.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

items_dict = rating_matrix[['artistID']].drop_duplicates()
items_dict = {name: np.array(value) for name, value in items_dict.items()}
items = tf.data.Dataset.from_tensor_slices(items_dict)

names_dict = rating_matrix[['name']].drop_duplicates()
names_dict = {name: np.array(value) for name, value in names_dict.items()}
names = tf.data.Dataset.from_tensor_slices(names_dict)

tags_dict = rating_matrix[['tagValue']].drop_duplicates()
tags_dict = {name: np.array(value) for name, value in tags_dict.items()}
tags = tf.data.Dataset.from_tensor_slices(tags_dict)

genre_dict = rating_matrix[['genre']].drop_duplicates()
genre_dict = {name:np.array(value) for name, value in genre_dict.items()}
genres = tf.data.Dataset.from_tensor_slices(genre_dict)

interactions = interactions.map(lambda x: {
                                            'userID' : x['userID'], 
                                            'artistID' : x['artistID'], 
                                            'name' : x['name'],
                                            'weight' : float(x['weight']),
                                            'tag' : x['tagValue'],
                                            'genre': x['genre'],
                                            'timestamp': x["timestamp"],})

#artists = names.map(lambda x: x['name'])
items = items.map(lambda x: x['artistID'])
tags = tags.map(lambda x: x['tagValue'])
genres = genres.map(lambda x: x['genre'])


# ### Timestamp Normalisation
# As timestamps are represented as large integers, they are not healthy to use as direct input into our model. We firstly normalise our timestamps by calculaitng our minimum and maximum timestamp, then creating buckets at equal intervals between these two times. We instantiate 1000 buckets which are used to host our timestamps.

# In[18]:


#Let's create bins for our timestamps
max_timestamp = interactions.map(lambda x: x["timestamp"]).reduce(
    tf.cast(0, tf.int64), tf.maximum).numpy().max()

min_timestamp = interactions.map(lambda x: x["timestamp"]).reduce(
    np.int64(1e9), tf.minimum).numpy().min()

timestamp_buckets = np.linspace( min_timestamp, max_timestamp, num=1000,)

timestamps = interactions.map(lambda x: x["timestamp"]).batch(100)


# ### Lookup Tables & Training, Test Data Split
# In the following cell, we define various lookup tables which we may use later on. We also shuffle our data and create testing and training batches which will be fed into our model.

# In[19]:


### get unique item and user id's as a lookup table
unique_artist_ids = (np.unique(a_ind)).astype(str)
unique_user_ids = (np.unique(u_ind)).astype(str)
unique_genre_ids = np.unique(rating_matrix.genre)
unique_user_tags = np.unique(rating_matrix.tagValue)


# In[20]:


# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(62_000)
test = shuffled.skip(62_000).take(30_000)

cached_train = train.shuffle(62_000).batch(5_000)
cached_test = test.batch(2_500).cache()


# In[21]:


print(f'our test set is: {len(train)}')
print(f'our train set is: {len(test)}')


# ## Model Creation
# The below cells host a more complex version of the model we saw previously. In our previous model, we simply instantiated our user and item embeddings as we would in a regular collaborative filtering model. In this instance, we further develop our user and item models.
# 
# ---
# 
# ### User Model
# In the below cell, we develop our user model. We incorporate the user ID, as well as the timestamp data. As the timestamp data signifies when a user provided a tag to an artist, it is more suitably found in the user model.
# 
# In our user model, we have included a parameter, `_use_timestamps`. When set to true, the model incorporates time stamp information. This will allow us to compare the results of the model with, or without the use of timestamps.
# 
# In our dataset, it's hard to interpret the utility of timestamps. This is because timestamp information is related to when the user-provided tags were actually applied to the artist, rather than when the user posted the tag. Also, there is an argument that including timestamps of today's date may have negative effects on the model. This model is trained on information from 2009 to 2011, essentially making it a model of that period. Timestamps from today allow the model to 'see into the future', which is obviously not a realistic trait of ML models.

# In[22]:


### user model

class UserModel(tf.keras.Model):

    def __init__(self, use_timestamps):
        super().__init__()
        max_tokes=25_000

        self._use_timestamps = use_timestamps

        ## embed user id from unique_user_ids
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
        ])
        
        ## embed timestamp
        if use_timestamps:
            self.timestamp_embedding = tf.keras.Sequential([
              tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
              tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
            ])
            self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
            self.normalized_timestamp.adapt(timestamps)

    def call(self, inputs):
        if not self._use_timestamps:
              return self.user_embedding(inputs["userID"])

        ## all features here
        return tf.concat([
            self.user_embedding(inputs["userID"]),
            self.timestamp_embedding(inputs["timestamp"]),
            tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
        ], axis=1)


# ### Item Model
# Our item model incorporates our artist ID as before. However, it also makes use of the genre associated with the artist. 
# 
# To make use of genre strings, we must first instantiate our `genre_vectorizer`. This will allow us to convert our genre string into a numerical representation. The `genre_vectorizer` is then used by our `genre_text_embedding` processing step to create an embedding of this word vector. Word embeddings allow us to measure similarity between text.

# In[23]:


### candidate model

class ItemModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        max_tokens = 10_000
        
        ## embed artist id from unique_artist_ids
        self.artist_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_artist_ids),
            tf.keras.layers.Embedding(len(unique_artist_ids) + 1, 32),])
        
        ## processing text features: item genre vectorizer
        self.artist_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        ## we apply genre vectorizer to genres
        self.artist_text_embedding = tf.keras.Sequential([
                              self.artist_vectorizer,
                              tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
                              tf.keras.layers.GlobalAveragePooling1D(),])

        self.artist_vectorizer.adapt(genres)

    def call(self, inputs):
        return tf.concat([
            self.artist_embedding(inputs["artistID"]),
            self.artist_text_embedding(inputs["genre"]),], axis=1)


# ### Combining Models
# The following cell is our parent model which we use to combine the output of both our User and Item models. We feed the outputs of each model into two dense embedding layers both of the same shape (*32*). 
# 
# We then define our task (in this case FactorizedTopK), then compute the loss as we did previously. 

# In[24]:


class MusicModel(tfrs.models.Model):
    def __init__(self, use_timestamps):
        super().__init__()

        ## query model is user model
        self.query_model = tf.keras.Sequential([
                          UserModel(use_timestamps),
                          tf.keras.layers.Dense(32)])
        
        ## candidate model is the item model
        self.candidate_model = tf.keras.Sequential([
                              ItemModel(),
                              tf.keras.layers.Dense(32)])
        
        ## retrieval task, choose metrics
        self.task = tfrs.tasks.Retrieval(
                    metrics=tfrs.metrics.FactorizedTopK(
                        candidates=items.batch(128).map(self.candidate_model),),)

    def compute_loss(self, features, training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        
        query_embeddings = self.query_model({ "userID": features["userID"],
                                               "timestamp": features["timestamp"],
                                                "tag": features["tag"],
                                            })
        
        item_embeddings = self.candidate_model(features["genre"])

        return self.task(query_embeddings, item_embeddings)


# ### Model Fitting and Evaluation
# In the following cells, we will perform two different fitting and evaluation scenarios: $(a)$ `_use_timestamps = True`, and $(b)$ `_use_timestamps = False`.

# In[25]:


model = MusicModel(use_timestamps=False)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True)


# In[28]:


model = MusicModel(use_timestamps=True)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True)


# ### Output Analysis
# From the above output, it's clear that timestamps do not improve recommendations for this model. This is likely due to the fact that a substantial amount of the timestamps are of today's date which is throwing off the model. Perhaps a better data imputation would have been the median date observed in the data.
# 
# Otherwise both models seem to have reasonable performance with a positive item being returned as the top candidate 50% of the time. These models will supply a basis for our next advanced deep retrieval model.

# ## Conclusions
# In this notebook, we experimented with building a more complex deep learning framework by enhancing the input used in our query and item towers. Leveraging context features such as timestamps and text data can lead to better model performance and higher quality recommendations being produced.
# 
# However, we learned that proper imputation of data is an important aspect of data quality. Poorly imputated data can lead to contextual features having a negative effect on retrieval. In our case, it does not make sense to use today's timestamp in our data, as this allows the model to essentially 'see into the future'. This is an unrealisticquality of our model.
# 
# Although being more complex than our previos models, it still remains in its imfancy, and TFRS offers many directions to expand our model further. We will leave this for future work.
