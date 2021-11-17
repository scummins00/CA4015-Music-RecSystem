#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling
# In this Notebook, we will be performing all the pre-requisite data operations required for the construction of our recommender system. In particular, we will be creating our ratings matrix $A$, where a given entry $A_{ij}$ indicates the amount a user $i$ has listened to an artist $j$. As in an given case a particular user will have listened to only a subset of artists, the matrix will be very sparse. 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#Constructing matrix A

