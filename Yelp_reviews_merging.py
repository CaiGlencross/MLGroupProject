
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# In[2]:

df = pd.read_csv('../yelp-dataset/merged_NV_restaurant.csv', header=0)
df = df.dropna(axis=1,thresh=300)


# In[3]:

df_reviews = pd.read_csv("../yelp-dataset/yelp_review.csv")


# In[4]:

df_reviews.head()


# In[5]:

df_attempt = pd.merge(df, df_reviews, on='business_id')


# In[6]:

df_attempt.info()


# In[7]:

df_attempt.to_csv('../yelp-dataset/review_shortened_rest_NV.csv')


# In[8]:

df_attempt.head()


# In[9]:

df_reviews.info()


# In[10]:

temp = df_attempt.groupby(['business_id'])


# In[12]:

a = temp['text'].apply(list)


# In[16]:

b = pd.DataFrame(a)


# In[19]:

print(b.ix[0].text)


# In[ ]:




# In[20]:

a


# In[21]:

b.info()
b.to_csv('../yelp-dataset/NV_combined_reviews.csv')


# In[ ]:



