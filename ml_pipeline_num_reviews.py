import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# 04/18/2018
# author: Katie
# purpose: match the stars to the collagated reviews, and the business id's 

# Get Business review dataframe

BUSINESS_FILEPATH = "../data/yelp-dataset/yelp_business.csv"
REVIEWS_FILEPATH = "../data/yelp-dataset/yelp_review.csv"

business = pd.read_csv(BUSINESS_FILEPATH)
MIN_REVIEW_COUNT = 100

business = business[business['categories'].str.contains("Restaurants")]

df_business = business.applymap(lambda x: np.nan if x == "Na" else x) #data cleaning
#apply filters
df_business = df_business[df_business['state'] == "NV"]
df_business = df_business[df_business['review_count'] >= MIN_REVIEW_COUNT]

# Get the review dataframes
reviews = pd.read_csv(REVIEWS_FILEPATH)

#replicates information but it's okay
df_review_list = pd.merge(df_business, reviews, on='business_id')

########################
grouped_obj = df_review_list.groupby(['business_id'])
# push reviews together
list_together = grouped_obj['text'].apply(list).reset_index()
condensed_reviews = pd.DataFrame(list_together)


print(condensed_reviews.head())
print(condensed_reviews.info())

#get the stars alongside the reviews
df_business_shortened = df_business[["business_id", "stars"]]
df_result = pd.merge(condensed_reviews, df_business_shortened, on ="business_id")

df_result.to_csv("star_results.csv")

print(df_result.head(10))
