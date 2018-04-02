import pandas as pd


resturants = pd.read_csv("resturants.csv")
reviews = pd.read_csv("yelp_review.csv")

resturant_reviews = pd.merge(resturants, reviews, on='business_id', copy = False)
resturants.info()
print("************************\n\n\n here are the reviews \n\n\n****************************")
resturant_reviews.info()
