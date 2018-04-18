
import pandas as pd
import numpy as np
df = pd.read_csv("merged_NV_restaurant.csv")


a = list(map(lambda u: u.split(";"), df['categories']))


flatten = lambda l: [item for sublist in l for item in sublist]

flat_list = flatten(a)


categories = np.unique(flat_list)


categories


for category in categories:
    within_list = lambda u: 1 if category in list(u.split(';')) else 0
    df[category] = df['categories'].apply(within_list)


df_categories = df[categories]

df_categories.info()

df.to_csv("categorized_data.csv")




