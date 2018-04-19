import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import io
import os
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from numpy import *


from string import punctuation
# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle



def convert_booleans(x):
	if (x=="TRUE"):
		return 1
	elif (x=="FALSE"):
		return 0
	else:
		return x



def main():

	#plotting for visualization
	restaurants = pd.read_csv('../yelp-dataset/merged_NV_restaurant.csv', header=0)
	#restaurants_reviews = pd.read_csv('../yelp-dataset/review_shortened_rest_NV.csv', header=0)
	review_count = restaurants['review_count']
	# plt.figure()
	# plt.hist(review_count, bins=np.arange(review_count.min(), review_count.max()+1))
	# plt.xlabel("Number of Reviews")
	# plt.ylabel("Number of Restaurants")
	# plt.show()

	# plt.hist(review_count, bins=np.logspace(np.log10(1),np.log10(100), 7000))
	# plt.gca().set_xscale("log")
	# plt.xlabel("Number of Reviews (Log Scale)")
	# plt.ylabel("Number of Restaurants")
	# plt.show()

	#print pd.qcut(review_count, 4)
	decile = pd.qcut(review_count, 10)
	print decile


	# pos_text = "postive_reviews"
	# neg_text = "negative_reviews"

	# pos_text = open("{}".format(pos_text), 'w')
	# neg_text = open("{}".format(neg_text), 'w')

	# for index, row in restaurants_reviews.iterrows():
	# 	if row['stars_x'] >= 4.5:
	# 		pos_text.write(row['text'])
	# 	elif row['stars_x'] < 4.5:
	# 		neg_text.write(row['text'])

	# pos_text.close()
	# neg_text.close()

	#dictionary_pos = extract_dictionary('postive_reviews.txt')
	#print dictionary_pos



	# ratings = data["stars"]
	# review_count = data['review_count']

	# plt.figure()
	# plt.plot(ratings,review_count, "bo")
	# plt.xlabel("Rating")
	# plt.ylabel("Number of Reviews")
	# plt.show()


	


	# nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
	# #features with a solid amount of non-null values
	# g_restaurants = nan_restaurants[["stars","review_count", "BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]]
	# #only use restuarants that have every one of those features
	# clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
	# #change the booleans to integers
	# int_restaurants_clean = clean_restaurants.applymap(convert_booleans)
	# #seperate data and labels
	# restaurant_data = int_restaurants_clean[["review_count", "BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]]
	# restaurant_labels = int_restaurants_clean[["stars"]]
	# #use binary labels
	# #TODO: figure out procedurally what the best threshold is
	# threshold = 3.5

	# restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)

	# #turn pandas dataframe into a numpy array
	# X = restaurant_data.values
	# y = restaurant_labels.values
	# y = np.ravel(y)


	# n, d = X.shape

	# train_partition = int(math.floor(.8 * n ))

	# X_training = X[:train_partition]
	# y_training = y[:train_partition]

	# X_test = X[train_partition:]
	# y_test = y[train_partition:]

	# #SVM model
	# model = SVC(C=1, kernel = "linear")

	# #k fold validation training
	# avg_acc = 0
	# kf = KFold(n_splits = 10)
	# for train_index, val_index in kf.split(X_training):
	# 	X_train, X_val = X[train_index], X[val_index]
	# 	y_train, y_val = y[train_index], y[val_index]
	# 	model.fit(X_train, y_train)
	# 	avg_acc += model.score(X_val, y_val)





	# print "average accuracy for this k-fold validation was ", avg_acc/10

	# model.fit(X_training, y_training)

	# print "coefficients after using all of training data:"
	# print "\treview count: ", model.coef_[0][1]
	# print "\tBusinessParking_street: ", model.coef_[0][1]
	# print "\tBusinessParking_lot: ", model.coef_[0][2]
	# print "\tGoodForMeal_dinner: ", model.coef_[0][3]
	# print "\tGoodForMeal_lunch: ", model.coef_[0][4]
	# print "\tGoodForMeal_breakfast: ", model.coef_[0][5]






if __name__ == "__main__":
    main()





