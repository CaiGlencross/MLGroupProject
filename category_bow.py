#Author: Cai Glencross

#This file is for training an SVM on a bag of words vector 
#Where each "word is a category"

#"merged_NV_restaurant.csv"
import pandas as pd
import numpy as np
from initial_model import *

def read_in_category_data(csv_filename, threshold=4.5):
	df = pd.read_csv(csv_filename)

	a = list(map(lambda u: u.split(";"), df['categories']))

	flatten = lambda l: [item for sublist in l for item in sublist]

	flat_list = flatten(a)

	categories = np.unique(flat_list)

	for category in categories:
	    within_list = lambda u: 1 if category in list(u.split(';')) else 0
	    df[category] = df['categories'].apply(within_list)


	df_categories = df[categories]

	X = df_categories.values

	restaurant_labels = df[["stars"]]
	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)

	y = np.ravel(restaurant_labels.values)

	return X, y



def main():
	X, y = read_in_category_data("merged_NV_restaurant.csv")

	percentage_min = np.count_nonzero(y == 1) / float(y.size)
	percentage_maj = 1 - percentage_min
	min_weight = percentage_maj/percentage_min
	print("weight that will be used: ", min_weight)

	X_training, y_training, X_test, y_test = partition_data(X,y)

	c_categories = determine_svm_hyperparameters(X_training, y_training, plot=True, weight = min_weight)
	print "c for weighted data = " , c_categories
	model = SVC(C=c_categories, kernel = "rbf", class_weight = {0 : 1, 1 : min_weight})
	model.fit(X_training, y_training)


	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)


	print("\n\n*****unweighted results*******\n\n")


	c_categories = determine_svm_hyperparameters(X_training, y_training, plot=True)
	print "c for unweighted data = " , c_categories
	model = SVC(C=c_categories, kernel = "rbf")
	model.fit(X_training, y_training)


	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)









if __name__ == "__main__" :
    main()


	