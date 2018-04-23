#Author: Cai Glencross

#This file is for training an SVM on a bag of words vector
#Where each "word is a category"

#"merged_NV_restaurant.csv"
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from initial_model import *
from info_gain_category import summary_finder

def read_in_category_data(csv_filename, threshold=4.5):
	df = pd.read_csv(csv_filename)

	a = list(map(lambda u: u.split(";"), df['categories']))

	flatten = lambda l: [item for sublist in l for item in sublist]

	flat_list = flatten(a)

	categories = np.unique(flat_list)

	for category in categories:
	    within_list = lambda u: 1 if category in list(u.split(';')) else 0
	    df[category] = df['categories'].apply(within_list)

	cat_count = np.hstack((categories,"review_count"))

	df_cat_count = df[cat_count]

	df_cat_count["review_count"] = df_cat_count["review_count"].map(lambda u: u if u>100 else np.nan)

	df_cat_count = df_cat_count.dropna(how="any", axis= 0)

	df_categories = df_cat_count[categories]





	X = df_categories.values

	restaurant_labels = df[["stars", "review_count"]]

	restaurant_labels["review_count"] = restaurant_labels["review_count"].map(lambda u: u if u>100 else np.nan)

	restaurant_labels = restaurant_labels.dropna(how="any", axis=0)

	restaurant_labels = restaurant_labels[["stars"]]


	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else -1)

	y = np.ravel(restaurant_labels.values)

	##### start TODO ######################
	# add in the curated indicies

	feature_names = df_categories.columns.tolist()
	X_curated, col_names = summary_finder(X,y, feature_names, N_CATEGORIES=100)
	print(col_names)
	# Change the X to smaller categories
	X = X_curated

	###### end TODO ########################


	return X, y



def read_supplemented_category_data(csv_filename, threshold=4.5, num_categories = 20):
	df = pd.read_csv(csv_filename)

	a = list(map(lambda u: u.split(";"), df['categories']))

	flatten = lambda l: [item for sublist in l for item in sublist]

	flat_list = flatten(a)

	categories = np.unique(flat_list)

	for category in categories:
	    within_list = lambda u: 1 if category in list(u.split(';')) else 0
	    df[category] = df['categories'].apply(within_list)

	cat_count = np.hstack((categories,"review_count"))

	df_cat_count = df[cat_count]

	df_cat_count["review_count"] = df_cat_count["review_count"].map(lambda u: u if u>100 else np.nan)

	df_cat_count = df_cat_count.dropna(how="any", axis= 0)

	df_categories = df_cat_count[categories]

	best_categories = get_best_categories(num_categories)

	best_metadata_stars = ["BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast", "stars"]

	best_features = np.hstack((best_categories, best_metadata_stars))

	df_supplemented = df[best_features]

	df_supplemented = df_supplemented.dropna(how="any", axis=0)

	best_metadata= ["BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	best_x_features = np.hstack((best_categories, best_metadata))


	X = df_supplemented[best_x_features].values

	# restaurant_labels = df[["stars", "review_count"]]

	# restaurant_labels["review_count"] = restaurant_labels["review_count"].map(lambda u: u if u>100 else np.nan)

	# restaurant_labels = restaurant_labels.dropna(how="any", axis=0)

	restaurant_labels = df_supplemented[["stars"]]


	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else -1)

	y = np.ravel(restaurant_labels.values)




	return X, y

def random_forest_trainer(X,y):
	X_training, y_training, X_test, y_test = partition_data(X,y)

	depth = determine_forest_hyperparameters(X_training, y_training)

	model = RandomForestClassifier(max_depth=depth)

	model.fit(X_training,y_training)

	y_pred_train = model.predict(X_training)
	y_pred_test = model.predict(X_test)
	print "\n\n***RANDOM FOREST RESULTS***\n\n"
	print_results(y_test, y_training, y_pred_test, y_pred_train)

def get_tfidf_train_test(csv_filename, threshold=4.0):

	restaurants = pd.read_csv(csv_filename, encoding='utf-8', low_memory=False)

	review_list = restaurants['text']


	cleaned_list = [x for x in review_list if str(x) != 'nan']
	cleaned_df = pd.DataFrame(cleaned_list, columns=['review'])

	#update after preprocessing
	cleaned_df = preprocessing(cleaned_df, 'review')

	#set the targets based on threshold of 4.5 stars
	y = restaurants['stars'].apply(lambda u: 1 if u >= threshold else -1)

	#get the training and test data ready
	#TODO: kfold cv for train/test split
	y = np.array(y)




	X_train, X_test, y_train, y_test = train_test_split(cleaned_df['review'],y, shuffle=True, random_state= 123)


	tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english')

	tfidf.fit(X_train)

	X_train = tfidf.transform(X_train).toarray()
	X_test = tfidf.transform(X_test).toarray()

	return X_train, y_train, X_test, y_test


def get_best_categories(number):
	df = pd.read_csv("category_infogain.csv")

	return df['feature'].values[:number]






def main():
	X, y = read_supplemented_category_data("merged_NV_restaurant.csv", threshold=4.0)

	print "X.shape", X.shape
	print "y.shape", y.shape

	# percentage_min = np.count_nonzero(y == 1) / float(y.size)
	# percentage_maj = 1 - percentage_min
	# min_weight = percentage_maj/percentage_min
	# print("weight that will be used: ", min_weight)

	# print("\n\n*****weighted results*******\n\n")

	X_training, y_training, X_test, y_test = partition_data(X,y)

	# c_categories = determine_svm_hyperparameters(X_training, y_training, plot=True, weight = min_weight)
	# print "c for weighted data = " , c_categories
	# model = SVC(C=c_categories, kernel = "rbf", class_weight = {-1 : 1, 1 : min_weight})
	# model.fit(X_training, y_training)


	# y_pred_test = model.predict(X_test)
	# y_pred_train = model.predict(X_training)

	# print_results(y_test, y_training, y_pred_test, y_pred_train)


	print("\n\n*****unweighted results*******\n\n")


	c_categories = determine_svm_hyperparameters(X_training, y_training, plot=True)
	print "c for unweighted data = " , c_categories
	model = SVC(C=c_categories, kernel = "rbf")
	model.fit(X_training, y_training)


	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)


	# print("\n\n*****logistic regression model results*****\n\n")


	# c_categories = determine_logreg_hyperparameters(X_training, y_training)
	# print "c for logistic unweighted data = " , c_categories
	# model = LogisticRegression(C=c_categories)
	# model.fit(X_training, y_training)


	# y_pred_test = model.predict(X_test)
	# y_pred_train = model.predict(X_training)

	# print_results(y_test, y_training, y_pred_test, y_pred_train)

	# X_train, y_train, X_test, y_test = get_tfidf_train_test("star_results.csv", threshold = 4.0)

	# print_baseline_classifiers(X_train, y_train, X_test, y_test, plot = True)


	#random_forest_trainer(X_baseline, y_baseline)



if __name__ == "__main__" :
    main()
