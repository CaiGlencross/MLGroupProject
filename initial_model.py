import pandas as pd
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def convert_booleans(x):
	if (x=="TRUE"):
		return 1
	elif (x=="FALSE"):
		return 0
	else:
		return x


def get_data_from_csv(csv_filename, feature_array, threshold=4.5):
	restaurants = pd.read_csv(csv_filename)

	nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
	#features with a solid amount of non-null values
	g_restaurants = nan_restaurants[np.hstack((["stars"],feature_array))]
	#only use restuarants that have every one of those features
	clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
	#review_count = clean_restaurants["review_count"]

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



	# decile = pd.qcut(clean_restaurants["review_count"], 10)
	# print decile
	#change the booleans to integers
	int_restaurants_clean = clean_restaurants.applymap(convert_booleans)
	#seperate data and labels
	restaurant_data = int_restaurants_clean[feature_array]
	restaurant_labels = int_restaurants_clean[["stars"]]
	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)

	#turn pandas dataframe into a numpy array
	X = restaurant_data.values
	y = restaurant_labels.values
	y = np.ravel(y)

	return X, y

def get_norm_data_from_csv(csv_filename, feature_array, threshold=4.5):
	restaurants = pd.read_csv(csv_filename)

	nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
	#features with a solid amount of non-null values
	g_restaurants = nan_restaurants[np.hstack((["stars"],feature_array))]
	#only use restuarants that have every one of those features
	clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
	#normalize the num_reviews column. if it exists
	if "review_count" in feature_array:
		print "entered the if statement"
		the_max = clean_restaurants["review_count"].max()
		print "max is %f" % (the_max)
		clean_restaurants["review_count"] = clean_restaurants["review_count"].map(lambda u: u / float(the_max))


	#change the booleans to integers
	int_restaurants_clean = clean_restaurants.applymap(convert_booleans)
	#seperate data and labels
	restaurant_data = int_restaurants_clean[feature_array]
	restaurant_labels = int_restaurants_clean[["stars"]]
	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)

	#turn pandas dataframe into a numpy array
	X = restaurant_data.values
	y = restaurant_labels.values
	y = np.ravel(y)

	return X, y



def get_data_with_cutoff(csv_filename, feature_array, cutoff_feature = "review_count", cutoff_threshold = 797, threshold = 4.5):
	restaurants = pd.read_csv(csv_filename)

	nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
	#features with a solid amount of non-null values
	g_restaurants = nan_restaurants[np.hstack((["stars"],feature_array))]
	#only use restuarants that have every one of those features
	clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
	#apply the cut off to the data
	clean_restaurants[cutoff_feature] = clean_restaurants[cutoff_feature].map(lambda u : cutoff_threshold if u>cutoff_threshold else u)
	#normalize over num_reviews if it exists
	if "review_count" in feature_array:
		the_max = clean_restaurants["review_count"].max()
		clean_restaurants["review_count"] = clean_restaurants["review_count"].map(lambda u: u / float(the_max))

	#change the booleans to integers
	int_restaurants_clean = clean_restaurants.applymap(convert_booleans)
	#seperate data and labels
	restaurant_data = int_restaurants_clean[feature_array]
	restaurant_labels = int_restaurants_clean[["stars"]]
	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)

	#turn pandas dataframe into a numpy array
	X = restaurant_data.values
	y = restaurant_labels.values
	y = np.ravel(y)

	return X, y


def get_small_reviews_data(csv_filename, feature_array, threshold=4.5, small=100):
	restaurants = pd.read_csv(csv_filename)

	nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
	#filter out large reviews
	nan_restaurants["review_count"] = nan_restaurants["review_count"].map(lambda u: np.nan if u > small else u)
	#features with a solid amount of non-null values
	g_restaurants = nan_restaurants[np.hstack((["stars", "review_count"],feature_array))]
	#only use restuarants that have every one of those features
	clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
	#get rid of review count
	g_restaurants = nan_restaurants[np.hstack((["stars"],feature_array))]
	#use boolean integers
	int_restaurants_clean = clean_restaurants.applymap(convert_booleans)
	#seperate data and labels
	restaurant_data = int_restaurants_clean[feature_array]
	restaurant_labels = int_restaurants_clean[["stars"]]
	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)

	#turn pandas dataframe into a numpy array
	X = restaurant_data.values
	y = restaurant_labels.values
	y = np.ravel(y)

	return X, y


def get_large_reviews_data(csv_filename, feature_array, threshold=4.5, big=100):
	restaurants = pd.read_csv(csv_filename)

	nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
	#filter out large reviews
	nan_restaurants["review_count"] = nan_restaurants["review_count"].map(lambda u: np.nan if u < big else u)
	#features with a solid amount of non-null values
	g_restaurants = nan_restaurants[np.hstack((["stars", "review_count"],feature_array))]
	#only use restuarants that have every one of those features
	clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
	#get rid of review count
	g_restaurants = nan_restaurants[np.hstack((["stars"],feature_array))]
	#use boolean integers
	int_restaurants_clean = clean_restaurants.applymap(convert_booleans)
	#seperate data and labels
	restaurant_data = int_restaurants_clean[feature_array]
	restaurant_labels = int_restaurants_clean[["stars"]]
	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)

	#turn pandas dataframe into a numpy array
	X = restaurant_data.values
	y = restaurant_labels.values
	y = np.ravel(y)

	return X, y



#normalize our data so it has zero mean and a standard deviation of 1
#no need to pre process labels as they are binary
#X = preprocessing.scale(X)
def partition_data(X, y, training_portion = .8):
	n, d = X.shape

	train_partition = int(math.floor(training_portion * n ))

	X_training = X[:train_partition]
	y_training = y[:train_partition]

	X_test = X[train_partition:]
	y_test = y[train_partition:]

	return X_training, y_training, X_test, y_test
def determine_svm_hyperparameters(X_training, y_training, plot=False, show=False):
	#SVM model
	C_vals = [.001, .01, .1, .2,.3, 1.0, 1.5, 2.0, 2.5, 10, 20, 40, 60, 80, 100]
	c_avg_accs = []
	c_avg_f1 = []
	c_avg_auroc = []
	c_avg_prec = []
	for c in C_vals:
		model = SVC(C=c, kernel = "rbf")

		#k fold validation training
		avg_accs = []
		avg_f1s = []
		avg_aurocs = []
		avg_precs = []
		avg_percentage = 0
		kf = KFold(n_splits = 10)
		for train_index, val_index in kf.split(X_training):
			X_train, X_val = X_training[train_index], X_training[val_index]
			y_train, y_val = y_training[train_index], y_training[val_index]
			model.fit(X_train, y_train)
			y_pred = model.predict(X_val)

			avg_accs.append(model.score(X_val, y_val))
			avg_f1s.append(f1_score(y_val, y_pred))
			avg_aurocs.append(roc_auc_score(y_val, y_pred))
			avg_precs.append(precision_score(y_val, y_pred))

		# print "average accuracy for %f k-fold validation was %f" % (c, np.mean(avg_accs))
		c_avg_accs.append(np.mean(avg_accs))
		c_avg_f1.append(np.mean(avg_f1s))
		c_avg_auroc.append(np.mean(avg_aurocs))
		c_avg_prec.append(np.mean(avg_precs))
		#if show: print "average distribution is %f" % (avg_percentage/float(10))
	if plot:
		plt.figure()
		plt.xlabel("C values")
		plt.ylabel("metric score")
		plt.plot(C_vals, c_avg_accs, label="Accuracy")
		plt.plot(C_vals, c_avg_f1, label="F1_Score")
		plt.plot(C_vals, c_avg_auroc, label="AUROC")
		plt.plot(C_vals, c_avg_prec, label="Precision")
		plt.legend()
		plt.show()
	if show:
		print "optimal C for accuracy %f"  % (C_vals[np.argmax(c_avg_accs)])
		print "optimal C for f1 score %f"  % (C_vals[np.argmax(c_avg_f1)])
		print "optimal C for Auroc %f"     % (C_vals[np.argmax(c_avg_auroc)])
		print "optimal C for precision %f" % (C_vals[np.argmax(c_avg_prec)])

	return ((C_vals[np.argmax(c_avg_f1)]) + (C_vals[np.argmax(c_avg_prec)])) / 2


def print_results(y_true_test, y_true_train, y_pred_test, y_pred_train):
	print "test accuracy for model was %f" 		% 	(accuracy_score(y_true_test, y_pred_test))
	print "test f1 score for model was %f" 		% 	(f1_score(y_true_test, y_pred_test))
	print "test auroc score for model was %f" 	% 	(roc_auc_score(y_true_test, y_pred_test))
	print "test precision for model was %f" 	% 	(precision_score(y_true_test, y_pred_test))
	print "\n"	

	print "train accuracy for model was %f" 	% 	(accuracy_score(y_true_train, y_pred_train))
	print "train f1 score for model was %f" 	% 	(f1_score(y_true_train, y_pred_train))
	print "train auroc score for model was %f" 	% 	(roc_auc_score(y_true_train, y_pred_train))
	print "train precision for model was %f" 	% 	(precision_score(y_true_train, y_pred_train))

def main():
	#regular model
	initial_features = ["review_count", "BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	X, y = get_data_from_csv("resturants.csv", initial_features)
	#print X 


	X_training, y_training, X_test, y_test = partition_data(X,y)

	c = determine_svm_hyperparameters(X_training, y_training)
	print "c chosen was %f" % (c)


	model = SVC(C=c, kernel = "rbf")
	model.fit(X_training, y_training)

	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)




	#normalized model
	print "\n\n\n"
	X_norm, y_norm = get_norm_data_from_csv("resturants.csv", initial_features)
	#print X_norm

	X_training, y_training, X_test, y_test = partition_data(X_norm,y_norm)

	c = determine_svm_hyperparameters(X_training, y_training)
	print "c chosen was %f" % (c)

	model = SVC(C=c, kernel = "rbf")
	model.fit(X_training, y_training)
	

	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)




	#partition num reviews at 363
	print "\n\n\n"
	X_cutoff, y_cutoff = get_data_with_cutoff("resturants.csv", initial_features)
	#print X_cutoff
	X_training, y_training, X_test, y_test = partition_data(X_cutoff,y_cutoff)

	c_cutoff = determine_svm_hyperparameters(X_training, y_training)
	print "c for cutoff chosen was %f" % (c_cutoff)


	model = SVC(C=c_cutoff, kernel = "rbf")
	model.fit(X_training, y_training)

	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)



	#fit two models one on small reviews one on large reviews

	#small
	review_features = ["BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	X_small, y_small = get_small_reviews_data("resturants.csv", review_features)

	n, d = X_small.shape
	print "num small samples: %d" % (n)

	X_training, y_training, X_test, y_test = partition_data(X_small,y_small)

	c_small = determine_svm_hyperparameters(X_training, y_training)
	print "c for cutoff chosen was %f" % (c_small)


	model = SVC(C=c_cutoff, kernel = "rbf")
	model.fit(X_training, y_training)

	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)



	#large
	print "\n\n\n"
	review_features = ["BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	X_large, y_large = get_large_reviews_data("resturants.csv", review_features)

	n, d = X_large.shape
	print "num large samples: %d" % (n)

	X_training, y_training, X_test, y_test = partition_data(X_large,y_large)

	c_large = determine_svm_hyperparameters(X_training, y_training)
	print "c for large model chosen was %f" % (c_large)


	model = SVC(C=c_large, kernel = "rbf")
	model.fit(X_training, y_training)

	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)

	#no review count
	print "\n\n\n"
	second_features = ["BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	X, y = get_data_from_csv("resturants.csv", second_features)
	print X 


	X_training, y_training, X_test, y_test = partition_data(X,y)



	c = determine_svm_hyperparameters(X_training, y_training)
	print "c chosen was %f" % (c)


	model = SVC(C=c, kernel = "rbf")
	model.fit(X_training, y_training)



	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)


	#



if __name__ == "__main__" :
    main()
