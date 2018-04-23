import pandas as pd
import numpy as np
import math
import matplotlib as mpl
mpl.use('TkAgg')
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from nltk_processing import *

from matplotlib.colors import Normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sys
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
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else -1)

	#turn pandas dataframe into a numpy array
	X = restaurant_data.values
	y = restaurant_labels.values
	y = np.ravel(y)

	return X, y

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

def get_norm_data_from_csv(csv_filename, feature_array, threshold=4.5):
	restaurants = pd.read_csv(csv_filename)

	nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
	#features with a solid amount of non-null values
	g_restaurants = nan_restaurants[np.hstack((["stars"],feature_array))]
	#only use restuarants that have every one of those features
	clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
	#normalize the num_reviews column. if it exists
	if "review_count" in feature_array:
		#print "entered the if statement"
		the_max = clean_restaurants["review_count"].max()
		print "max is %f" % (the_max)
		clean_restaurants["review_count"] = clean_restaurants["review_count"].map(lambda u: u / float(the_max))


	#change the booleans to integers
	int_restaurants_clean = clean_restaurants.applymap(convert_booleans)
	#seperate data and labels
	restaurant_data = int_restaurants_clean[feature_array]
	restaurant_labels = int_restaurants_clean[["stars"]]
	#use binary labels
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else -1)

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
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else -1)

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
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else -1)

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
	restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else -1)

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


def determine_svm_hyperparameters(X_training, y_training, plot=False, show=False, weight = 1):
	#SVM model
	C_vals = [.001, .01, .1, 1.0, 1.5, 2.0, 2.5, 10, 20, 40, 60, 80, 100]
	c_avg_accs = []
	c_avg_f1 = []
	c_avg_auroc = []
	c_avg_prec = []
	for c in C_vals:
		model = SVC(C=c, kernel = "rbf", class_weight = {-1:1, 1:weight})

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

	# need to pick the best f1_score, within some threshold of precision

	precision_threshold = .6 #given by prof Wu
	best_index = -1
	while best_index == -1:
		max_f1 = 0
		for i in range(len(c_avg_prec)):
			if c_avg_prec[i] >= precision_threshold and c_avg_f1[i] >= max_f1:
				max_f1 = c_avg_f1[i]
				best_index = i


		precision_threshold = precision_threshold-.1


	return C_vals[best_index]


def determine_logreg_hyperparameters(X_training, y_training, plot=False, show=False, weight = 1):
	#SVM model
	C_vals = [100,1000,5000, 7500, 10000, 15000, 20000]
	c_avg_accs = []
	c_avg_f1 = []
	c_avg_auroc = []
	c_avg_prec = []
	for c in C_vals:
		model = LogisticRegression(C=c, class_weight = {-1:1, 1:weight})

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

	# need to pick the best f1_score, within some threshold of precision

	precision_threshold = .6 #given by prof Wu
	best_index = -1
	while best_index == -1:
		max_f1 = 0
		for i in range(len(c_avg_prec)):
			if c_avg_prec[i] >= precision_threshold and c_avg_f1[i] > max_f1:
				max_f1 = c_avg_f1[i]
				best_index = i


		precision_threshold = precision_threshold-.1


	return C_vals[best_index]


def determine_forest_hyperparameters(X_training, y_training, plot=False, show=False, weight = 1):
	#SVM model
	C_vals = [10, 20, 50, 75,100, 400, 800, 1000,5000, 7000]
	c_avg_accs = []
	c_avg_f1 = []
	c_avg_auroc = []
	c_avg_prec = []
	for c in C_vals:
		model = RandomForestClassifier(max_depth = c)

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

	# need to pick the best f1_score, within some threshold of precision

	precision_threshold = .6 #given by prof Wu
	best_index = -1
	while best_index == -1:
		max_f1 = 0
		for i in range(len(c_avg_prec)):
			if c_avg_prec[i] >= precision_threshold and c_avg_f1[i] > max_f1:
				max_f1 = c_avg_f1[i]
				best_index = i


		precision_threshold = precision_threshold-.1


	return C_vals[best_index]





def print_baseline_classifiers(X_training, y_training, X_test, y_test, plot = False, show=False):

	#X_training, y_training, X_test, y_test = partition_data(X,y)

	train_f1_scores = []
	train_acc_scores = []
	train_prec_scores = []

	test_f1_scores = []
	test_acc_scores = []
	test_prec_scores = []

	#majority classifier:

	classes = np.unique(y_training)

	max_count = 0
	majority = -1

	for ass in classes:
		count = np.count_nonzero(y_training == ass)
		if count > max_count:
			max_count = count
			majority = ass


	y_pred_train = np.repeat(majority, y_training.size)
	y_pred_test = np.repeat(majority, y_test.size)

	print "\nResults for the Majority Classifier\n"

	if show: print_results(y_test, y_training, y_pred_test, y_pred_train)

	train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	train_f1_scores.append(train_f1)
	train_acc_scores.append(train_acc)
	train_prec_scores.append(train_prec)

	test_f1_scores.append(test_f1)
	test_acc_scores.append(test_acc)
	test_prec_scores.append(test_prec)



	#Random Classifier
	rate_of_bads = max_count/float(y_training.size)
	y_pred_train = np.random.rand(y_training.size)
	y_pred_test = np.random.rand(y_test.size)
	thresholder = lambda u: -1 if u<rate_of_bads else 1
	y_pred_train = np.array([thresholder(yi) for yi in y_pred_train])
	y_pred_test = np.array([thresholder(yi) for yi in y_pred_test])


	print "\nResults for the RandomClassifier\n"

	if show: print_results(y_test, y_training, y_pred_test, y_pred_train)

	train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	train_f1_scores.append(train_f1)
	train_acc_scores.append(train_acc)
	train_prec_scores.append(train_prec)

	test_f1_scores.append(test_f1)
	test_acc_scores.append(test_acc)
	test_prec_scores.append(test_prec)

	random_f1 = test_f1
	random_precision = test_prec


	#Dummy Classifier
	# clf = DummyClassifier()
	# clf.fit(X_training, y_training)
	# y_pred_train = clf.predict(X_training)
	# y_pred_test = clf.predict(X_test)

	# print "\nResults for the Dummy Classifier\n"


	# print_results(y_test, y_training, y_pred_test, y_pred_train)


	# train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	# test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_f1_scores.append(train_f1)
	# train_acc_scores.append(train_acc)
	# train_prec_scores.append(train_prec)

	# test_f1_scores.append(test_f1)
	# test_acc_scores.append(test_acc)
	# test_prec_scores.append(test_prec)



	#Naive Bayes

	
	# clf = GaussianNB()
	# clf.fit(X_training, y_training)
	# y_pred_train = clf.predict(X_training)
	# y_pred_test = clf.predict(X_test)

	# print "\nResults for the Gaussian Nave Bayesian\n"

	# print_results(y_test, y_training, y_pred_test, y_pred_train)


	# train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	# test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_f1_scores.append(train_f1)
	# train_acc_scores.append(train_acc)
	# train_prec_scores.append(train_prec)

	# test_f1_scores.append(test_f1)
	# test_acc_scores.append(test_acc)
	# test_prec_scores.append(test_prec)


	if plot:
		fig, ax = plt.subplots()
		ind = np.arange(len(test_acc_scores))
		width = 0.1

		p1 = ax.bar(ind, test_acc_scores, width, color = "maroon", label = "test accuracy")

		p2 = ax.bar(ind+width, test_f1_scores, width,  color ="orange", label = "test f1 score")

		p3 = ax.bar(ind+2*width, test_prec_scores, width,  color = "green",label = "test precision")

		p1 = ax.bar(ind+3*width, train_acc_scores, width,alpha = 0.5, color = "maroon", label = "train accuracy")

		p2 = ax.bar(ind+4*width, train_f1_scores, width,  alpha = 0.5, color ="orange", label = "train f1 score")

		p3 = ax.bar(ind+5*width, train_prec_scores, width, alpha = 0.5, color = "green",label = "train precision")

		ax.set_title("Baseline Results")
		ax.set_xticks((ind + 2.5*width) )
		ax.set_xticklabels(('Majority Classifier', 'Random Classifier', 'Dummy Classifier', 'Naive Bayes'))
		ax.set_ylim(bottom = 0, top = 1.0)
		ax.legend()


		plt.show()

		# fig, ax = plt.subplots()
		# ind = np.arange(4)
		# width = 0.15
		# p1 = ax.bar(ind, train_acc_scores, width, label = "accuracy")

		# p2 = ax.bar(ind+width, train_f1_scores, width, label = "f1 scores")

		# p3 = ax.bar(ind+2*width, train_prec_scores, width, label = "accuracy")

		# ax.set_title("Baseline Results (Training)")
		# ax.set_xticks((ind + width))
		# ax.set_xticklabels(('Majority Classifier', 'Random Classifier', 'Dummy Classifier', 'Naive Bayes'))

		# ax.legend()


		# plt.show()
	return random_f1, random_precision








def print_baseline_results(y_true_test, y_pred_test):
	print "test accuracy for model was %f" 		% 	(accuracy_score(y_true_test, y_pred_test))
	print "test f1 score for model was %f" 		% 	(f1_score(y_true_test, y_pred_test))
	print "test auroc score for model was %f" 	% 	(roc_auc_score(y_true_test, y_pred_test))
	print "test precision for model was %f" 	% 	(precision_score(y_true_test, y_pred_test))
	print "\n"	



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


def get_test_results(y_true_test, y_true_train, y_pred_test, y_pred_train):

	return accuracy_score(y_true_test, y_pred_test) , f1_score(y_true_test, y_pred_test), precision_score(y_true_test, y_pred_test)

def get_train_results(y_true_test, y_true_train, y_pred_test, y_pred_train):

	return accuracy_score(y_true_train, y_pred_train), f1_score(y_true_train, y_pred_train), precision_score(y_true_train, y_pred_train)

def main():


	train_f1_scores = []
	train_acc_scores = []
	train_prec_scores = []

	test_f1_scores = []
	test_acc_scores = []
	test_prec_scores = []


	# #regular model
	# print "\n\n\nResults for basic model"
	# initial_features = ["review_count", "BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	# X, y = get_data_from_csv("merged_NV_restaurant.csv", initial_features , threshold = 4.5)
	# #print X 


	# X_training, y_training, X_test, y_test = partition_data(X,y)

	# c = determine_svm_hyperparameters(X_training, y_training)
	# print "c chosen was %f" % (c)


	# model = SVC(C=c, kernel = "rbf")
	# model.fit(X_training, y_training)

	# y_pred_test = model.predict(X_test)
	# y_pred_train = model.predict(X_training)

	# print_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	# test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_f1_scores.append(train_f1)
	# train_acc_scores.append(train_acc)
	# train_prec_scores.append(train_prec)

	# test_f1_scores.append(test_f1)
	# test_acc_scores.append(test_acc)
	# test_prec_scores.append(test_prec)




	#normalized model
	# print "\n\n\nResults for normalized model"
	# X_norm, y_norm = get_norm_data_from_csv("merged_NV_restaurant.csv", initial_features , threshold = 4.0)
	# #print X_norm

	# X_training, y_training, X_test, y_test = partition_data(X_norm,y_norm)

	# c = determine_svm_hyperparameters(X_training, y_training)
	# print "c chosen was %f" % (c)

	# model = SVC(C=c, kernel = "rbf")
	# model.fit(X_training, y_training)
	

	# y_pred_test = model.predict(X_test)
	# y_pred_train = model.predict(X_training)

	# print_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	# test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_f1_scores.append(train_f1)
	# train_acc_scores.append(train_acc)
	# train_prec_scores.append(train_prec)

	# test_f1_scores.append(test_f1)
	# test_acc_scores.append(test_acc)
	# test_prec_scores.append(test_prec)




	#partition num reviews at 363
	# print "\n\n\nResults for cutoff reviews"
	# X_cutoff, y_cutoff = get_data_with_cutoff("merged_NV_restaurant.csv", initial_features , threshold = 4.5)
	# #print X_cutoff
	# X_training, y_training, X_test, y_test = partition_data(X_cutoff,y_cutoff)

	# c_cutoff = determine_svm_hyperparameters(X_training, y_training)
	# print "c for cutoff chosen was %f" % (c_cutoff)


	# model = SVC(C=c_cutoff, kernel = "rbf")
	# model.fit(X_training, y_training)

	# y_pred_test = model.predict(X_test)
	# y_pred_train = model.predict(X_training)

	# print_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	# test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_f1_scores.append(train_f1)
	# train_acc_scores.append(train_acc)
	# train_prec_scores.append(train_prec)

	# test_f1_scores.append(test_f1)
	# test_acc_scores.append(test_acc)
	# test_prec_scores.append(test_prec)



	#fit two models one on small reviews one on large reviews

	#small
	print "\n\n\nResults for small model"
	review_features = ["BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	X_small, y_small = get_small_reviews_data("merged_NV_restaurant.csv", review_features , threshold = 4.5)

	n, d = X_small.shape
	print "num small samples: %d" % (n)

	X_training, y_training, X_test, y_test = partition_data(X_small,y_small)

	c_small = determine_svm_hyperparameters(X_training, y_training)
	print "c for cutoff chosen was %f" % (c_small)


	model = SVC(C=c_small, kernel = "rbf")
	model.fit(X_training, y_training)

	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)

	train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	train_f1_scores.append(train_f1)
	train_acc_scores.append(train_acc)
	train_prec_scores.append(train_prec)

	test_f1_scores.append(test_f1)
	test_acc_scores.append(test_acc)
	test_prec_scores.append(test_prec)



	#large
	print "\n\n\nResults for large model"
	review_features = ["BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	X_large, y_large = get_large_reviews_data("merged_NV_restaurant.csv", review_features , threshold = 4.5)

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

	train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	train_f1_scores.append(train_f1)
	train_acc_scores.append(train_acc)
	train_prec_scores.append(train_prec)

	test_f1_scores.append(test_f1)
	test_acc_scores.append(test_acc)
	test_prec_scores.append(test_prec)

	#no review count
	print "\n\n\nResults for Model with no review count"
	second_features = ["BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]

	X, y = get_data_from_csv("merged_NV_restaurant.csv", second_features, threshold = 4.5)


	X_training, y_training, X_test, y_test = partition_data(X,y)



	c = determine_svm_hyperparameters(X_training, y_training)
	print "c chosen was %f" % (c)


	model = SVC(C=c, kernel = "rbf")
	model.fit(X_training, y_training)



	y_pred_test = model.predict(X_test)
	y_pred_train = model.predict(X_training)

	print_results(y_test, y_training, y_pred_test, y_pred_train)

	train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	train_f1_scores.append(train_f1)
	train_acc_scores.append(train_acc)
	train_prec_scores.append(train_prec)

	test_f1_scores.append(test_f1)
	test_acc_scores.append(test_acc)
	test_prec_scores.append(test_prec)



	#updated meta-data model.
	# print "\n Results for most updated meta model\n"

	# X, y = get_large_reviews_data("merged_NV_restaurant.csv", review_features, threshold = 4.5)

	# X_training, y_training, X_test, y_test = partition_data(X,y)



	# c = determine_svm_hyperparameters(X_training, y_training)
	# print "c chosen was %f" % (c)


	# model = SVC(C=c, kernel = "rbf")
	# model.fit(X_training, y_training)



	# y_pred_test = model.predict(X_test)
	# y_pred_train = model.predict(X_training)


	# print_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_acc, train_f1, train_prec = get_train_results(y_test, y_training, y_pred_test, y_pred_train)
	# test_acc, test_f1, test_prec = get_test_results(y_test, y_training, y_pred_test, y_pred_train)

	# train_f1_scores.append(train_f1)
	# train_acc_scores.append(train_acc)
	# train_prec_scores.append(train_prec)

	# test_f1_scores.append(test_f1)
	# test_acc_scores.append(test_acc)
	# test_prec_scores.append(test_prec)
	plot = False
	if plot:

		fig, ax = plt.subplots()
		ind = np.arange(len(test_acc_scores))
		width = 0.1

		# p1 = ax.bar(ind, train_acc_scores, width,alpha = 0.5, color = "maroon", label = "train accuracy")

		p2 = ax.bar(ind+width, train_f1_scores, width,  alpha = 0.5, color ="orange", label = "train f1 score")

		p3 = ax.bar(ind+2*width, train_prec_scores, width, alpha = 0.5, color = "green",label = "train precision")

		# p1 = ax.bar(ind+3*width, test_acc_scores, width, color = "maroon", label = "test accuracy")

		p2 = ax.bar(ind+3*width, test_f1_scores, width,  color ="orange", label = "test f1 score")

		p3 = ax.bar(ind+4*width, test_prec_scores, width,  color = "green",label = "test precision")

		X_train, y_train, X_test, y_test = get_tfidf_train_test("star_results.csv", threshold = 4.0)

		baseline = print_baseline_classifiers(X_train, y_train, X_test, y_test)
		ax.axhline(y=baseline, linestyle='dashed', color = 'orange', label = 'f1 baseline')
		ax.set_title("MetaData Model Results")
		ax.set_xticks((ind + 2.5*width) )
		ax.set_xticklabels(('Small # of Reviews', 'Large # of Reviews', 'Without # of Reviews'))

		ax.legend()


		plt.show()


	plot2 = True
	if plot2:
		test_f1_scores = [0.673295454545, 0.685365853659, 0.66568914956, 0.698550724638]
		test_f1_y_err = [.711538461538-0.632183908046, .722090261283-.64631043257, .706586826347 - .627329192547, .7354138 - .6567164]

		train_f1_scores = [0.855645545498,  0.737009123364,  0.987355110643]
		train_f1_y_err = [0.872131147541- 0.839980305268, 0.756756756757- 0.71716357776, 0.992207792208- 0.981897970378]

		train_precision = [0.772805507745, 0.58648989899, 0.975026014568,]
		train_precision_err = [0.797089041096-0.749143835616, 0.611601513241- 0.561557788945, 0.985216473073-0.963752665245]


		test_precision = [0.568345323741,  0.527204502814, 0.574683544304, .63925729443]
		test_precision_err = [ 0.614457831325- 0.518957345972,  0.568480300188- 0.480225988701,  0.618453865337-0.523690773067, .687830687831-.588709677419]

		ind = np.arange(len(train_f1_scores))
		width = .15

		plt.bar(ind, train_f1_scores, yerr = train_f1_y_err, width=width, capsize=7, color = 'orange', alpha = 0.5, label = "training f1 score")
		plt.bar(ind+width, test_f1_scores, yerr=test_f1_y_err, width = width, capsize=7, color='orange', label = "test f1 score")
		plt.bar(ind+2*width, train_precision, yerr = train_precision_err, width=width, capsize=7, color = 'green', alpha = 0.5, label = "training precision")
		plt.bar(ind+3*width, test_precision, yerr=test_precision_err, width = width, capsize=7, color='green', label = "test precision")


		X_train, y_train, X_test, y_test = get_tfidf_train_test("star_results.csv", threshold = 4.0)
		baseline_f1, baseline_precision = print_baseline_classifiers(X_train, y_train, X_test, y_test)
		plt.axhline(y=baseline_f1, linestyle='dashed', color = 'orange', label = 'f1 baseline')
		plt.axhline(y=baseline_precision, linestyle='dashed', color = 'green', label = 'precision baseline')		

		plt.xticks(ind+(3/2)*width, ['LogisticRegression', 'DecisionTree', 'RandomForest'])
		plt.legend()
		plt.title("Tfidf Model Results")
		plt.ylim(bottom=0, top=1)

		plt.show()
	 



if __name__ == "__main__" :
    main()
