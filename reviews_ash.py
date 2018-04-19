import numpy as np
import pandas as pd

from util_ash import *

import matplotlib.pyplot as plt

from nltk_processing import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# encoding=utf8
import sys

#old data

# # cleaning and merging the data for preprocessing
# restaurants = pd.read_csv("../yelp-dataset/merged_NV_restaurant.csv", encoding='utf-8', low_memory=False)
# restaurants = restaurants.dropna(axis=1, thresh=300)

# df_merged = pd.read_csv('../yelp-dataset/review_shortened_rest_NV.csv', encoding='utf-8', low_memory=False)


# grouped_reviews_long = df_merged.groupby(['business_id'])

# grouped_reviews = grouped_reviews_long['text'].apply(list)


# # now we have all the reviews for a single business as a huge review
# reviews = pd.DataFrame(grouped_reviews)
# review_list = reviews.text.apply(lambda u: u[0]).tolist()


# # clean the data for null values
# cleaned_list = [x for x in review_list if str(x) != 'nan']
# cleaned_df = pd.DataFrame(cleaned_list, columns=['review'])

# create the document term matrix, this is our X or feature matrix
# cv = CountVectorizer()


# ################################################################################
# first we load in the data and preprocess the reviews
# ################################################################################

# new dataset (restaurants with >100 reviews)
restaurants = pd.read_csv("../yelp-dataset/star_results.csv", encoding='utf-8', low_memory=False)

review_list = restaurants['text']


cleaned_list = [x for x in review_list if str(x) != 'nan']
cleaned_df = pd.DataFrame(cleaned_list, columns=['review'])

#update after preprocessing
cleaned_df = preprocessing(cleaned_df, 'review')


# ################################################################################
# # now we investigate the model's performance using TFIDF rather than count
# ################################################################################
tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english')

# get the training and test data ready
# # TODO: vary model hyperparameters and normalization
# # find best hyperparameters based on f1_score and precision
X = tfidf.fit_transform(cleaned_df['review']).toarray()
X = np.array(X)




################################################################################
# binary classification
################################################################################

#set the targets bassed on threshold of 4.5 stars
y = restaurants['stars'].apply(lambda u: 1 if u >= 4.5 else -1)

#get the training and test data ready
#TODO: kfold cv for train/test split
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state= 123)


# SVM CLASSIFICATION

# #one way of K-fold cross validation on smaller set

# ss = StratifiedShuffleSplit(test_size=0.9)
# C_vals = []
# g_vals = []
# kf = StratifiedKFold(shuffle=True)
# for train, _ in ss.split(X, y):
#     reduced_X = X[train]
#     reduced_y = y[train]
#     C, gamma = select_param_rbf(reduced_X,reduced_y,kf,metric="f1_score")
#     C_vals.append(C)
#     g_vals.append(gamma)

# print C_vals, g_vals


# #K-fold cross validation on whole dataset

#kf = StratifiedKFold(shuffle=True)

#C, gamma = select_param_rbf(X, y, kf, metric="f1_score")
#print 'found C: ', C
#print 'found gamma: ', gamma


#  #grid search validation for C and gamma
# parameters = {'C': C_vals, 'gamma': G_vals}

# svc = SVC()
# clf = GridSearchCV(svc, parameters)


# #still need to set gamma and C
# clf_svm = SVC(kernel='rbf', C=0.1, gamma=0.01)

# # learn the model
# clf_svm.fit(X_train, y_train)
# y_pred_test = clf_svm.predict(X_test)

# printScores(y_test, y_pred_test)


#LOGISTIC REGRESSION CLASSIFIER

# check for optimal hyperparameter

# one way of K-fold cross validation on smaller set

# ss = StratifiedShuffleSplit(n_splits = 8, test_size=0.2)
# C_vals = []
# kf = StratifiedKFold(shuffle=True)
# for train, _ in ss.split(X, y):
#     reduced_X = X[train]
#     reduced_y = y[train]
#     C= select_param_logReg(reduced_X,reduced_y,kf,metric="f1_score")
#     C_vals.append(C)

# print 'final c values', C_vals


# # add class weights
# percentage_min = np.count_nonzero(y == 1) / float(y.size)
# percentage_maj = 1 - percentage_min
# min_weight = percentage_maj/percentage_min
# print("weight that will be used: ", min_weight)

# # with weights
# #clf_logReg = LogisticRegression(C=10000, class_weight = {-1 : 1, 1 : min_weight})

# without weights
clf_logReg = LogisticRegression(C=10000)

# learn the model
clf_logReg.fit(X_train, y_train)
y_pred_test = clf_logReg.predict(X_test)

printScores(y_test, y_pred_test)


################################################################################
# multiclass classification
################################################################################

# round each label
y_multi = restaurants['stars'].apply(lambda u: round(u))

# get the training and test data ready
# TODO: kfold cv for train/test split
#X = np.array(X)
y_multi = np.array(y_multi)


X_train, X_test, y_train, y_test = train_test_split(X, y_multi, shuffle=True, random_state= 123)


# without weights
clf_logReg_multi = LogisticRegression(C=10000, multi_class='ovr')

# learn the model
clf_logReg_multi.fit(X_train, y_train)
y_pred_test = clf_logReg_multi.predict(X_test)
y_pred_train = clf_logReg_multi.predict(X_train)



print "Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(y_train, y_pred_train)
print "Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(y_test, y_pred_test)

#confusion matrix
print confusion_matrix(y_test, y_pred_test)


# num_classes = 5

# #generate OVA, OVO, and 2 random output codes 
# output_ova = generate_output_codes(num_classes, 'ova')
# output_ovo = generate_output_codes(num_classes, 'ovo')

# output_codes = [output_ova, output_ovo]
# output_names = ["ova", "ovo"]
# loss_funcs = ["hamming", "sigmoid", "logistic"]


# for i, output in enumerate(output_codes):
# 	for j, func in enumerate(loss_funcs):
# 		#create classifier
# 		multiSVM = MulticlassSVM(output, C=10.0, kernel='poly', degree=4, coef0=1.0, gamma=1)
# 		# fit the classifier 
# 		multiSVM.fit(X_train, y_train)


# 		# predict
# 		prediction = multiSVM.predict(X_train,loss_func=func)

# 		# count the number of errors
# 		count = metrics.zero_one_loss(prediction, y_test, normalize = False)
# 		print " %s and %s error is: %d" % (output_names[i] , loss_funcs[j], count)


