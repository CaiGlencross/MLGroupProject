import numpy as np
import pandas as pd

from util_ash import *

import matplotlib.pyplot as plt

from nltk_processing import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sys



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

#set the targets based on threshold of 4.5 stars
y = restaurants['stars'].apply(lambda u: 1 if u >= 4.5 else -1)

#get the training and test data ready
#TODO: kfold cv for train/test split
y = np.array(y)

sum= 0
for i in y:
	if i == 1:
		sum +=1

print 'number of positive reviews: ', sum

################################################################################
# binary classification
# now we investigate the model's performance using TFIDF rather than count
# ################################################################################

X_train, X_test, y_train, y_test = train_test_split(cleaned_df['review'],y, shuffle=True, random_state= 123)


tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english')

tfidf.fit(X_train)

X_train = tfidf.transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()


n,d = X_train.shape
n2,d2 = X_test.shape

print 'number of training restaurants (examples): ', n
print 'number of testing restaurants (examples): ', n2
print 'number of features from tfidf train: ', d
print 'number of features from tfidft test: ', d2




# can also use stratified split with 1 fold
#X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state= 123)


################################################################################
# SVM CLASSIFICATION

# #one way of K-fold cross validation on smaller set

# ss = StratifiedShuffleSplit(n_splits = 4, test_size=0.2)
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


# #still need to set gamma and C from Jack's findings
# clf_svm = SVC(kernel='rbf', C=1000, gamma=0.05)

# # learn the model
# clf_svm.fit(X_train, y_train)
# y_pred_train = clf_svm.predict(X_train)
# y_pred_test = clf_svm.predict(X_test)

# printScores(y_train, y_pred_train, 'SVM', train = True)
#print "\t"
# printScores(y_test, y_pred_test, 'SVM')

#confusion_matrix = confusion_matrix(y_test, y_pred_test)
#print(confusion_matrix)

################################################################

#LOGISTIC REGRESSION CLASSIFIER

# check for optimal hyperparameter

# one way of K-fold cross validation on smaller set

# ss = StratifiedShuffleSplit(n_splits = 6, test_size=0.2)
# C_vals = []
# kf = StratifiedKFold(shuffle=True)
# for train, _ in ss.split(X_train, y_train):
#     reduced_X = X_train[train]
#     reduced_y = y_train[train]
#     C= select_param_logReg(reduced_X,reduced_y,kf,metric="precision")
#     C_vals.append(C)

# print 'final c values', C_vals

##test from 1 to 10 and show graph (underfit --> always predicting negative, so scores all 0, then overfitting)
##then try even finer resolution 
#determine_logreg_hyperparameters(X_train, y_train, plot=True, show=True)

print 'Using Logistic Regression...'

# # add class weights
# percentage_min = np.count_nonzero(y == 1) / float(y.size)
# percentage_maj = 1 - percentage_min
# min_weight = percentage_maj/percentage_min
# print("weight that will be used: ", min_weight)

# # with weights
# clf_logReg = LogisticRegression(C=9, class_weight = {-1 : 1, 1 : min_weight})

#without weights

clf_logReg = LogisticRegression(C=9)

# learn the model
result = clf_logReg.fit(X_train, y_train)
y_pred_train = clf_logReg.predict(X_train)
y_pred_test = clf_logReg.predict(X_test)


printScores(y_train, y_pred_train, 'Logistic Regression', train= True)
print "\t"
printScores(y_test, y_pred_test, 'Logistic Regression')

confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)


################################################################

# Decision Tree

print 'Using DecisionTreeClassifier...'

#determine_DT_hyperparameters(X_train, y_train, plot=True, show=True)

# add class weights
percentage_min = np.count_nonzero(y == 1) / float(y.size)
percentage_maj = 1 - percentage_min
min_weight = percentage_maj/percentage_min
print("weight that will be used: ", min_weight)

#clf_DT = DTC(criterion='entropy', random_state=123, max_depth=40, class_weight = {-1 : 1, 1 : min_weight})
clf_DT = DTC(criterion='entropy', random_state=123, max_depth=40)
clf_DT.fit(X_train, y_train)
#print_tree(clf_DT.tree_, feature_names=tfidf.vocabulary_, class_names=["-1", "1"])

#print clf_DT.feature_importances_
y_pred_test = clf_DT.predict(X_test)
y_pred_train = clf_DT.predict(X_train)

#print 'y_pred = ', y_pred_test

printScores(y_train, y_pred_train, 'DecisionTreeClassifier', train= True)
print "\t"
printScores(y_test, y_pred_test, 'DecisionTreeClassifier')

confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)




