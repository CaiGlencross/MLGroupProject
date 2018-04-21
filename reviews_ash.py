import numpy as np
import pandas as pd

from util_ash import *
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

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
y = restaurants['stars'].apply(lambda u: 1 if u >= 4 else -1)

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
feature_names = np.array(tfidf.get_feature_names())

X_train = tfidf.transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()


n,d = X_train.shape
n2,d2 = X_test.shape

print 'number of training restaurants (examples): ', n
print 'number of testing restaurants (examples): ', n2
print 'number of features from tfidf train: ', d
print 'number of features from tfidf test: ', d2


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
#     C= select_param_logReg(reduced_X,reduced_y,kf,metric="f1_score")
#     C_vals.append(C)

# print 'final c values', C_vals

# #test from 1 to 10 and show graph (underfit --> always predicting negative, so scores all 0, then overfitting)
# #then try even finer resolution 
# determine_logreg_hyperparameters(X_train, y_train, plot=True, show=True)

print 'Using Logistic Regression...'

# # # add class weights
# # percentage_min = np.count_nonzero(y == 1) / float(y.size)
# # percentage_maj = 1 - percentage_min
# # min_weight = percentage_maj/percentage_min
# # print("weight that will be used: ", min_weight)

# # # with weights
# # clf_logReg = LogisticRegression(C=9, class_weight = {-1 : 1, 1 : min_weight})

# #without weights

#maximize precision
clf_logReg = LogisticRegression(C=2.5)

# learn the model
result = clf_logReg.fit(X_train, y_train)
y_pred_train = clf_logReg.predict(X_train)
y_pred_test = clf_logReg.predict(X_test)

sort_coef_tocsv(feature_names, clf_logReg.coef_, "sorted_clf_features_C2_5.csv")

printScores(y_train, y_pred_train, 'Logistic Regression for precision', train= True)
print "\t"
printScores(y_test, y_pred_test, 'Logistic Regression for precision')

confusion_matrix2 = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix2)

#maximize F1
clf_logReg2 = LogisticRegression(C=0.5)

# learn the model
result2 = clf_logReg2.fit(X_train, y_train)
y_pred_train2 = clf_logReg2.predict(X_train)
y_pred_test2 = clf_logReg2.predict(X_test)


printScores(y_train, y_pred_train2, 'Logistic Regression for F1', train= True)
print "\t"
printScores(y_test, y_pred_test2, 'Logistic Regression for F1')

confusion_matrix = confusion_matrix(y_test, y_pred_test2)
print(confusion_matrix)


sort_coef_tocsv(feature_names, clf_logReg2.coef_, "sorted_clf_features_C_0_5.csv")

################################################################
"""
# Decision Tree

print 'Using DecisionTreeClassifier...'

#determine_DT_hyperparameters(X_train, y_train, plot=True, show=True)

# # add class weights
# percentage_min = np.count_nonzero(y == 1) / float(y.size)
# percentage_maj = 1 - percentage_min
# min_weight = percentage_maj/percentage_min
# print("weight that will be used: ", min_weight)

#clf_DT = DTC(criterion='entropy', random_state=123, max_depth=5, class_weight = {-1 : 1, 1 : min_weight})
clf_DT = DTC(criterion='entropy', random_state=123, max_depth=5)
clf_DT.fit(X_train, y_train)
#print_tree(clf_DT.tree_, feature_names=tfidf.vocabulary_, class_names=["-1", "1"])

#print clf_DT.feature_importances_
y_pred_test = clf_DT.predict(X_test)
y_pred_train = clf_DT.predict(X_train)

#print 'y_pred = ', y_pred_test

printScores(y_train, y_pred_train, 'DecisionTreeClassifier', train= True)
print "\t"
printScores(y_test, y_pred_test, 'DecisionTreeClassifier')

# confusion_matrix3 = confusion_matrix(y_test, y_pred_test)
# print(confusion_matrix3)

################################################################

#Random Forest

print 'Using Random Forest...'

# n_est_range = [30,32,34, 36,38, 40,42,44,46]
# max_depth_range = [40,42,44,46,48,50,52,54,56,58, 60]
# parameters = {'n_estimators': n_est_range, 'max_depth': max_depth_range}

# random_forest = RandomForestClassifier(criterion="entropy")
# grid = GridSearchCV(random_forest, parameters,return_train_score=True, scoring='f1_weighted')

# X_cv, _, y_cv, _= train_test_split(X_train, y_train, test_size=0.2) 

# grid.fit(X_cv, y_cv)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))


# test_scores = grid.cv_results_['mean_test_score'].reshape(len(n_est_range),len(max_depth_range))
# print "test scores: ", test_scores

# # Draw heatmap of the validation accuracy as a function of number of estimators and max_depth
# #
# # The score are encoded as colors with the hot colormap which varies from dark
# # red to bright yellow. As the most interesting scores are all located in the
# # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# # as to make it easier to visualize the small variations of score values in the
# # interesting range while not brutally collapsing all the low score values to
# # the same color.

# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# # plt.imshow(test_scores, interpolation='nearest', cmap=plt.cm.hot,
# #            norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
# plt.imshow(test_scores, interpolation='nearest', cmap=plt.cm.hot,
#            norm=MidpointNormalize(vmin=0.55, midpoint=0.62))
# plt.xlabel('Max Depth')
# plt.ylabel('Number of Estimators')
# plt.colorbar()
# plt.xticks(np.arange(len(max_depth_range)), max_depth_range, rotation=45)
# plt.yticks(np.arange(len(n_est_range)), n_est_range)
# plt.title('Cross Validation F1 Score')
# plt.show()



clf_RF = RandomForestClassifier(criterion='entropy', n_estimators= 37, max_depth=48)
clf_RF.fit(X_train, y_train)

y_pred_test = clf_RF.predict(X_test)
y_pred_train = clf_RF.predict(X_train)

printScores(y_train, y_pred_train, 'RandomForestClassifier', train= True)
print "\t"
printScores(y_test, y_pred_test, 'RandomForestClassifier')

confusion_matrix4 = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix4)
"""



