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
from sklearn.multiclass import OneVsRestClassifier
import sys
import mord as m


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

# round each label
y = restaurants['stars'].apply(lambda u: round(u))

# get the training and test data ready
# TODO: kfold cv for train/test split
#X = np.array(X)
y = np.array(y)

sum = 0
for i in y:
	if i ==2:
		sum += 1

print sum





# ################################################################################
# # now we investigate the model's performance using TFIDF rather than count
# ################################################################################


X_train, X_test, y_train, y_test = train_test_split(cleaned_df['review'],y, shuffle=True, random_state= 123)

tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english')
tfidf.fit(X_train)

X_train = tfidf.transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()



################################################################################
# multiclass classification
################################################################################


# without weights
#clf_logReg_multi = LogisticRegression(C=20000, multi_class='ovr')

# higher C --> less regularization (working better)
#clf_logReg_multi = LogisticRegression(C=200000000, multi_class='ovr')
clf_logReg_multi = LogisticRegression(C=8.5, multi_class='ovr')

# learn the model
clf_logReg_multi.fit(X_train, y_train)
y_pred_test = clf_logReg_multi.predict(X_test)
y_pred_train = clf_logReg_multi.predict(X_train)


print "Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(y_train, y_pred_train)
print "Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(y_test, y_pred_test)

print "f1 scores: "
print 


#confusion matrix
print 'train data: '
print confusion_matrix(y_train, y_pred_train)

print 'test data: '
print confusion_matrix(y_test, y_pred_test)


clf_svm_multi = OneVsRestClassifier(SVC(kernel='rbf', C=1000, gamma=0.01))

# higher C worked better than 1000
# clf_svm_multi = OneVsRestClassifier(SVC(kernel='rbf', C=5000, gamma=0.05))

# lower C is much worse 
# clf_svm_multi = OneVsRestClassifier(SVC(kernel='rbf', C=10, gamma=0.05))
# learn the model
clf_svm_multi.fit(X_train, y_train)
y_pred_test = clf_svm_multi.predict(X_test)
y_pred_train = clf_svm_multi.predict(X_train)

print "Multinomial SVM Train Accuracy : ", metrics.accuracy_score(y_train, y_pred_train)
print "Multinomial SVM Test Accuracy : ", metrics.accuracy_score(y_test, y_pred_test)

#confusion matrix
print 'train data: '
print confusion_matrix(y_train, y_pred_train)

print 'test data: '
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


################################################################################
# mord ordinal regression - could be an extension 
################################################################################

# still working on getting this code working

# clf2 = m.LogisticAT()
# clf2.fit(X_train, y_train)
# print('Mean Absolute Error of LogisticAT %s' % metrics.mean_absolute_error(clf2.predict(X_test), y_test))

# clf3 = m.LogisticIT(alpha=1.)
# clf3.fit(X_train, y_train)
# print('Mean Absolute Error of LogisticIT %s' %
#       metrics.mean_absolute_error(clf3.predict(X_test), y_test))

# clf4 = m.LogisticSE(alpha=1.)
# clf4.fit(X_train, y_train)
# print('Mean Absolute Error of LogisticSE %s' %
#       metrics.mean_absolute_error(clf4.predict(X_test), y_test))


