import pandas as pd
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
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

restaurants = pd.read_csv("resturants.csv")

nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
#features with a solid amount of non-null values
g_restaurants = nan_restaurants[["stars","review_count", "BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]]
#only use restuarants that have every one of those features
clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
#change the booleans to integers
int_restaurants_clean = clean_restaurants.applymap(convert_booleans)
#seperate data and labels
restaurant_data = int_restaurants_clean[["review_count", "BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]]
restaurant_labels = int_restaurants_clean[["stars"]]
#use binary labels
#TODO: figure out procedurally what the best threshold is
threshold = 4.5

restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)

#turn pandas dataframe into a numpy array
X = restaurant_data.values
y = restaurant_labels.values
y = np.ravel(y)

#normalize our data so it has zero mean and a standard deviation of 1
#no need to pre process labels as they are binary
#X = preprocessing.scale(X)



n, d = X.shape

train_partition = int(math.floor(.8 * n ))

X_training = X[:train_partition]
y_training = y[:train_partition]

X_test = X[train_partition:]
y_test = y[train_partition:]

#SVM model
C_vals = [.001, .01, .1, .2,.3, 1.0, 1.5, 2.0, 2.5]
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
		X_train, X_val = X[train_index], X[val_index]
		y_train, y_val = y[train_index], y[val_index]
		model.fit(X_train, y_train)
		y_pred = model.predict(X_val)

		avg_accs.append(model.score(X_val, y_val))
		avg_f1s.append(f1_score(y_val, y_pred))
		avg_aurocs.append(roc_auc_score(y_val, y_pred))
		avg_precs.append(precision_score(y_val, y_pred))


		# count_ones = 0
		# for y_ex in y_val:
		# 	if y_ex ==1:
		# 		count_ones+=1
		# #print "count ones for validation data: %f" % (1-(count_ones/float(n)))
		# y_pred = model.predict(X_val)

		#avg_percentage += (1-(count_ones/float(n)))
	print "average accuracy for %f k-fold validation was %f" % (c, np.mean(avg_accs))
	c_avg_accs.append(np.mean(avg_accs))
	c_avg_f1.append(np.mean(avg_f1s))
	c_avg_auroc.append(np.mean(avg_aurocs))
	c_avg_prec.append(np.mean(avg_precs))
	print "average distribution is %f" % (avg_percentage/float(10))

plt.figure()
plt.xlabel("C values")
plt.ylabel("metric score")
plt.plot(C_vals, c_avg_accs, label="Accuracy")
plt.plot(C_vals, c_avg_f1, label="F1_Score")
plt.plot(C_vals, c_avg_auroc, label="AUROC")
plt.plot(C_vals, c_avg_prec, label="Precision")
plt.legend()
plt.show()

model.fit(X_training, y_training)

print "coefficients after using all of training data:"
#print "\treview count: ", model.coef_[0][1]
#print "\tBusinessParking_street: ", model.coef_[0][1]
#print "\tBusinessParking_lot: ", model.coef_[0][2]
#print "\tGoodForMeal_dinner: ", model.coef_[0][3]
#print "\tGoodForMeal_lunch: ", model.coef_[0][4]
#print "\tGoodForMeal_breakfast: ", model.coef_[0][5]






