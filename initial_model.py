import pandas as pd
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.model_selection import KFold

restaurants = pd.read_csv("resturants.csv")

nan_restaurants = restaurants.applymap(lambda x: np.nan if x == "Na" else x)
#features with a solid amount of non-null values
g_restaurants = restaurants[["stars","review_count", "BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]]
#only use restuarants that have every one of those features
clean_restaurants = g_restaurants.dropna(how="any", axis = 0)
#change the booleans to integers
int_restaurants_clean = clean_restaurants.applymap(lambda x: 0 if x == "FALSE" else 1)
#seperate data and labels
restaurant_data = int_restaurants_clean[["review_count", "BusinessParking_street", "BusinessParking_lot", "GoodForMeal_dinner", "GoodForMeal_lunch", "GoodForMeal_breakfast"]]
restaurant_labels = int_restaurants_clean[["stars"]]
#use binary labels
#TODO: figure out procedurally what the best threshold is
threshold = 3.5
print restaurant_labels.head()
restaurant_labels = restaurant_labels.applymap(lambda x: 1 if x >=threshold else 0)
print restaurant_labels.head()
#turn pandas dataframe into a numpy array
X = restaurant_data.values
y = restaurant_labels.values
y = np.ravel(y)

print y

n, d = X.shape

train_partition = int(math.floor(.8 * n ))

X_training = X[:train_partition]
y_training = y[:train_partition]

X_test = X[train_partition:]
y_test = y[train_partition:]

#SVM model
model = SVC(C=1, kernel = "rbf")

#k fold validation training
avg_acc = 0
kf = KFold(n_splits = 10)
for train_index, val_index in kf.split(X_training):
	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]
	model.fit(X_train, y_train)
	avg_acc += model.score(X_val, y_val)

print "average accuracy for this k-fold validation was ", avg_acc/10





