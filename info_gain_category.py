
# coding: utf-8

# author: Katie
#
#
# Purpose: find the most important categories in this dataset
# through decision stumps
# Use information gain as the metric

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# cleaning and merging the data for preprocessing
#TODO: change this data for your own reading pleasure
df = pd.read_csv("../data/yelp-dataset/categorized_data.csv")


#TODO: @Cai --- is this what you use for your model?..
df['stars'] = df['stars'].apply(lambda u: 1 if u >= 4.5 else 0)

OFF_SET = 96 #start column
END_COLUMN = len(df.columns)
print(df.columns[OFF_SET]) # the hand-hard-coded column which is the category
print(df.columns[END_COLUMN - 1]) #manually check that you're good for the categories...not ideal oh well

column_range = range(OFF_SET, END_COLUMN)
category_names = df.columns[column_range]
X = df.iloc[:,column_range].values
y = df.stars.values


## ENTROPY CODE ##
################################################################################
def split_data(X, y, feature) :
    """
    Split dataset (X,y) into two datasets (X1,y1) and (X2,y2)
    based on feature and threshold.

    (X1,y1) contains the subset of (X,y) such that X[i,feature] <= threshold.
    (X2,y2) contains the subset of (X,y) such that X[i,feature] > threshold.

    Parameters
    --------------------
        X         -- numpy array of shape (n,d), samples
        y         -- numpy array of shape (n,), target classes
        feature   -- int, feature index to split on
        threshold -- float, feature threshold

    Returns
    --------------------
        X1        -- numpy array of shape (n1,d), samples
        y1        -- numpy array of shape (n1,), target classes
        X2        -- numpy array of shape (n2,d), samples
        y2        -- numpy array of shape (n2,), target classes
    """
    n, d = X.shape
    if n != len(y) :
        raise Exception("feature vector and label vector must have same length")

    X1, X2 = [], []
    y1, y2 = [], []
    ### ========== TODO : START ========== ###

    #TODO: filter on the matching indicies where feature level is gt or lt
    X1 = X[X[:,feature] == 0,:]
    y1 = y[X[:,feature] == 0]

    X2 = X[X[:,feature] != 0, :]
    y2 = y[X[:,feature] != 0]

    ### ========== TODO : END ========== ###
    X1, X2 = np.array(X1), np.array(X2)
    y1, y2 = np.array(y1), np.array(y2)

    return X1, y1, X2, y2


def entropy(y) :
    """
    Compute entropy.

    Parameters
    --------------------
        y -- numpy array of shape (n,), target classes

    Returns
    --------------------
        H -- entropy
    """

    # compute counts
    _, counts = np.unique(y, return_counts=True)

    ### ========== TODO : START ========== ###
    # part a: compute entropy
    # hint: use np.log2 to take log
    H = 0
    total_sum = np.sum(counts)
    for c in counts:
        prob_y = float(c)/total_sum
        H += -1.0* prob_y * np.log2(prob_y)
    ### ========== TODO : END ========== ###

    return H

def calc_information_gain(X, y):
    """
    Calculates the conditional entropies

    Parameters
    --------------------
        X         -- numpy array of shape (n,d), samples
        y         -- numpy array of shape (n,), target classes
        feature   -- int, feature index to split on

    Returns
    --------------------
        info_gains -- numpy array filled with information gain - splitting at each category

    """
    N, d = X.shape

    info_gains = np.empty(d)    # associated conditional entropies
    H_entropy_y = entropy(y) # get the total entropy for information gain calc.

    # calculate conditional entropy for each feature (one per column)
    for i in range(d):
        X1, y1, X2, y2 = split_data(X, y, i)
        H_1 = entropy(y1)
        H_2 = entropy(y2)
        H_cond = (float(len(X1))/N)*H_1 + (float(len(X2))/N)*H_2
        info_gain = H_entropy_y - H_cond
        info_gains[i] = info_gain
    return info_gains


###############################################################################
ig = calc_information_gain(X,y)

feature_names = category_names.tolist() #get the
d = {'IG': ig, 'names': feature_names}
stump_df = pd.DataFrame(data = d)
stump_df_sorted = stump_df.sort_values('IG', ascending=False)



# Now that we have a dataframe with the information gain for each category,
# and it is sorted in descending order for the information gain,
#it's time to get the indicies that matter the most


def get_important_categories_index(df_ig_sorted, n):
    """
    Parameters
    ------------
    df_info_gain: the information gained within the dataframe. Assumes that is sorted.
    n: top number of categories

    Returns
    ------
    top_n : list for indicies of the top n columns
    """
    top_n = stump_df_sorted.index[0:n].tolist()
    return top_n

def get_important_categories_name(df_ig_sorted, n):
    """
    Parameters
    ------------
    df_info_gain: the information gained within the dataframe. Assumes that is sorted.
    n: top number of categories

    Returns
    ------
    top_n : a list of the top information_gain columns
    """
    top_n = stump_df_sorted.names.iloc[0:n].tolist()
    return top_n


# set the arbitrary number of categories that are the best
N_CATEGORIES = 50

index = get_important_categories_index(stump_df_sorted, N_CATEGORIES)
index_original_df = list(map(lambda u: u+ OFF_SET, index)) # this list corresponds to the columns in original df, "df"


name_important_columns = get_important_categories_name(stump_df_sorted,N_CATEGORIES)

print(name_important_columns)


maps_correctly = df.columns[index_original_df ].tolist() == name_important_columns # sanity check -- these should be equal
print("maps correctly: ")
print(maps_correctly)
