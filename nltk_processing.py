import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import io
import os
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from numpy import *
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.moses import MosesDetokenizer



from string import punctuation

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def preprocessing(data, row_name):
	"""
    Reads reviews in from csv and preprocesses the
    
    Parameters
    --------------------
        data  -- data frame 
        row_name -- name of row containing reviews
    
    Returns
    --------------------
        df -- dataframe with preprocessed reviews
    """

	#create tokenizer
	tokenizer = RegexpTokenizer(r'\w+')

	# create English stop words list
	stop_words = set(stopwords.words('english'))

	# Create p_stemmer of class PorterStemmer
	p_stemmer = PorterStemmer()

	# create list to store preprocessed text
	new_data = []

	for index, row in data.iterrows():
		# lower case the text 
		lower_case = row[row_name].lower()

		#tokenize the text (removes punctuation)
		tokens = tokenizer.tokenize(lower_case)

		# remove stop words from tokens
		stopped_tokens = [i for i in tokens if not i in stop_words]

		#stemming
		stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

		#put it back into a string
		detokenizer = MosesDetokenizer()

		detokenized_text = detokenizer.detokenize(stemmed_tokens, return_str=True)
		new_data.append(detokenized_text)

	#data frame of reviews
	df = pd.DataFrame({'review':new_data})
	return df



def main():
	data = pd.read_csv('../dummy_restaurants.csv', header=0)

	df = preprocessing(data, 'review')
	print df['review'][0]




if __name__ == "__main__":
    main()