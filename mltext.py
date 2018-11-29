from __future__ import print_function

import pandas as pd
#Set the output width in consol
pd.set_option("display.max_columns", 8)

#*************************************************************
#Part 1: Model building in scikit-learn (Supervised Learning)
#*************************************************************

#Load the iris dataset as an example
from sklearn.datasets import load_iris
iris = load_iris()

#Store the feature matrix(X) and response vector(y)
X = iris.data
y = iris.target

#Check the shapes of X and y
#print(X.shape)
#print(y.shape)

"""
#output
(150, 4) => 150 x 4 observations which are also known as samples, instances or records
(150,)
"""
#Examine the first 5 rows of the feature matrix
#print(pd.DataFrame(X, columns=iris.feature_names).head())

"""
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2
[5 rows x 4 columns]
"""

#Examine the response vector
#print(y)
"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""

#In order to build a Model, the features must be numeric, and every observation must have same features in the same
# order

#4-Step modeling process
#Import, Instantiate, Fit, Predict

#Import the class
from sklearn.neighbors import KNeighborsClassifier

#Instantiate the model (with default parameters)
knn = KNeighborsClassifier()

#Fit the model with data (occurs in-place)
#It is learning the relationship between X and y (features and response)
knn.fit(X,y)


#In order to make a prediction, the new observation must have the same features as the training observations, both in
# number and meaning.

#Predict the response for a new observation
knn.predict([[3,5,4,2]])


#********************************************
#Part 2: Representing text as numerical data
#********************************************

#Example text for Model training (SMS messages)
simple_train = ['call you tonight', 'Call me a cab', 'please call me..Please']

"""
From the scikit-learn documentation:

Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols 
cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size
rather than raw text documents with variable length. 
"""

#We will use CountVectorizer() to convert text into a matrix of token counts.

#Import and instantiate CountVectorizer (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

#Learn the vocabulary of the training data (occurs in-place)
vect.fit(simple_train)

#Examine the fitted vocabulary
vect.get_feature_names()

"""
['cab', 'call', 'me', 'please', 'tonight', 'you']
"""

#Transform training data into a "Document-term matrix"
simple_train_dtm = vect.transform(simple_train)
#print(simple_train_dtm)
"""
  (0, 1)	1
  (0, 4)	1
  (0, 5)	1
  (1, 0)	1
  (1, 1)	1
  (1, 2)	1
  (2, 1)	1
  (2, 2)	1
  (2, 3)	2
"""

#Convert sparse matrix (matrix with non-zero values with their coordinates) to a dense matrix (matrix with all values)
simple_train_dtm.toarray()

"""
[[0 1 0 0 1 1]
 [1 1 1 0 0 0]
 [0 1 1 2 0 0]]
"""

#Examine the vocabulary and document-term matrix together
pd.DataFrame(simple_train_dtm.toarray(),columns=vect.get_feature_names())

"""
   cab  call  me  please  tonight  you
0    0     1   0       0        1    1
1    1     1   1       0        0    0
2    0     1   1       2        0    0
"""

"""
In this scheme, features and samples are defined as follows:

=> Each individual token occurrence frequency is treated as a Feature.
=> The vector of all the token frequencies for a given document is considered a multivarient Sample. 


A corpus of documents can thus be represented by a matrix with one row per document and one column per token 
(e.g. word) occurring in the corpus.

This process of counting vectorizers is called Vectorization. It is a process of converting a collection of documents 
into numeric feature vectors.
"""

#Check the datatype of the document-term matrix

type(simple_train_dtm)

"""
<class 'scipy.sparse.csr.csr_matrix'>
"""

