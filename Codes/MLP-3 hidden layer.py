    # -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:30:52 2017

@author: SainiD
"""
# Importing the libraries
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


# Importing the dataset
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')


# Splitting the dataset into the Training set and Test set
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index1 = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index1], y_train[shuffle_index1]
shuffle_index2=np.random.permutation(10000)
X_test, y_test = X_test[shuffle_index2], y_test[shuffle_index2]


# Fitting the 2-hidden layer MLPClassifier to the Training set
from sklearn.neural_network import MLPClassifier
#Adding a third hidden layer of 25 neurons
mlp_clf = MLPClassifier(hidden_layer_sizes=(100,50,25), activation='logistic', 
                        learning_rate_init=0.001, batch_size='auto', 
                        verbose= bool, max_iter=20, random_state=1)
mlp_clf.fit(X_train,y_train)


# Predicting the Test Set Results
y_predict=mlp_clf.predict(X_test)
print('\nPredicted Labels for Test Images: ', y_predict)


#Performing K-fold cross-validation on the Classifier
from sklearn.model_selection import cross_val_score
crs_vld=cross_val_score(mlp_clf, X_train, y_train, cv=3, scoring="accuracy")
print('\nThe Cross-Validation score is', crs_vld)


# Show the Test Images with Original and Predicted Labels
a = np.random.randint(1,50,20)
for i in a:
	two_d = (np.reshape(X_test[i], (28, 28)) * 255).astype(np.uint8)
	plt.title('Original Label: {0}  Predicted Label: {1}'.format(y_test[i],y_predict[i]))
	plt.imshow(two_d, interpolation='nearest',cmap='gray')
	plt.show()


# Plot Confusion Matrix for Test Data
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)


#Calculating Classification Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_predict)
print('\nThe Models Classification Accuracy is', accuracy)


#Calculating Classification Error
error=1 - accuracy
print('\nThe Models Classification Error is',error)