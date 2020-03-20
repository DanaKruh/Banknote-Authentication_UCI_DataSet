# Logistic Regression example of Iris data set
# author: d updated by sdm
import pandas as pd
from pml53 import plot_decision_regions  # plotting function
import matplotlib.pyplot as plt  # so we can add to the plot
from sklearn import datasets  # read the data sets
import numpy as np  # needed for arrays
from sklearn.model_selection import train_test_split  # splits database
from sklearn.preprocessing import StandardScaler  # standardize the data
from sklearn.svm import SVC  # the algorithm
from sklearn.metrics import accuracy_score  # grade the results
from sklearn.linear_model import Perceptron  # the algorithm
from sklearn.linear_model import LogisticRegression  # the algorithm
from sklearn.tree import DecisionTreeClassifier  # the algorithm
from sklearn.ensemble import RandomForestClassifier    # the algorithm
from sklearn.neighbors import KNeighborsClassifier     # the algorithm
from sklearn.tree import export_graphviz  # a cool graph

dvaDat = pd.read_table("data_banknote_authentication.txt", sep=",",
                       names=["variance", "skewness", "curtosis", "entropy", "class"])  # load the data set

X = dvaDat.iloc[:, [0, 1]].values  # separate the features we want
y = dvaDat.iloc[:, 4].values  # extract the classifications

# dvaDat = pd.read_table  # load the data set
# X = dvaDat.iloc[:, [0, 1]].values   # separate the features we want
# y = dvaDat.iloc[:, 4].values  # extract the classification
# split the problem into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()  # create the standard scalar
sc.fit(X_train)  # compute the required transformation
X_train_std = sc.transform(X_train)  # apply to the training data
X_test_std = sc.transform(X_test)  # and SAME transformation of test data!!!

##########                 Perceptron               #################

ppn = Perceptron(max_iter=4, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=True)
ppn.fit(X_train_std, y_train)  # do the training
print('-----------------------------Perceptron-----------------------------------------')
print('Perceptron -> Number in test ', len(y_test))
y_pred = ppn.predict(X_test_std)  # now try with the test data

# Note that this only counts the samples where the predicted value was wrong
print('Perceptron -> Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('Perceptron -> Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# NOTE the double parens!
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Perceptron -> Number in combined ', len(y_combined))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred = ppn.predict(X_combined_std)
print('Perceptron -> Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Perceptron -> Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# # now visualize the results
# plt.title("Perceptron")
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
#                       test_idx=range(len(X_train), len(X)))
# plt.xlabel('variance of Wavelet Transformed image [standardized]')
# plt.ylabel('skewness of Wavelet Transformed image  [standardized]')
# plt.legend(loc='upper left')
# plt.show()



##########                 Support Vector Machine               #################

# kernal - specify the kernal type to use
# C - the penalty parameter - it controls the desired margin size

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)  # do the training

y_pred = svm.predict(X_test_std)  # work on the test data
print('-----------------------------Support Vector Machine-----------------------------------------')

# show the results
print('SVC -> Number in test ', len(y_test))
print('SVC -> Misclassified samples: %d' % (y_test != y_pred).sum())
print('SVC -> Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# combine the train and test sets
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# and analyze the combined sets
print('SVC -> Number in combined ', len(y_combined))
y_combined_pred = svm.predict(X_combined_std)
print('SVC -> Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('SVC -> Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# # and visualize the results
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm, test_idx=range(105, 150))
# plt.title("Support Vector Machine")
# plt.xlabel('variance of Wavelet Transformed image [standardized]')
# plt.ylabel('skewness of Wavelet Transformed image  [standardized]')
# plt.legend(loc='upper left')
# plt.show()


#######################   LogisticRegression     #################################

lr = LogisticRegression(C=10, solver='liblinear', multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)  # apply the algorithm to training data
y_pred = lr.predict(X_test_std)  # work on the test data
# Note that this only counts the samples where the predicted value was wrong
print('-----------------------------LogisticRegression-----------------------------------------')
print('LogisticRegression -> Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('LogisticRegression -> Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
# combine the train and test data

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('LogisticRegression -> Number in combined ', len(y_combined))

# we did the stack so we can see how the combination of test and train data did
y_combined_pred = lr.predict(X_combined_std)
print('LogisticRegression -> Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('LogisticRegression -> Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
#
# # plot the results
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))
# plt.title("Logistic Regression")
# plt.xlabel('variance of Wavelet Transformed image [standardized]')
# plt.ylabel('skewness of Wavelet Transformed image  [standardized]')
# plt.legend(loc='upper left')
# plt.show()



#######################   DecisionTree     #################################
# create the classifier and train it
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
tree.fit(X_train_std, y_train)
y_pred = tree.predict(X_test_std)  # work on the test data
print('-----------------------------DecisionTree-----------------------------------------')
print('DecisionTree -> Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
print('DecisionTree -> Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# see how we do on the combined data
y_combined_pred = tree.predict(X_combined_std)
print('DecisionTree -> Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('DecisionTree -> Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# # and visualize it
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=tree,
#                       test_idx=range(105, 150))
# plt.title("Decision Tree")
# plt.xlabel('variance of Wavelet Transformed image [standardized]')
# plt.ylabel('skewness of Wavelet Transformed image  [standardized]')
# plt.legend(loc='upper left')
# plt.show()


################### Random Forest #######################
# This exports the file tree.dot. To view this file, on the Mac:
# Install graphviz: brew install graphviz
# NOTE: You may have to install brew first...
# Then execute: dot -T png -O tree.dot
# Then execute: open tree.dot.png
export_graphviz(tree, out_file='tree.dot',
                feature_names=['petal length', 'petal width'])

# create the classifier and train it
# n_estimators is the number of trees in the forest
# the entropy choice grades based on information gained
# n_jobs allows multiple processors to be used
forest = RandomForestClassifier(criterion='entropy', n_estimators=10,
                                random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

print('-----------------------------Random Forest-----------------------------------------')
y_pred = forest.predict(X_test)  # see how we do on the test data
print('Random Forest -> Number in test ', len(y_test))
print('Random Forest -> Misclassified samples: %d' % (y_test != y_pred).sum())

print('Random Forest -> Accuracy: %.2f' % accuracy_score(y_test, y_pred))  # check accuracy

# combine the train and test data

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Random Forest -> Number in combined ', len(y_combined))

# see how we do on the combined data
y_combined_pred = forest.predict(X_combined_std)
print('Random Forest -> Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Random Forest -> Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# # and visualize the results
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=forest,
#                       test_idx=range(105, 150))
# plt.title("Random Forest")
# plt.xlabel('variance of Wavelet Transformed image [standardized]')
# plt.ylabel('skewness of Wavelet Transformed image  [standardized]')
# plt.legend(loc='upper left')
# plt.show()



################### K-Nearest Neighbors #######################

# create the classifier and fit it
# using 10 neighbors
# since only 2 features, minkowski is same as euclidean distance
# where p=2 specifies sqrt(sum of squares). (p=1 is Manhattan distance)
knn = KNeighborsClassifier(n_neighbors=10,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)

# run on the test data and print results and check accuracy
y_pred = knn.predict(X_test_std)
print('-----------------------------K-Nearest Neighbors-----------------------------------------')
print('K-Nearest Neighbors -> Number in test ',len(y_test))
print('K-Nearest Neighbors -> Misclassified samples: %d' % (y_test != y_pred).sum())
print('K-Nearest Neighbors -> Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('K-Nearest Neighbors -> Number in combined ',len(y_combined))

# check results on combined data
y_combined_pred = knn.predict(X_combined_std)
print('K-Nearest Neighbors -> Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('K-Nearest Neighbors -> Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# # visualize the results
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn,
#                       test_idx=range(105,150))
# plt.xlabel('variance of Wavelet Transformed image [standardized]')
# plt.ylabel('skewness of Wavelet Transformed image  [standardized]')
# plt.title("K-Nearest Neighbors")
# plt.legend(loc='upper left')
# plt.show()