from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from patsy import dmatrices
from sklearn import tree
import matplotlib.dates as mdates
import random
from sklearn.metrics import accuracy_score
from scipy.stats.kde import gaussian_kde
from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt


class Analytics:

	def __init__(self, df):
		self.new_df = df

	def test_train_data(self):

		y, X = dmatrices(
			'trump_clinton_win ~ state + '
			'polling_weight + trump_adjusted + clinton_adjusted',
			self.new_df, return_type="dataframe")
		y = np.ravel(y)
		X_train, X_test, y_train, y_test = train_test_split(X,
															y,
															test_size=0.3,
															random_state=0)
		return X, y, X_train, X_test, y_train, y_test

	def logistic_model(self, X_train, y_train, X_test):
		print 'Modelling with logistic regression \n'
		model2 = LogisticRegression()
		model2.fit(X_train, y_train)
		predicted_regression = model2.predict(X_test)
		probs = model2.predict_proba(X_test)
		print ' \n Done Modelling with logistic regression \n'
		return model2, predicted_regression, probs

	def accuracy(self, y_test,logistic_model_predicted,logistic_model_prob,
	             decision_tree_model_predicted,decision_tree_model_prob):
		print " \n Doing Accuracy measurement with both models \n"

		print 'logistic_model classification_report', metrics.classification_report(
			y_test, logistic_model_predicted)
		print 'logistic_model for clinton_probability' , logistic_model_prob[:, 1]
		print 'logistic_model for trump_probability' , logistic_model_prob[:, 0]

		print 'decision_model classification_report', metrics.classification_report(
			y_test, decision_tree_model_predicted)

		print 'logistic_model for clinton_probability' , decision_tree_model_prob[:, 1]
		print 'logistic_model for trump_probability' , decision_tree_model_prob[:, 0]

		print " \n Done with  Accuracy measurement with both models \n"

	def decision_tree_model(self, X_train, y_train, X_test):
		print " Modelling with decision tree"
		'''
		 Help from here
		 http://scikit-learn.org/stable/modules/tree.html
		'''
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(X_train, y_train)
		predicted_decision_tree = clf.predict(X_test)
		probs_decision_tree = clf.predict_proba(X_test)
		print " \n Done with Modelling with decision tree \n"
		return clf, predicted_decision_tree, probs_decision_tree

	def cross_validation(self, logistic_model, X_test, y_test, decison_model):
		'''
		#Another way http://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/
		'''
		print "\nFinding the cross validation scores\n"

		scores = cross_val_score(logistic_model, X_test, y_test, scoring='accuracy',
		                         cv=5)
		print '\n for logictic regression after 5-fold cross-validation', scores, scores.mean()
		kfold = KFold(n=len(X_test), n_folds=5)
		results = cross_val_score(logistic_model, X_test, y_test, cv=kfold)
		print("Accuracy for LogisticRegression: %.3f%% (%.3f%%)") % (
		results.mean() * 100.0, results.std() * 100.0)

		scores = cross_val_score(decison_model,
									X_test,
									y_test,
									scoring='accuracy',
									cv=5)
		print '\n for decision tree after 5-fold cross-validation', \
			scores, \
			scores.mean()
		kfold = KFold(n=len(X_test), n_folds=5)
		results = cross_val_score(decison_model, X_test, y_test, cv=kfold)
		print("Accuracy for decision tree: %.3f%% (%.3f%%)") % (
		results.mean() * 100.0, results.std() * 100.0)
		print "\n======= Done with Cross validation ==========\n"

	def draw_precision(self, logistic_model_prob, decision_model_prob,
	                   logistic_model_pred):
		fig = plt.figure()
		kde = gaussian_kde(logistic_model_prob[:, 1])
		dist_space = linspace(min(logistic_model_prob[:, 1]),
		                      max(logistic_model_prob[:, 1]), 100)
		plt.xlabel('logistic probability distribution')
		plt.ylabel('clinton win probability')
		plt.plot(dist_space, kde(dist_space))
		plt.show()

		fig = plt.figure()
		kde = gaussian_kde(decision_model_prob[:, 1])
		dist_space = linspace(min(decision_model_prob[:, 1]),
		                      max(decision_model_prob[:, 1]), 100)
		plt.xlabel('decision tree probability distribution')
		plt.ylabel('clinton win probability')
		plt.plot(dist_space, kde(dist_space))
		plt.show()