import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from mpl_toolkits.mplot3d import Axes3D
from patsy import dmatrices
from sklearn import tree
import matplotlib.dates as mdates
import random
from sklearn.metrics import accuracy_score
from scipy.stats.kde import gaussian_kde
from numpy import linspace



#convert presidential_polls.csv to dataframe format named df
def read_data(csv_file):
	df = pd.read_csv(csv_file)
	#print list(df.columns.values)
	return df

#visualizing the dataframe df where date is converted to
#date format'%m/%d/%y' and is appended in list as dates.
def draw_visualization(df):
	'''
	 Answer for :
	 What are the trends of the polls over time (by month)? Present visualization.
	'''
	dates =  []
	'''
	 Converting date to datetime Datetime format 
	''' 
	for v in df.createddate:
		date_tmp = datetime.datetime.strptime(v, '%m/%d/20%y').date()
		dates.append(date_tmp)

	# adjustment data of Clinton and Trump
	trump_adj = df.adjpoll_trump
	clinton_adj = df.adjpoll_clinton
	
	fig, ax = plt.subplots(1,1)
	'''
	 for scatter plot use ax.scatter
	 for bar plot use ax.bar
	''' 
	ax.scatter(dates, clinton_adj, color='b')
	ax.xaxis.set_major_locator(mdates.MonthLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.20%y'))
	ax.set_xlabel('Poll end date')
	ax.set_ylabel('clinton adjusted')
	plt.show()
	fig, ax = plt.subplots(1,1)
	ax.scatter(dates, trump_adj, color='r')
	ax.xaxis.set_major_locator(mdates.MonthLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.20%y'))
	ax.set_xlabel('Poll end date')
	ax.set_ylabel('trump adjusted')
	plt.show()

	'''
	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	ax2.bar(dates, trump_adj, color='r')
	ax2.xaxis.set_major_locator(mdates.MonthLocator())
	ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.20%y'))
	ax1.bar(dates, clinton_adj, color='g')
	ax1.xaxis.set_major_locator(mdates.MonthLocator())
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.20%y'))

	ax1.set_ylabel('clinton adjusted data')
	ax2.set_ylabel('trump adjusted data')
	plt.show()
	'''
	
#preprocessing data
	
def clean_prepare_data(df):

	clinton_error_data = []
	trump_error_data = []

	adj_clinton = df['adjpoll_clinton'].tolist()
	raw_clinton = df['rawpoll_clinton'].tolist()

	adj_trump = df['adjpoll_trump'].tolist()
	raw_trump = df['rawpoll_trump'].tolist()

	'''
	 Going through both columns of adj_clinton, raw_clinton
	 and storing the error in actual and trump_adjusted
	''' 

	for k,v in zip(adj_clinton, raw_clinton):
		clinton_error_data.append(k-v)

	for k,v in zip(adj_trump, raw_trump):
		trump_error_data.append(k-v)

	'''
	fig=plt.figure()
	plt.plot(raw_clintion,clinton_error_data,color='black')
	plt.xlabel('This is the x axis')
	plt.ylabel('This is the y axis')
	plt.show()
	'''

	'''
	colors = np.random.rand(len(clinton_error_data))
	plt.scatter(raw_clintion, clinton_error_data,  c=colors)
	plt.xlabel('raw_clintion')
	plt.ylabel('clinton_error_data')
	plt.show()
	'''

	#finding out the number of days between startdate and today's date
	enddate_list = df['startdate'].tolist()
	days_from_poll = []


	TODAY = datetime.date(year=2016, month=12, day=12)
	for v in enddate_list:
		dates = datetime.datetime.strptime(v, '%m/%d/20%y').date()
		days_from_poll.append((TODAY - dates).days)

	'''
	colors = np.random.rand(len(clinton_error_data))
	plt.scatter(days_from_poll, clinton_error_data,  c=colors)
	plt.xlabel('days_from_poll')
	plt.ylabel('clinton_error_data')
	plt.show()
	'''

	binomial_data = []
	clinton_win = 1
	trump_win = 0


	#Comparing adj_clinton and adj_trump, if it is greater than 0 then Clinton wins
	#and if adj_clinton and adj_trump is less than 0 then Trump wins
	for k,v in zip(adj_clinton, adj_trump):
		if k - v >= 0:
			binomial_data.append(clinton_win)
		else:
		    binomial_data.append(trump_win)	

	#Creating weighted_data list
	weighted_data = []

	'''
		colors = np.random.rand(2)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(clinton_error_data, binomial_data, days_from_poll, c= 'r', marker='o')

		ax.set_xlabel('clinton_error_data')
		ax.set_ylabel('winner')
		ax.set_zlabel('days from poll')
		plt.show()

	'''
	'''
	 set is used to remove all duplicate state names
	 now we assign each state name present here a number
	 eg :  {'Wisconsin': 43, 'Mississippi': 0, 'Washington': 49 }
	''' 
	index = 0
 	states_map = {}
	for st in set(df.state):
		states_map[st] = index
		index += 1
	state_data = []

	#print states_map
	'''
	 Creating state table with the map
	'''
     
	for st in df.state:
		state_data.append(states_map[st])	

	'''
	 Creating a weighted data with samplesize, days from poll ,
	 polling weight
	''' 

	count = 0
	for weight in df.poll_wt:
		tmp_weight = weight
		tmp_weight *= df.samplesize[count]
		tmp_weight /= days_from_poll[count]
		weighted_data.append(tmp_weight)
		count += 1

	'''
	 Final data preparation , with 5 columns
	'''

	table = {'trump_clinton_win' : binomial_data,'state' : state_data, 
	 'polling_weight' : weighted_data, 'trump_adjusted' : trump_error_data, 'clinton_adjusted':clinton_error_data }
	new_df = pd.DataFrame(data=table)

	return new_df

def test_train_data(new_df):
	'''
	Source  for LogisticRegression code

	http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976
	'''
	y, X = dmatrices('trump_clinton_win ~ state + polling_weight + trump_adjusted + clinton_adjusted' ,new_df, return_type="dataframe")
	y = np.ravel(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	return  X, y, X_train, X_test, y_train, y_test

def logistic_model(new_df, X_train , y_train, X_test):
	'''
	Source  for LogisticRegression code

	http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976
	'''
	#Finding out the class label of test set using attributes and class label of training set
	#using Logistic Regression.
	print 'Modelling with logistic regression \n'
	model2 = LogisticRegression()
	model2.fit(X_train, y_train)
	predicted_regression = model2.predict(X_test)
	probs = model2.predict_proba(X_test)

	#print ' accuracy on Logistic model training set ', model2.score(X_train, y_train)
        print ' \n Done Modelling with logistic regression \n'
	return model2, predicted_regression, probs

def accuracy(y_test, logistic_model_predicted, logistic_model_prob,
               decision_tree_model_predicted, decision_tree_model_prob):

	print " \n Doing Accuracy measurement with both models \n"


	print 'logistic_model classification_report', metrics.classification_report(y_test, logistic_model_predicted)
	#print 'logistic_model for clinton_probability' , logistic_model_prob[:, 1]
	#print 'logistic_model for trump_probability' , logistic_model_prob[:, 0]

	print 'decision_model classification_report' , metrics.classification_report(y_test, decision_tree_model_predicted)

	#print 'logistic_model for clinton_probability' , decision_tree_model_prob[:, 1]
	#print 'logistic_model for trump_probability' , decision_tree_model_prob[:, 0]

	print " \n Done with  Accuracy measurement with both models \n"



def decision_tree_model(new_df, X_train , y_train, X_test):
        #Finding out the class label of test set using attributes and class label of training set
	#using Decision Tree.
	print " Modelling with decision tree"
	'''
	 Help from here
	 http://scikit-learn.org/stable/modules/tree.html
	''' 
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	predicted_decision_tree = clf.predict(X_test)
	probs_decision_tree = clf.predict_proba(X_test)

	#print ' accuracy on Decison tree model training set ', clf.score(X_train, y_train)
	print " \n Done with Modelling with decision tree \n"
	return clf, predicted_decision_tree, probs_decision_tree

#5-fold Cross Validation
def cross_validation(logistic_model, X_test , y_test, decison_model):
	'''
	#Another way http://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/
	'''
	print "\nFinding the cross validation scores\n"

	scores = cross_val_score(logistic_model, X_test, y_test, scoring='accuracy', cv= 5)
	print '\n for logictic regression after 5-fold cross-validation', scores, scores.mean()
	kfold = KFold(n=len(X_test), n_folds=5)
	results = cross_val_score(logistic_model, X_test, y_test, cv=kfold)
	print("Accuracy for LogisticRegression: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

	 
	scores = cross_val_score(decison_model, X_test, y_test, scoring='accuracy', cv= 5)
	print '\n for decision tree after 5-fold cross-validation', scores, scores.mean()
	kfold = KFold(n=len(X_test), n_folds=5)
	results = cross_val_score(decison_model, X_test, y_test, cv=kfold)
	print("Accuracy for decision tree: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)

	print "\n======= Done with Cross validation ==========\n"
	
	
def draw_precision(logistic_model_prob, decision_model_prob, logistic_model_pred):
        # http://stackoverflow.com/questions/15415455/plotting-probability-density-function-by-sample-with-matplotlib
        '''
          Following code is to show the probability distribution from the output .
          The density shows
        '''  
	fig=plt.figure()
	kde = gaussian_kde( logistic_model_prob[:,1] )
	dist_space = linspace( min(logistic_model_prob[:,1]), max(logistic_model_prob[:,1]), 100 )
	plt.xlabel('logistic probability distribution')
	plt.ylabel('clinton win probability')
	# plot the results
	plt.plot( dist_space, kde(dist_space) )
	plt.show()
	
	fig=plt.figure()
	kde = gaussian_kde( decision_model_prob[:,1] )
	dist_space = linspace( min(decision_model_prob[:,1]), max(decision_model_prob[:,1]), 100 )
	plt.xlabel('decision tree probability distribution')
	plt.ylabel('clinton win probability')
	# plot the results
	plt.plot( dist_space, kde(dist_space) )
	plt.show()
	
	
def project():
	df = read_data('F:\Ritu\DataMining\Project\presidential_polls.csv')
	#question - 1
	draw_visualization(df)
    
	#question 2
	new_df = clean_prepare_data(df)
	X,y, X_train, X_test, y_train, y_test = test_train_data(new_df)
	log_model, logistic_model_pred, logistic_model_prob = logistic_model(new_df, X_train , y_train, X_test)
	desc_model, decision_model_pred, decision_model_prob =  decision_tree_model(new_df, X_train , y_train, X_test)

	#question 3
	cross_validation(log_model,X,y, desc_model)

	#question 4
	accuracy(y_test, logistic_model_pred, logistic_model_prob, decision_model_pred, decision_model_prob)
	#draw_precision(logistic_model_prob, decision_model_prob, logistic_model_pred)

project()