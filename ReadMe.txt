
Language Used : Python 
Package required sklearn , numpy, matplotlib , patsy , pandas
Code source :  Help is taken from http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976
		stackoverflow.com
                http://scikit-learn.org/stable/modules/tree.html

Requirement of modules:
sklearn for data mining packagaes
matplotlib for plotting charts
pandas for data frame operations
pprint for pretty print of dictionary
numpy for mathmatical operations

How to Run :

python source_code.py 

Answer: In the program need to change parameter for read_data. Need to provide the path 
for the csv file .

How to execute Question 1:

Answer : Need to uncomment draw_visualization(df) . It will draw two graph .
 The trend of polling  result against each Month is drawn.

How to Answer Question 2 :

Answer: Two models are used. Logistics Regressiona and decision tree model.
 logistic_model(new_df, X_train , y_train, X_test) 
 decision_tree_model(new_df, X_train , y_train, X_test)


How to Answer for Question 3:

Answer: cross_validation() method is used to do cross validation for both of my models

How to Answer for Question 4:

Answer: accuracy() method is used.
