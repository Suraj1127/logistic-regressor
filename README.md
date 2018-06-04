# linear-regressor
Simple linear regression program which builds linear regression model and then predicts on the input data.

Language: Python

Version: Python3.x

Python libraries required:
1) Numpy
2) Pandas
3) Matplotlib
4) Sklearn

Instructions:
1) Put the preprocessed, clean, without-null training data in input.csv and output.csv files located in train folder.  The files should be in csv format and first line should contain the name of the variables.  The first column of both the files should contain indices so they should not have any variable.  The first value of the first line(containing name of variables) in both files is empty. 
2) Put the test or to-predict input data in the input.csv file located in test folder in the same format the input.csv file is in train folder.
3) Run main.py and then the program would predict the output and save as output.csv in test folder in the same format other csv files are in.

The code in linear_regression.py is well documented and we can set the hyperparameters there while training.  In performance metrics, root mean squared error and coefficient of determination are used.  They are used to evaluate the model.
