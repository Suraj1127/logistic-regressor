# logistic-regressor
Simple logistic regression program which builds logistic regression model using sigmoid function and then predicts on the input data.

Language: Python

Version: Python3.x

Python libraries required:
1) Numpy
2) Pandas
3) Matplotlib
4) Sklearn

Instructions:

i) If you want to just see the fitting and working of logistic regression:

    1) Run logistic_regression.py and then it would build a simple dataset and train on them.  It would also test on validation dataset and show the accuracies, both training as well as validation.  You can see the initial data distribution as well as how our curve fits the model.
    

ii) If you want to train with your dataset and fit it with logistic regression:

    1) Put the preprocessed, clean, without-null training data in input.csv and output.csv files located in train folder.  The files should be in csv format and first line should contain the name of the variables.  The first column of both the files should contain indices so they should not have any variable.  The first value of the first line(containing name of   variables) in both files is empty.
    
    2) Put the test or to-predict input data in the input.csv file located in test folder in the same format the input.csv file is in train folder.
    
    3) Run main.py and then the program would predict the output and save as output.csv in test folder in the same format  other csv files are in.
    

The code in logistic_regression.py is well documented and we can set the hyperparameters there while training.  In performance metrics, training and test accuracy are used.  They are used to evaluate the model and how it is fitting.  You can also test with your test set and check the accuracy yourself by using predict method.

Learn more about it here:

https://en.wikipedia.org/wiki/Logistic_regression
