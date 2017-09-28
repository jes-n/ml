# ml (Machine Learning)
This repository will cover my programming assignments for Coursera's online course on Machine Learning by Dr. Andrew Ng at Stanford University.

https://www.coursera.org/learn/machine-learning

It's an 11-week course covering an introduction to machine learning, including: logistic regression, neural networks, support vector machines, dimensionality reduction, and unsupervised learning.

### lr (Logistic Regression)
Applying logistic regression with first- and higher-order decision boundaries, regularization, and multiple classes.
* ```lr.py``` - Basic 2-class linear regression
* ```rlr.py``` - Regularized linear regression
* ```mclr.py``` - Multi-class linear regression

### nn (Neural Networks)
Applying multi-layer neural networks to more efficiently and accurately predict classifications.
* ```wnn.py``` - Pre-weighted, 3-layer neural network

### data
All of the data used in the machine learning software is kept here as CSV text files.
* ```ex2data1.txt``` - 100 samples of two exam scores and a admission or rejection decision (used by ```lr.py```)
* ```ex2data2.txt``` - 118 samples of two tests and an overall pass or fail (used by ```rlr.py```)
* ```ex3data1.txt``` - 5000 20x20 pixel images of hand-written numbers between 0 and 9 (used by ```mclr.py``` and ```wnn.py```)