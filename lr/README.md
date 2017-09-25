# Logistic Regression
This example applies logistic regression to classify admission based on two exam scores.

The `lr.py` file uses logistic regression with a first-order decision boundary and `rlr.py` uses a sixth-order decision boundary along with regularization to combat over-fitting.

The data files are `ex2data1.txt` for `lr.py` and `ex2data2.txt` for `rlr.py`.

The results are saved as plots (`ex2-1.svg` and `ex2-2.svg`) with the accepted and rejected data points along with a plotted decision boundary.

## Results
The first classification problem invovled two exam scores and the resulting admission or rejection from a school. A first-order model was fit to the data and is shown on the plot as the blue dotted line.

![Logistic Regression](ex2-1.svg)

The second classification problem invovled two tests. This time a sixth-order model was fit to the data with regularization, due to the potential for overfitting. The unregularized, sixth-order decision boundary is shown as the blue dotted lined.

![Logistic Regression with Regularizaiton](ex2-2.svg)

## Matrix Algebra in Python and NumPy
The most important concept to keep in mind when using Python and NumPy for matrix algebra is the dimensionality of the arrays, vectors, and lists that you are operating on. By default, Python does not include any form of vector or matrix operations, instead using a data type called a 'list'. NumPy, part of the scientific computing module stack for Python, adds multi-dimensional arrays and operators for these arrays.
