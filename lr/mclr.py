import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

### "ex3data.txt" contains 5,000 samples of 401 element arrays
### These correspond to 20x20 pixel images (400 elements) and 1 classification
### The images are written numbers and the classifications are the number (0-9)
X = np.loadtxt(os.path.join("..", "data", "ex3data1.txt"), delimiter=",", skiprows=1)
[m,n] = X.shape
# Assign 'y' as the classification array
y = X[:,[-1]]
# Remove 'y' from X
X = np.delete(X, -1, axis=1)

### View one of the images (0 to 4999)
# image = 1484
# plt.imshow(X[image].reshape(20,20), origin='lower', cmap=plt.cm.binary)
# plt.title("Image of {}".format(int(y[image])))
# plt.show()
###################

# Add the bias array to 'X'
X = np.column_stack((np.ones(m), X))

def sigmoid(z):
	return 1/(1+np.exp(-z))

def costFunction(theta, X, y, check=False):
	hyp = lambda X: sigmoid(np.dot(X,theta.T))
	m = X.shape[1]

	### This try/except condition catches a 1D weight vector
	try:
		# If this condition fails it means 'theta' is dimensioned (n,)
		theta.shape[1]
	except:
		# 'theta' is then changed to a dimension (1,n)
		theta = theta.reshape(1,X.shape[1])

	### Check the dimensionality
	if (check):
		print("X:", X.shape, type(X))
		print("y:", y.shape, type(y))
		print("theta:", theta.shape, type(theta))
	###############

	return np.average(np.multiply(-y,np.log(hyp(X))) - np.multiply((1-y),np.log(1-hyp(X))), axis=0)

# Define the number of classes
classes = 10
### Assume a weight vector ('theta') of random floats between 0 and 1
# theta = np.random.rand(classes,n)
### Assume a weight vector ('theta') of zeros
theta = np.zeros((classes,n))

### Verify the dimensions and type of 'X', 'y', and 'theta'
print("X:", X.shape, type(X))
print("y:", y.shape, type(y))
print("theta:", theta.shape, type(theta))

### This is the callback function for SciPy's minimize()
### It retuns xk, which is the current minimization iteration's theta value
iteration = 1
def callback(xk):
	global iteration
	print(costFunction(xk.reshape(1,n), X, cat))
	iteration += 1
	return 0

### Determine the weight vector for each classification
###################
### This will be done by iterating over each class
### A Conjugate Gradient minimization will be preformed to determine each set of weight vectors
###################
# start = time.time() # For minimization timing purposes
# # samples = 500
# for i in range(classes):
# 	print("Learning {}".format(i))
# 	cat = [1 if (int(val)%10 == i) else 0 for val in y]
# 	cat = np.array((cat)).reshape(m,1)
# 	res = minimize(costFunction, theta[[i]], args=(X, cat),
# 		method='BFGS', options={"disp":True, "gtol":5e-4}, callback=callback)
# 	print(res.message)
# 	# print("Weight vector for '{}':".format(i), res.x)
# 	print("Minimum cost for '{}':".format(i), costFunction(res.x.reshape(1,n), X, cat))
# 	theta[i,:] = res.x.reshape(1,n)
# print("Time elapsed:", time.time() - start, "seconds") # Show the elapsed time
# ### Save the weight vector 'theta' for later use because it is computationally expensive
# np.savetxt("weights.txt", theta, delimiter=",")
###################
### Load the previously determined weight vector
theta = np.loadtxt("weights.txt", delimiter=",")
### Verify the dimensions and type of 'theta'
print("theta:", theta.shape, type(theta))
###################

### 'hypothesis' contains a 5000x10 matrix
### Each row i corresponds to the probabilities of image i being a given number
hypothesis = sigmoid(np.dot(X, theta.T))
### Check the predictions
correct = 0
for i, hyp, img in zip(range(len(y)), hypothesis, y):
	if (int(img)%10 == int(np.argmax(hyp))):
		# print("Success!")
		correct += 1
	else:
		print("#{} (guess: {}, actual: {})".format(i, int(np.argmax(hyp)), int(img)%10))

### Display the final results!
print("{} right out of 5000 ({}%)".format(correct, '%0.1f'%(correct/5000*100)))