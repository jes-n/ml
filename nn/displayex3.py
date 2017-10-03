from matplotlib.pyplot import figure, show
from matplotlib.cm import binary
from numpy import zeros, linspace, arange, flip
from numpy.random import random_integers

### Module built to display '/data/ex3data1.csv'

def displayData(X):
	""" Selects and displays a random sample of 100 of the hand-written images. """

	fig = figure(figsize=(6,6))
	ax = fig.add_subplot(111)

	images = zeros((10*20, 10*20)) # Empty 200x200 array for 10x10 images
	# Randomly select 100 integers between 0 and 4999 (inclusive)
	indicies = random_integers(low=0, high=4999, size=100)

	# This for-loop will step through the randomly selected indicies
	# Placing the selected 20x20 image into the larger 200x200 'images' array
	# The index will be place in order of (0,0), (0,1), ... (1,0), (1,1), ... (9,8), (9,9)
	# Taking into acount that each image is itself a 20x20 array
	# *** Fundamentally this loop puts many smaller matricies into a larger matrix ***
	for i, index in enumerate(indicies):
		row = [int(int(i/10)*20), int(int(i/10+1)*20)]
		col = [int((i%10)*20), int((i%10+1)*20)]
		images[row[0]:row[1],col[0]:col[1]] = X[index].reshape(20,20)
		# ### Plot text corresponding to the image index along with the image (looks too cluttered)
		# plt.text(int(row[0]+1), int(col[0]+19), "{}".format(index+1), fontsize=5,
		# 	bbox={"boxstyle":"Round, pad=0.05", "facecolor":"white", "edgecolor":"black", "lw":0.5)

	# Plot a grid to highlight each image individually
	for line in linspace(0, 200, 11):
		ax.axhline(line, color="k")
		ax.axvline(line, color="k")

	# Adjust the plot elements to better reflect the data
	ax.set_xticks(arange(10, 210, 20))
	ax.set_xticklabels(arange(1, 11, 1))
	ax.set_yticks(arange(10, 210, 20))
	ax.set_yticklabels(flip(arange(1, 11, 1), axis=0))
	ax.set_title("Random Sample of 100 Hand-Written Numbers")

	# Plot and show the result
	ax.imshow(images.T, cmap=binary)
	show()

def displayImage(X, index):
	""" Specify a single image 'index' to display. """

	fig = figure(figsize=(6,6))
	ax = fig.add_subplot(111)

	# Adjust the plot elements to better reflect the data
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	ax.set_title("Image #{}".format(index))

	# Plot and show the result
	ax.imshow(X[index].reshape(20,20).T, cmap=binary)
	show()