import sklearn
import numpy as np
from matplotlib import pyplot as plt

class WeakClassifier:
  def __init__(self, nFeatures):
    # TODO: Initialize stuff you need here
    self.alpha = 1.0 # The Alpha-Value for later if this classifier is picked

    # The weak classifier shall pick two random dimensions out of the 
    # feature vector. It will classify a sample as positive if featureA is bigger or equal than featureB
    
  def predict(self, samples):
    # TODO: Implement classifier prediction
    # Predict is an (N x 64) array, holding all feature vectors of 
    # the N samples, each 64-dimensional. 
    # Your task is to (weakly) classify each individual sample
    # as described in the task description. 
    # The expected return value is an (N x 1) array with only 
    # +1 or -1 in it. Return +1 if the sample is classified positve
    # and -1 if the sample is classified negative. Return nothing else
    # than -1 or +1. 
    pass

def generate_weak_classifiers():
  # Generate a random selection of weak classifiers. The AdaBoost Algorithm
  # will pick one of these for the next cascade stage
  weakClassifiers = []
    
  for _ in range(8):
    weakClassifiers.append(WeakClassifier(64))
  
  return weakClassifiers

def pick_weak_classifier(data, labels, weights, classifiers):
  # TODO: Implement the picking stage of the AdaBoost Algorithm.
  # We try to find the one classifier out of the given classifiers
  # which minimize the sum of weights for wrongly classifier samples
  #
  # data is a (N x 64) array containing all the training samples
  # labels is a (N x 1) array containing 1 or -1 depending on whether 
  # the sample is positive or negative
  # weights is a (N x 1) array containing the weights for each sample
  # classifiers is a list of potential WeakClassifiers (see above)
  pass

def build_one_stage(data, labels, weights, classifiers, cascade):
  # Pick the best weak classifier given current weights
  classifier = pick_weak_classifier(data, labels, weights, classifiers)

  # TODO: Calculate the alpha-value for the choosen classifier
  # according to the script and store it within the classifier object 
  # for later reference

  # TODO: Update weights for each samples according to script
  
  # Remember alpha and add choosen classifier to cascade
  classifier.alpha = 0 # TODO: Replace with calculated alpha
  cascade.append(classifier)
  
  return weights, cascade

def predict_cascade(data, cascade):
  # TODO: Implement evaluation of the whole AdaBoost cascade
  # Data is an (N x 64) array containing all data samples
  # Cascade is a list of choosen classifiers. Their respective 
  # Alpha values are stored within the classifier object. 
  pass


def load_data():
  # Load the Digits dataset
  # The digits dataset contains images of resolution 8x8 pixels. Each pixel contains values between 0 and 15. 
  # They resemble images of the hand-written digits 0 to 9
  digits = sklearn.datasets.load_digits()

  # Select two digits for classification. Flatten the images as we don´t need the 2D structure anyway
  positive_class = digits.images[digits.target == 4].reshape(-1,64)
  negative_class = digits.images[digits.target == 8].reshape(-1,64)
  positive_label = np.ones(positive_class.shape[0])
  negative_label = -np.ones(negative_class.shape[0])

  # Concatenate both into the same set  
  data = np.concatenate([positive_class, negative_class])
  labels = np.concatenate([positive_label, negative_label])

  # Start with equal weights for each sample
  weights = np.ones_like(labels)

  return data, labels, weights

# Load the data
data, labels, weights = load_data()
print(data.shape)
print(labels.shape)
print(weights.shape)
exit()

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(data[i, :].reshape(8,8), cmap=plt.cm.binary, interpolation='nearest')

plt.show()    

# Start with an empty cascade
cascade = []

# Add 50 weak classifiers 
for i in range(50):
  # Generate a new set of weak classifier
  classifiers = generate_weak_classifiers()  

  # Pick one and re-evaluate the weights for each samples
  weights, cascade = build_one_stage(data, labels, weights, classifiers, cascade)
  
  # Calculate predictions for the whole cascade
  predictions = predict_cascade(data, cascade)

  # Count wrong samples
  wrong = predictions != labels
  total_wrong = wrong.sum()

  # Also calculate total error value
  E = np.sum(np.exp(-predictions * labels))

  # Output
  print(f"Stage {i}, E={E:.5f}, total wrong = {total_wrong}")
