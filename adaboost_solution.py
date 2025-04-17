import sklearn
import numpy as np
from matplotlib import pyplot as plt

class WeakClassifier:
  def __init__(self, nFeatures):
    # The alpha value for later if this classifier is picked into the cascade
    self.alpha = 1.0

    # The weak classifier will pick two random dimensions out of the feature vector
    # It will classify a sample as positive if featureA is bigger or equal than featureB
    self.featureA = int(np.random.uniform(0, nFeatures))
    self.featureB = int(np.random.uniform(0, nFeatures))

  def predict(self, samples):
    values = samples[:, self.featureA] - samples[:, self.featureB]
    return 2 * (values >= 0) - 1

def generate_weak_classifiers():
  # Generate a random selection of weak classifiers. The AdaBoost Algorithm
  # will pick one of these for the next cascade stage
  weakClassifiers = []
    
  for _ in range(8):
    weakClassifiers.append(WeakClassifier(64))
  
  return weakClassifiers

def pick_weak_classifier(data, labels, weights, classifiers):
  # We try to find the one classifier out of the given classifiers
  # which minimize the sum of weights for wrongly classifier samples
  minimalSum = None
  bestClassifier = None
  
  # Iterate over all options
  for classifier in classifiers:
    # Make a prediction for each samples
    predictions = classifier.predict(data)

    # Wrong samples are those whose prediction differs from the label
    wrong = predictions != labels

    # Sum the current weights for wrongly predicted samples
    sumW = weights[wrong].sum()

    # If this is lower, keep this classifier as current best
    if bestClassifier is None or sumW < minimalSum:
      bestClassifier = classifier
      minimalSum = sumW

  # Return best classifier
  return bestClassifier

def build_one_stage(data, labels, weights, classifiers, cascade):
  # Pick the best weak classifier given current weights
  classifier = pick_weak_classifier(data, labels, weights, classifiers)

  # Calculate predictions
  predictions = classifier.predict(data)
  wrong = predictions != labels

  # Calculate weighted error sum
  e = weights[wrong].sum() / weights.sum()

  # Calculate alpha value
  alpha = 0.5 * np.log((1 - e) / e)
  #print(e, alpha)

  # Update weights for each samples
  weights = weights * np.exp(-alpha * classifier.predict(data) * labels)

  # Remember alpha and add to cascade
  classifier.alpha = alpha
  cascade.append(classifier)
  
  return weights, cascade

def predict_cascade(data, cascade):
  # Evaluate the cascaded classifier
  # This is the weighted (with alpha) sum of all individual classification decisions
  values = np.zeros(data.shape[0])
  for classifier in cascade:
    values = values + classifier.alpha * classifier.predict(data)
  
  return 2 * (values >= 0) - 1


def load_data():
  # Load the Digits dataset
  # The digits dataset contains images of resolution 8x8 pixels. Each pixel contains values between 0 and 15. 
  # They resemble images of the hand-written digits 0 to 9
  digits = sklearn.datasets.load_digits()

  # Select two digits for classification. Flatten the images as we don´t need the 2D structure anyway
  positive_class = digits.images[digits.target == 2].reshape(-1,64)
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
