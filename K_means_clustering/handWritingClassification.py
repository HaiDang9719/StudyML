import numpy as np 
from mnist import MNIST # require `pip install python-mnist`
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from display_network import *

mndata = MNIST('../MNIST/') # path to your MNIST folder 
mndata.load_testing()
X = mndata.test_images

kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)