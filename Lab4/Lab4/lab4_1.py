import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


# vectors
a = np.array([[1,2,3]])
b = np.array([[1,1,4]])

cos_lib = cosine_similarity(a, b)
print(cos_lib)