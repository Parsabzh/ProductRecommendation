import numpy as np
import pandas as pd
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
# from sklearn.externals import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import gzip


columns=['userId', 'productId', 'ratings','timestamp']
df=pd.read_csv('all_csv_files.csv',names=columns,nrows=10_000_000)

print(df.describe())
print(df.head())

df.drop('timestamp',axis=1,inplace=True)
print(df.info())


