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



# Define column names for the data
columns = ['userId', 'productId', 'ratings', 'timestamp']

# Read the CSV file and store the data in a DataFrame
df = pd.read_csv('all_csv_files.csv', names=columns, nrows=50_000)

# Print descriptive statistics of the DataFrame
print(df.describe())

# Print the first few rows of the DataFrame
print(df.head())

# Drop the 'timestamp' column from the DataFrame
df.drop('timestamp', axis=1, inplace=True)

# Print information about the DataFrame
print(df.info())

# Calculate and print descriptive statistics of the 'ratings' column
print(df['ratings'].describe().transpose())

# Print the minimum rating
print(f"Minimum rating is: {df['ratings'].min()}")

# Print the maximum rating
print(f"Maximum rating is: {df['ratings'].max()}")

# Check for missing values across columns
print('Number of missing values across columns:\n', df.isnull().sum())

# Generate a histogram of the 'ratings' column
plt.hist(df["ratings"])
# plt.show()

most_rated=df.groupby('userId').size().sort_values(ascending=False)[:10]
print('Top 10 users based on ratings: \n',most_rated)

counts=df.userId.value_counts()
df_final=df[df.userId.isin(counts[counts>=15].index)]
print('Number of users who have rated 25 or more items =', len(df_final))
print('Number of unique users in the final data = ', df_final['userId'].nunique())
print('Number of unique products in the final data = ', df_final['productId'].nunique())
df_final = df_final.drop_duplicates(subset='userId')
final_ratings_matrix = df_final.pivot(index = 'userId', columns ='productId', values = 'ratings').fillna(0)
print(final_ratings_matrix.head())

