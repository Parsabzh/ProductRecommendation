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

# Calculate the number of ratings per user and sort them in descending order
most_rated = df.groupby('userId').size().sort_values(ascending=False)[:10]

# Print the top 10 users based on ratings
print('Top 10 users based on ratings: \n', most_rated)

# Count the number of ratings for each user
counts = df.userId.value_counts()

# Filter the DataFrame to include only users who have rated 15 or more items
df_final = df[df.userId.isin(counts[counts >= 15].index)]

# Print the number of users who have rated 25 or more items
print('Number of users who have rated 25 or more items =', len(df_final))

# Print the number of unique users in the final data
print('Number of unique users in the final data = ', df_final['userId'].nunique())

# Print the number of unique products in the final data
print('Number of unique products in the final data = ', df_final['productId'].nunique())

# Remove duplicate user entries to ensure unique user IDs in the final DataFrame
df_final = df_final.drop_duplicates(subset='userId')

# Create a pivot table with 'userId' as the index, 'productId' as the columns, and 'ratings' as the values
# Fill missing values with 0
final_ratings_matrix = df_final.pivot(index='userId', columns='productId', values='ratings').fillna(0)

# Print the first few rows of the final ratings matrix
print(final_ratings_matrix.head())

# Print the shape of the final ratings matrix
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

# Calculate the number of non-zero ratings in the final ratings matrix
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)

# Calculate the possible number of ratings in the final ratings matrix
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]

# Calculate the density of the ratings matrix (ratio of given ratings to possible ratings)
density = (given_num_of_ratings / possible_num_of_ratings)
density *= 100

# Print the density of the ratings matrix as a percentage
print('density: {:4.2f}%'.format(density))
print(given_num_of_ratings)
print(possible_num_of_ratings)

train_data, test_data = train_test_split(df_final, test_size = 0.2, random_state=0)

train_data = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()
train_data.rename(columns = {'userId': 'score'},inplace=True)
print(train_data.head(40))



