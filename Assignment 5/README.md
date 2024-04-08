# Kimia-Kariman
Collaborative Filtering Recommender Systems
In this fifth homework assignment titled "Collaborative Filtering Recommender Systems," I embarked on an exploration of implementing collaborative filtering to create a movie recommender system. This work is a significant endeavor in the fields of data science and machine learning, especially relevant in the context of entertainment and e-commerce platforms.

Overview
My assignment comprised 43 cells, including a mix of 15 code cells and 28 markdown cells, demonstrating a balanced approach between coding practices and explanatory documentation. The main objective was to implement collaborative filtering techniques to generate movie recommendations, an essential component of modern recommendation systems.

Introduction and Setup
I began with an introductory markdown cell that set a thematic tone with movie and film images, outlining the exercise's structure from basic notation and system recommendations to the collaborative filtering learning algorithm and actual movie recommendations.

In the code segments, I started by importing necessary libraries such as numpy for numerical operations and tensorflow for building the recommendation system. This setup step was crucial, indicating an understanding of the essential tools required for machine learning tasks:

import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *
Data Loading and Initial Exploration
Following the introduction, I proceeded to load the dataset and pre-calculated parameters, including movie ratings, user interactions, and movie features. This foundational step was crucial for understanding the dataset and setting up the recommender system:

#Load data
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)
I also calculated the average rating for the first movie to start engaging with the dataset, an initial step towards deeper data interaction:

# From the matrix, we can compute statistics like average rating.
tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )
Instructional Guidance
The notebook is well-documented with instructions, including notes on not editing or adding cells to ensure assignment compliance, and an extensive outline. It also lists the packages used, emphasizing NumPy and TensorFlow for their importance in data manipulation and machine learning respectively.

Conclusion
Throughout this assignment, I demonstrated a methodical approach to building a collaborative filtering recommender system. From setting up the environment and loading the data to beginning data exploration, I laid a solid foundation for more complex operations that likely followed in the subsequent sections.

This journey showcases not only the technical skills involved in implementing machine learning algorithms but also the ability to document and structure work comprehensively. The provided snippets offer a glimpse into the broader methodologies I applied to achieve the objectives, emphasizing engagement and understanding of the task at hand.
