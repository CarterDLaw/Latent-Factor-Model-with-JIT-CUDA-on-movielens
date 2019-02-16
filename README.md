# Latent-Factor-Model-with-JIT-CUDA-on-movielens
2018/12/14 16:20 Wuhan, China
Recommendation System, Big Data Processing, Python

It is a recommendation system project with moivelens https://grouplens.org/datasets/movielens/ Latest Dataset.
This work aims to accomplish a SVD Latent Factor Model (LFM) approach towards a movie recommendation system. 

Data Source: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip 

Main Data Files: movies.csv(include movie ID, title, genre), ratings.csv(include user ID, movie ID, rating 1-5)

Main Python Files: 1. 

1.Estabilish a system to transfer data format from csv. to npy. for better I/O performance, which is represented by 'csv_npy.py'. 

2.Create a user-movie rating matrix (user - column axis, movie - horizontal axis) that contains elements from 1-5 to represent the rating of each movie by each user. However, movie ID is not a serie of continuous numbers from 1 to N. It would be costly to storage a huge matrix  where some columns are comlete empty. Therefore, we reorder the movie ID to be continuous from 1 to M (M <= N). Then we eventually seperate the matrix into two parts: train_set.npy and test_set.npy. These are achieved by 'alter.py'. 

3.Now the data is ready for training! The core model here is LFM, which is an iterative learning algorithm. To boost the speed of training, we use JIT and CUDA acceleration with numba. After tests, it is comfirmed that the training speed has been twenty times faster after using JIT and CUDA. All these are achieved by 'main_SGD.py'.

The parameter tuning is tricky though. We find out that the learning_rate is best to be dynamic. Therefore, it is recommended to use a custom method to adjust the learning_rate after each iteration. (In this project: if the loss is declining, increase the learning_rate by 0.00001; if the loss is increasing, halve the learning_rate. Also, set up a threshold to clamp the learning_rate to be smaller than a maximum and bigger than zero)

Sincerely welcome any comments or advice:)
