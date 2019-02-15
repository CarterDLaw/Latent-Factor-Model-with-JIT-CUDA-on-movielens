import csv
import numpy as np

with open("ratings.csv", encoding='utf-8') as csvRating:
    reader = csv.reader(csvRating)
    data = []
    for index, item in enumerate(reader):
        if (index != 0):
            data.append(item)
    for index, item in enumerate(data):
        item[0] = int(item[0])
        item[1] = int(item[1])
        item[2] = float(item[2])
        data[index] = [item[0], item[1], item[2]]
    np.save("rating.npy", data)

with open("movies.csv", encoding='utf-8') as csvMovie:
    reader = csv.reader(csvMovie)
    data = []
    Movie = []
    dic = {}
    for index, item in enumerate(reader):
        if index != 0:
            data.append(item)
    for item in data:
        item[0] = int(item[0])
        Movie.append(item[0])
    for index, item in enumerate(Movie):
        dic[item] = index
    np.save("movie_dict.npy", dic)




