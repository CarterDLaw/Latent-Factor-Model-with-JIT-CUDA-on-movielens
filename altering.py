import numpy as np

dic = np.load("movie_dict.npy").item()
rating = np.load("rating.npy")
rating_alter = []
for index, item in enumerate(rating):
    item[1] = dic.get(item[1])
    rating_alter.append(item)
np.save("rating_alter.npy", rating_alter)

train_set = []
test_set = []
all_set = []
rating_alter = np.load("rating_alter.npy")
rating_alter[:,0] = rating_alter[:,0] - 1
print(rating_alter[:,0])
for index, item in enumerate(rating_alter):
    if item[0] % 20 == 0 and item[1] % 25 == 0:
       test_set.append(item)
    else:
       train_set.append(item)
    # all_set.append(item)
np.save("train_set.npy", train_set)
np.save("test_set.npy", test_set)
# np.save("all_set.npy", all_set)

