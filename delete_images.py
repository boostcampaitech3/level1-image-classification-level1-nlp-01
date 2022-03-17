# leave only 'asian' images

import os, sys

data_dir = './UTKFace'

delete_list = []
temp =[]

for image in os.listdir(data_dir):
    if len(image.split("_")) != 4:
        temp.append(os.path.join(data_dir,image))

while temp:
    os.remove(temp[0]) # remove files that has invalid name

# for image in os.listdir(data_dir):
#     age, gender, race, id = image.split("_")
#     if race != '2':
#         delete_list.append(os.path.join(data_dir, image))

print(len(os.listdir(data_dir)))
# print(len(delete_list))
