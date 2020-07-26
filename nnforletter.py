import keras
import glob
from tqdm import tqdm
import os
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

num_classes = 32
EPOCHS = 25
BS = 32

train_dirs = glob.glob("./dataset/letters/*")
train_dirs.sort(key = lambda x: int(x.rsplit('/',1)[1]))


data = []
labels = []
for train_dir in tqdm(train_dirs):
    imgPaths = glob.glob(train_dir + "/*.jpg")
    imgPaths.sort()
    for imgPath in tqdm(imgPaths):
        image = load_img(imgPath, target_size=(28, 28), grayscale=True)
        image = img_to_array(image) 
        data.append(image)

        label = imgPath.split(os.path.sep)[-2]
        label = int(label)
        labels.append(label)
        


data = np.array(data, dtype=np.float) / 255.
labels = np.array(labels)

train_input, valid_input, train_target, valid_target = train_test_split(data,
                                                                        labels,
                                                                        test_size=0.25,
                                                                        random_state=123)




