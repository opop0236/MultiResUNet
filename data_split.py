import os
import cv2
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

''' Set to DataSet '''
def data_split():
    msk_files = \
    next(os.walk('/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/ISIC-2017_Training_Part1_GroundTruth'))[2]
    img_files = next(os.walk('/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/ISIC-2017_Training_Data'))[2]

    img_files.sort()
    msk_files.sort()

    print(len(img_files))
    print(len(msk_files))

    X = []
    Y = []

    for img_fl in tqdm(img_files):
        if (img_fl.split('.')[-1] == 'jpg'):
            img = cv2.imread(
                '/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/ISIC-2017_Training_Data/{}'.format(img_fl),
                cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_CUBIC)

            X.append(resized_img)

            msk = cv2.imread(
                '/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/ISIC-2017_Training_Part1_GroundTruth/{}'.format(
                    img_fl.split('.')[0] + '_segmentation.png'), cv2.IMREAD_GRAYSCALE)
            resized_msk = cv2.resize(msk, (256, 192), interpolation=cv2.INTER_CUBIC)

            Y.append(resized_msk)

    ''' Train-Test data split '''
    print(len(X))
    print(len(Y))

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
    Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

    X_train = X_train / 255
    X_test = X_test / 255
    Y_train = Y_train / 255
    Y_test = Y_test / 255

    Y_train = np.round(Y_train,0)
    Y_test = np.round(Y_test,0)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test