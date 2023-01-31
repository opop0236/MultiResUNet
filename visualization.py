import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if (activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def MultiResBlock(U, inp, alpha=1.67):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) +
                         int(W * 0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath

    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, n_channels):
    '''
    MultiResUNet

    Arguments:
        height {int} -- height of image
        width {int} -- width of image
        n_channels {int} -- number of channels in image

    Returns:
        [keras model] -- MultiResUNet model
    '''

    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32 * 2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32 * 2, 3, mresblock2)

    mresblock3 = MultiResBlock(32 * 4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32 * 4, 2, mresblock3)

    mresblock4 = MultiResBlock(32 * 8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32 * 8, 1, mresblock4)

    mresblock5 = MultiResBlock(32 * 16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32 * 8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32 * 8, up6)

    up7 = concatenate([Conv2DTranspose(
        32 * 4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32 * 4, up7)

    up8 = concatenate([Conv2DTranspose(
        32 * 2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32 * 2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jacard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection / union

if __name__ == "__main__":
    model = MultiResUnet(height=192, width=256, n_channels=3)
    model.load_weights('/content/drive/MyDrive/Colab Notebooks/modelW.h5')
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])

    model.summary()

    # img_files = next(os.walk('/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/Test'))[2]
    img_files = next(os.walk('/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/ISIC-2017_Test_v2_Data'))[2]
    img_files.sort()

    print(len(img_files))

    org_img = []
    X = []
    Y = []

    for img_fl in tqdm(img_files):
        if (img_fl.split('.')[-1] == 'jpg'):
            img = cv2.imread(
                '/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/ISIC-2017_Test_v2_Data/{}'.format(img_fl),
                cv2.IMREAD_COLOR)
            msk = cv2.imread(
                '/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/ISIC-2017_Test_v2_Part1_GroundTruth/{}'.format(
                    img_fl.split('.')[0] + '_segmentation.png'), cv2.IMREAD_GRAYSCALE)
            # img = cv2.imread('/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/Test/{}'.format(img_fl), cv2.IMREAD_COLOR)
            # msk = cv2.imread('/content/drive/MyDrive/Colab Notebooks/ISIC_2017_Images/Test/{}'.format(img_fl.split('.')[0]+'_segmentation.png'), cv2.IMREAD_GRAYSCALE)

            resized_org_img = cv2.resize(img, (256, 192))
            org_img.append(resized_org_img)

            resized_img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_CUBIC)
            X.append(resized_img)

            resized_msk = cv2.resize(msk, (256, 192), interpolation=cv2.INTER_CUBIC)
            Y.append(resized_msk)

    X = np.array(X)
    Y = np.array(Y)

    X = X / 255
    Y = Y / 255

    X = np.round(X, 0)
    Y = np.round(Y, 0)

    print(X.shape)
    print(Y.shape)

    try:
        os.makedirs('results')
    except:
        pass

    yp = model.predict(X, batch_size=1, verbose=1)

    yp = np.round(yp, 0)

    for i in tqdm(range(0, len(X))):
        plt.figure(figsize=(25, 10))

        # Plot original Image
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(org_img[i], cv2.COLOR_BGR2RGB))
        plt.title('Input')

        # Plot Prediction Image
        plt.subplot(1, 4, 2)
        plt.imshow(yp[i].reshape(yp[i].shape[0], yp[i].shape[1]))
        plt.title('Prediction')

        # Plot Prediction Contour Image
        predict_mask = np.zeros((yp[i].shape[0], yp[i].shape[1]), dtype=np.uint8)
        predict_mask = yp[i].reshape(yp[i].shape[0], yp[i].shape[1])

        plt.subplot(1, 4, 3)
        predict_mask = predict_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(predict_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(org_img[i], contours, -1, [0, 255, 0], 1, cv2.LINE_AA)
        plt.imshow(cv2.cvtColor(org_img[i], cv2.COLOR_BGR2RGB))
        plt.title('Predict Contour')

        # Plot Prediction Contour with Ground Truth Contour Image
        gt_mask = np.zeros((yp[i].shape[0], yp[i].shape[1]), dtype=np.uint8)
        gt_mask = Y[i].reshape(Y[i].shape[0], Y[i].shape[1])

        plt.subplot(1, 4, 4)
        gt_mask = gt_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(gt_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(org_img[i], contours, -1, [0, 0, 255], 1, cv2.LINE_AA)
        plt.imshow(cv2.cvtColor(org_img[i], cv2.COLOR_BGR2RGB))
        plt.title('With Ground Truth Contour')

        intersection = yp[i].ravel() * Y[i].ravel()
        union = yp[i].ravel() + Y[i].ravel() - intersection

        jacard = (np.sum(intersection) / np.sum(union))
        plt.suptitle('Jacard Index' + str(np.sum(intersection)) + '/' + str(np.sum(union)) + '=' + str(jacard))

        plt.savefig('results/' + str(i) + '.png', format='png')
        # plt.show()
        plt.close()