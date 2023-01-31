import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from MultiResUNet import MultiResUnet

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