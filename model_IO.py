import os
import numpy as np
import matplotlib.pyplot as plt

''' Save Model '''
def saveModel(model):
    model_json = model.to_json()

    try:
        os.makedirs('models')
    except:
        pass

    fp = open('models/modelP.json', 'w')
    fp.write(model_json)
    model.save_weights('models/modelW.h5')

''' Evaluate Model '''
def evaluateModel(model, X_test, Y_test, batchSize):
    try:
        os.makedirs('results')
    except:
        pass

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)

    yp = np.round(yp, 0)

    for i in range(10):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1, 3, 2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0], Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1, 3, 3)
        plt.imshow(yp[i].reshape(yp[i].shape[0], yp[i].shape[1]))
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jacard = (np.sum(intersection) / np.sum(union))
        plt.suptitle('Jacard Index' + str(np.sum(intersection)) + '/' + str(np.sum(union)) + '=' + str(jacard))

        plt.savefig('results/' + str(i) + '.png', format='png')
        plt.close()

    jacard = 0
    dice = 0

    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()

        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection) / np.sum(union))

        dice += (2. * np.sum(intersection)) / (np.sum(yp_2) + np.sum(y2))

    jacard /= len(Y_test)
    dice /= len(Y_test)

    print('Jacard Index : ' + str(jacard))
    print('Dice Coefficient : ' + str(dice))

    fp = open('models/log.txt', 'a')
    fp.write(str(jacard) + '\n')
    fp.close()

    fp = open('models/best.txt', 'r')
    best = fp.read()
    fp.close()

    if (jacard > float(best)):
        print('***********************************************')
        print('Jacard Index improved from ' + str(best) + ' to ' + str(jacard))
        print('***********************************************')
        fp = open('models/best.txt', 'w')
        fp.write(str(jacard))
        fp.close()

        saveModel(model)

''' Train Model '''
def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch+1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1)

        evaluateModel(model,X_test, Y_test,batchSize)

    return model