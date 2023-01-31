from data_split import data_split
import model_IO
from MultiResUNet import MultiResUnet
import metrix

''' Train & Evaluate Model '''
if __name__ == "__main__":
    model = MultiResUnet(height=192, width=256, n_channels=3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrix.dice_coef, metrix.jacard, 'accuracy'])

    model_IO.saveModel(model)

    fp = open('models/log.txt', 'w')
    fp.close()
    fp = open('models/best.txt', 'w')
    fp.write('-1.0')
    fp.close()

    X_train, Y_train, X_test, Y_test = data_split()

    model_IO.trainStep(model, X_train, Y_train, X_test, Y_test, epochs=150, batchSize=10)