# Part 3 - Making the predictions and visualising the results

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class prediction():
    
    def pred(self, dataset_train, sc, regressor):

        dataset = pd.read_csv('Tata_Global_Stock_Price_Test.csv')#importing the dataset
        real_stock_price = dataset.iloc[:, 1:2].values#from the dataset the column is sliced to for getting the real stock price
        
        # in the block below trying to get the predicted stock prices
        dataset_combined = pd.concat((dataset_train['Open'], dataset['Open']), axis = 0)#combining the columns to input previous 60 samples
        inputs = dataset_combined[len(dataset_combined) - len(dataset) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(60, 80):
            X_test.append(inputs[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_price = regressor.predict(X_test)
        predicted_price = sc.inverse_transform(predicted_price)
        
        # Visualising the output between the real and predicted price
        plt.plot(real_stock_price, color = 'red', label = 'Original Stock Price')
        plt.plot(predicted_price, color = 'blue', label = 'Predicted Stock Price')
        plt.title('Tata Global Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Tata Global Stock Price')
        plt.legend()
        plt.show()