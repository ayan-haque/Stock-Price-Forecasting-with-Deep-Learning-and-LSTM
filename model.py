from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

class modeling:
    
    def model(self, X_train, y_train):
        
        regressor = Sequential()
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        return regressor
        
    
