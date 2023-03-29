# Part 2 - Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



class modeling:
    
    def model(self, X_train, y_train):
        # Initialising the RNN
        
        regressor = Sequential()
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        
        # Adding the output layer
        regressor.add(Dense(units = 1))
        
        # Compiling the RNN
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        # Fitting the RNN to the Training set
        regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        
        return regressor
        
    