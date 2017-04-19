import keras 
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Conv2D, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model



# Building the network
def ANN(WIDTH, HEIGHT, CHANNELS, LABELS):
    dropout_value = 0.35

    # Building the network
    x = Input(shape = (WIDTH, HEIGHT, CHANNELS), name = 'Input')

    # Branch 1
    branch1 = Conv2D(10, (2, 2), activation = 'relu', padding = 'same', name = 'B1Conv2d_2x2')(x)
    #branch1 = dropout(branch1, dropout_value)

    # Branch 2
    branch2 = Conv2D(10, (2, 2), activation = 'relu', padding = 'same', name = 'B2Conv2d_2x2')(branch1)
    #branch2 = dropout(branch2, dropout_value)
    branch2 = Flatten()(branch2)

    # Fully connected 1
    full_1 = Dense(100, activation='relu', name = 'Dense_1')(branch2)
    full_1 = Dropout(dropout_value)(full_1)

    # Fully connected 2
    full_2 = Dense(100, activation='relu', name = 'Dense_2')(full_1)
    full_2 = Dropout(dropout_value)(full_2)
    
    # Output layer
    y = Dense(LABELS, activation = 'softmax', name = 'Output')(full_2)

    model = Model(inputs = x , outputs = y)
    optimizer = Adam()
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    print('The model has been successfully compiled!')

    plot_model(model, to_file = 'model.png')
    print("Topology of the model has been successfully saved!")
    return model

