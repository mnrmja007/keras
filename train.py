from __future__ import division, print_function, absolute_import

import keras
from import_data import *
from network import *

# Getting the data
x_train, x_val, y_train, y_val = get_data_MNIST()

# Recalling the network defined in network.py
model = ANN(WIDTH = 28, HEIGHT = 28, CHANNELS = 1, LABELS = 10)
# Training
tbCallBack = keras.callbacks.TensorBoard(log_dir ='./logs', histogram_freq = 0, write_graph = True, write_images = True)
best_model = keras.callbacks.ModelCheckpoint(filepath = './best_checkpoint', monitor = 'val_loss', verbose = 1, 
	save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)

model.summary()
model.fit(x_train, y_train, batch_size = 200, epochs = 20, validation_data = (x_val, y_val), shuffle = True, verbose = 1, callbacks = [best_model, tbCallBack])

train_eval = model.evaluate(x_val, y_val, batch_size = 200,  verbose = 1)
print('Accuracy:', train_eval)
# Loading the best model
from keras.models import load_model
model = load_model('./best_checkpoint')

#model.load_weights('./best_checkpoint', by_name = True)

print('Loaded the best model')
print('Evaluation in progress:')
train_eval = model.evaluate(x_val, y_val, batch_size = 200, verbose = 1)
print('Accuracy:', train_eval)


from keras.utils import plot_model
plot_model(model, to_file='model.png')
#tensorboard --logdir=logs/	




