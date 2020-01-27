from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Conv1D
from keras import optimizers 
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.models import Sequential

class Model:
  def __init__(self, config):
    self.config = config
    self.model_type = None
    self.model = None
    self.rec_models = ['lstm', 'gru']
    self.conv_models = ['cnn']
    self.units = {
      'lstm': LSTM,
      'gru': GRU,
      'rnn': SimpleRNN,
      'cnn': Conv1D
    }

  def get_optimiser(self):
    network = self.config.network
    lr = network['learning_rate']
    if network['optimiser'] == 'adam':
      return optimizers.Adam(lr=lr)
    elif network['optimiser'] == 'sgd':
      return optimizers.SGD(lr=lr, momentum=network['momentum'], nesterov=network['nesterov'])
    elif network['optimiser'] == 'nadam':
      return optimizers.Nadam(lr=lr)
    elif network['optimiser'] == 'rmsprop':
      return optimizers.RMSprop(lr=lr)
    else:
      print('| Invalid optimiser')
      exit()

  def build_model(self, input_shape):

    # Parameter configurations
    network = self.config.network
    arch = network['architecture']
    layers = network['layers']
    dropout = network['dropout']
    arch_type = self.units[arch]
    model = Sequential()

    if arch in self.rec_models:
      print('| Building recurrent model ...')
      
      # Stack layers
      for units in layers[:-1]:
        model.add(arch_type(units, return_sequences=True, activation=network['activation_rec']))
        if dropout:
          model.add(Dropout(dropout))
      model.add(arch_type(layers[-1], return_sequences=False))

    elif arch in self.conv_models:
      print('| Building convolutional model ...')
      kernel_size = network['kernel_size']
      params = ['strides', 'padding', 'dilation_rate', 'activation']
      params = dict([(param, network[param]) for param in params])

      # Stack layers
      model.add(arch_type(layers[0], kernel_size=kernel_size, input_shape=(input_shape[1], input_shape[2]), **params))
      for units in layers[1:]:
        model.add(arch_type(filters=units, kernel_size=kernel_size, **params))
        if dropout:
          model.add(Dropout(dropout))
      model.add(Flatten())
    else:
      print('| Invalid architecture')
      exit()

    # Add dense and class distribution layer    
    model.add(Dense(self.config.data['classes']))
    optimiser = self.get_optimiser()
    if self.config.usecase == self.config.FOOTBALL:
      model.add(Activation('sigmoid'))
      model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    else:
      model.add(Activation('softmax'))
      model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

    self.model = model
    return model




  def train_model(self, x_train, y_train):

    # Hyperparameters
    data_config = self.config.data
    patience = self.config.data['es_patience']
    batches = data_config['batches']
    epoch = data_config['epoch']
    val_size = data_config['val_size']

    # Early stopping measures
    es_loss = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    es_acc = EarlyStopping(monitor='val_acc', mode='max', patience=epoch)
    csv_log = CSVLogger(f'{self.config.log_dir}/_train.log', append=True)
    callbacks = [es_acc, csv_log]

    if data_config['es_loss']:
      print('| Using ES on loss ...')
      callbacks.append(es_loss)
    
    # Train model
    history = self.model.fit(x_train, y_train, 
                  batch_size=batches,
                  epochs=epoch,
                  validation_split=val_size,
                  callbacks=callbacks)
    return history