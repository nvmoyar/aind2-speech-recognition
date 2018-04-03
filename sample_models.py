from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D)
from keras.initializers import RandomUniform

# Model  0
def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = SimpleRNN(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# Model 1
def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))    
    # Add recurrent layer
    simp_rnn = SimpleRNN(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)    
    # Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)     
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)  
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# Model 2
def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer 
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation=1)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride, dilation):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): `same`, `valid`, `causal`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid', 'causal'}
    if border_mode == 'same' or border_mode == 'valid': dilation = 1
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid' or border_mode == 'causal':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

# Model 3
def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    rnn_layer = input_data
    
    #Add recurrent layers, each with batch normalization    
    for i in range(recur_layers):       
        # Add recurrent layer
        rnn_layer = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn_{}'.format(i))(rnn_layer)   
        # Add batch normalization
        rnn_layer = BatchNormalization(name="bnn_{}".format(i))(rnn_layer)
   
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

# Model 4
def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = LSTM(units, activation='relu', return_sequences=True, implementation=2, name='rnn')
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(simp_rnn)(input_data)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(bidir_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


# Model 5
def conv_rnn_model_w_init(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, recur_layers, output_dim=29):
    
    """ Build convolutional network + custom number of rnn layers
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1, seed=None),
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Initialise rnn_layer
    rnn_layer = bn_cnn
    #Add recurrent layers, each with batch normalization    
    for i in range(recur_layers):       
        # Add recurrent layer
        rnn_layer = GRU(units, activation='relu', return_sequences=True, implementation=2,
                        kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1, seed=None),
                        name='rnn_{}'.format(i))(rnn_layer)   
        # Add batch normalization
        rnn_layer = BatchNormalization(name="bnn_{}".format(i))(rnn_layer)
    
    # Add a TimeDistributed(Dense(output_dim)) layer **
    time_dense = TimeDistributed(Dense(output_dim))(rnn_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation=1)
    print(model.summary())
    return model


# Final model
def final_model(input_dim, filters, kernel_size, conv_border_mode, units, recur_layers, n_dilation, output_dim=29):
    """ Build dilated convolution network + custom number of rnn layers
    """ 
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, 
                     kernel_size, 
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=n_dilation,
                     kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1, seed=None),
                     name='conv1d')(input_data)
    # Pooling layer
    pool_layer = MaxPooling1D(pool_size=2, strides=1)(conv_1d)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(pool_layer)
    # Initialise rnn_layer
    rnn_layer = bn_cnn
    # Add a customisable number of recurrent layers, each with batch normalization    
    for i in range(recur_layers):       
        # Add recurrent layer
        rnn_layer = GRU(units, activation='relu', return_sequences=True, implementation=2,
                        kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1, seed=None),
                        dropout=0.2,
                        name='rnn_{}'.format(i))(rnn_layer)   
        # Add batch normalization
        rnn_layer = BatchNormalization(name="bnn_{}".format(i))(rnn_layer)
    
    # Add a TimeDistributed(Dense(output_dim)) layer 
    time_dense = TimeDistributed(Dense(output_dim))(rnn_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # def cnn_output_length(input_length, filter_size, border_mode, stride, dilation):
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, stride=1, dilation=n_dilation)
    print(model.summary())
    return model
