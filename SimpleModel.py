from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Flatten
from keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
from MILAttentionLayer import MILAttentionLayer

def SimpleModel(instance_shape,bag_size):
    """ Create Keras model for Multiply Instance Learning
    Parameters
    -------------------
    instance_shape (tuple) - shape of 1 instance in the bag
    bag_size (int) - size of the bag
    Returns
    -------------------
     keras.Model
    """

    # Extract features from inputs.
    inputs, embeddings = [], []
    conv1_1 = Conv2D(16, kernel_size=(2,2), activation='relu') 
    conv1_2 = Conv2D(16, kernel_size=(2,2), activation='relu')  
    mpool_1 = MaxPooling2D((2,2))

    conv2_1 = Conv2D(32, kernel_size=(2,2),   activation='relu')  
    conv2_2 = Conv2D(32, kernel_size=(2,2),activation='relu') 
    mpool_2 = MaxPooling2D((2,2))

    fc0 = Dense(512, activation='relu', name='fc0') 
    fc1 = Dense(512, activation='relu', name='fc1') 
    fc2 = Dense(256, activation= 'relu',  name='fc2')
  
   
    for _ in range(bag_size):
        inp = layers.Input(instance_shape)
        inputs.append(inp)
        x = conv1_1(inp)
        x = conv1_2(x)
        x = mpool_1(x)

        x = conv2_1(x)
        x = conv2_2(x)
        x = mpool_2(x)

        x = Flatten()(x)
        x = fc0(x)
        x = Dropout(0.5)(x)
        x = fc1(x)
        x = Dropout(0.5)(x)
        x = fc2(x)
        x = Dropout(0.2)(x)
        
        embeddings.append(x)

    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=1024,
        kernel_regularizer=keras.regularizers.l2(0),# previous - 0.01
        use_gated=True, 
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # Concatenate layers.
    concat = layers.concatenate(multiply_layers, axis=1)

    # Classification output node.
    output = layers.Dense(2, activation = 'softmax')(concat)

    return keras.Model(inputs, output) 
