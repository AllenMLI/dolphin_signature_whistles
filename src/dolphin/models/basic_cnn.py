"""Custom models"""
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Dense


def three_layer_cnn(input_shape: tuple =None,
                    alpha=None,
                    include_top: bool =True,
                    weights=None,
                    input_tensor=None,
                    pooling=None,
                    classes: int =None,
                    classifier_activation: str ='softmax'):
    """
    Basic 3-layer CNN for comparison

    Args:
        input_shape (tuple): data's input shape to be feed into the model
        alpha: Not implemented
        include_top (bool): to include the top layer for classification or not
        weights: Not implemented.
        input_tensor: Not implemented
        pooling: Not implemented. Default to max pooling
        classes (int): number of classes to classifier
        classifier_activation (str): activation for the last layer fo the model
    Returns:
        A custom Keras model
    """
    inputs = Input(shape=input_shape)

    # first local layer
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
               input_shape=input_shape)(inputs)
    x = MaxPool2D(2, 2)(x)
    x = Dropout(0.5)(x)

    # second local layer
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D(2, 2)(x)
    x = Dropout(0.5)(x)

    # third local layer
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D(2, 2)(x)
    x = Dropout(0.5)(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=classes, activation=classifier_activation)(x)

    model = Model(inputs=inputs, outputs=x)
    return model
