# coding: utf-8

import keras

from keras.initializers import he_normal
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from keras.layers import GlobalAveragePooling2D, Input, Lambda, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def vgg19(input_shape=(32, 32, 3), num_classes=10, weight_decay=1e-4, dropout=0.5,
          input_norm=None, use_logits=False):
    img_input = Input(shape=input_shape)

    if input_norm is None:
        x = img_input
    else:
        x = Lambda(lambda x: input_norm(x))(img_input)

    kernel_regularizer = keras.regularizers.l2(weight_decay)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block1_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block3_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block4_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=kernel_regularizer,
               kernel_initializer=he_normal(), name='block5_conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # model modification for cifar-10
    x = Flatten(name='flatten')(x)
    x = Dense(4096, use_bias = True, kernel_regularizer=kernel_regularizer,
              kernel_initializer=he_normal(), name='fc_cifa10')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(4096, kernel_regularizer=kernel_regularizer,
              kernel_initializer=he_normal(), name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(num_classes, kernel_regularizer=kernel_regularizer,
              kernel_initializer=he_normal(), name='predictions_cifa10')(x)
    x = BatchNormalization()(x)
    if not use_logits:
        x = Activation('softmax')(x)

    model = Model(img_input, x)

    return model


def nin(input_shape=(32, 32, 3), num_classes=10, weight_decay=1e-4, dropout=0.5,
        input_norm=None, use_logits=False):
    img_input = Input(shape=input_shape)

    if input_norm is None:
        x = img_input
    else:
        x = Lambda(lambda x: input_norm(x))(img_input)

    kernel_regularizer = keras.regularizers.l2(weight_decay)

    x = Conv2D(192, (5, 5), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)
    x = Conv2D(160, (1, 1), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)
    x = Conv2D( 96, (1, 1), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Dropout(dropout)(x)

    x = Conv2D(192, (5, 5), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)
    x = Conv2D(192, (1, 1), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)
    x = Conv2D(192, (1, 1), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(x)

    x = Dropout(dropout)(x)

    x = Conv2D(192, (3, 3), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)
    x = Conv2D(192, (1, 1), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)
    x = Conv2D(num_classes, (1, 1), padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    if not use_logits:
        x = Activation('softmax')(x)

    model = Model(img_input, x)

    return model


def resnet(num_layers=3, input_shape=(32, 32, 3), num_classes=10, weight_decay=1e-4,
           input_norm=None, use_logits=False):
    # total layers = 6 * n + 2
    #  3 -> resnet20
    #  5 -> resnet32
    # 18 -> resnet110
    stack_n = (num_layers - 2) // 6

    img_input = Input(shape=input_shape)

    if input_norm is None:
        x = img_input
    else:
        x = Lambda(lambda x: input_norm(x))(img_input)

    kernel_regularizer = keras.regularizers.l2(weight_decay)

    def residual_block(x, o_filters, increase=False):
        stride = (1,1)
        if increase:
            stride = (2,2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3,3), strides=stride, padding='same',
                        kernel_initializer=he_normal(),
                        kernel_regularizer=kernel_regularizer)(o1)
        o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3,3), strides=(1,1), padding='same',
                        kernel_initializer=he_normal(),
                        kernel_regularizer=kernel_regularizer)(o2)

        if increase:
            projection = Conv2D(o_filters, kernel_size=(1,1), strides=(2,2), padding='same',
                                kernel_initializer=he_normal(),
                                kernel_regularizer=kernel_regularizer)(o1)
            block = keras.layers.add([conv_2, projection])
        else:
            block = keras.layers.add([conv_2, x])

        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
               kernel_initializer=he_normal(),
               kernel_regularizer=kernel_regularizer)(x)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(num_classes, kernel_initializer=he_normal(),
              kernel_regularizer=kernel_regularizer)(x)
    if not use_logits:
        x = Activation('softmax')(x)

    model = Model(img_input, x)

    return model


def cnn(input_shape=(32, 32, 3), num_classes=18, use_logits=False):
    img_input = Input(shape=input_shape)

    x = Conv2D( 64, (8, 8), strides=(2, 2), padding='same')(img_input)
    x = Activation('relu')(x)
    x = Conv2D(128, (6, 6), strides=(2, 2), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (5, 5), strides=(1, 1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(num_classes)(x)
    if not use_logits:
        x = Activation('softmax')(x)

    model = Model(img_input, x)

    return model
