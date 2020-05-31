#import libraries
import os
import cv2
import math
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications.resnet50 import ResNet50, preprocess_input

class DressClassification:

    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def define_base_model(self,HEIGHT,WIDTH):
        self.base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=(HEIGHT, WIDTH, 3)
        )
        return self.base_model

    def build_finetune_model(self, dropout, fc_layers, num_classes):
        for layer in self.base_model.layers:
            layer.trainable = False

        x = self.base_model.output
        x = Flatten()(x)
        for fc in fc_layers:
            # New FC layer, random init
            x = Dense(fc, activation='relu')(x) 
            x = Dropout(dropout)(x)

        # New softmax layer
        predictions = Dense(num_classes, activation='softmax')(x) 
        
        self.finetune_model = Model(inputs=self.base_model.input, outputs=predictions)

        return self.finetune_model

    def fit(self,epoch=10,batch_size=8,dropout=0.5,fc_layers=[1024, 1024],lr=0.00001):
        
        self.define_base_model(224,224)

        self.build_finetune_model( 
            dropout=dropout, 
            fc_layers=fc_layers, 
            num_classes=5
        )

        adam = Adam(lr=lr)
        self.finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

        filepath="./" + "ResNet50" + "_model_weights.h5"
        checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
        callbacks_list = [checkpoint]

        self.history = self.finetune_model.fit(
            self.x_train, self.y_train,
            epochs=epoch,
            workers=8, 
            batch_size=batch_size, 
            validation_data=(self.x_test,self.y_test), 
            callbacks=callbacks_list
        )
        return self.history

if __name__ == '__main__':
    #model = DressClassification(x_train,x_test,y_train,y_test)
    #model.fit(
    #    epoch=10,
    #    batch_size=8,
    #    dropout=0.5,
    #    fc_layers=[1024,1024],
    #    lr=0.00001
    #)
