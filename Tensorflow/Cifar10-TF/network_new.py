import tensorflow as tf
import numpy as np

class MyFirstNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2,channel_3, num_classes):
        super(MyFirstNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation = None)
        self.relu1 = tf.keras.layers.ReLU()
        self.drop1 = tf.keras.layers.Dropout(rate = 0.5)
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation = None)
        self.relu2 = tf.keras.layers.ReLU()
        self.drop2 = tf.keras.layers.Dropout(rate = 0.5)
        self.conv3 = tf.keras.layers.Conv2D(128, 3, activation = None)
        self.relu3 = tf.keras.layers.ReLU()
        self.drop3 = tf.keras.layers.Dropout(rate = 0.5)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64, activation = 'relu')
        self.fc2 = tf.keras.layers.Dense(64, activation = 'relu')
        self.fc3 = tf.keras.layers.Dense(10)
        
    def call(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
        

    

