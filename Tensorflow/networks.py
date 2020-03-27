import tensorflow as tf
import numpy as np

class MyFirstNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2,channel_3, num_classes):
        super(MyFirstNet, self).__init__()

        initializer = tf.initializers.VarianceScaling(scale=2.0)
        #initializer = tf.keras.initializers.GlorotNormal()
        self.conv1 = tf.keras.layers.Conv2D(filters = channel_1, kernel_size = [3, 3],
                                           strides = 1, padding='same', kernel_initializer=initializer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.drop1 = tf.keras.layers.Dropout(rate = 0.5)
        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size = [2, 2], strides = [2, 2])
        self.conv2 = tf.keras.layers.Conv2D(filters = channel_2, kernel_size = [3, 3],
                                           strides = 1, padding='valid', kernel_initializer=initializer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.drop2 = tf.keras.layers.Dropout(rate = 0.5)
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size = [2, 2], strides = [2, 2])
        
        self.conv3 = tf.keras.layers.Conv2D(filters = channel_3, kernel_size = [3, 3],
                                           strides = 1, padding='valid', kernel_initializer=initializer)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        self.drop3 = tf.keras.layers.Dropout(rate = 0.5)
        self.avg_pool3 = tf.keras.layers.AveragePooling2D(pool_size = [5, 5], strides = [1, 1])
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc4 = tf.keras.layers.Dense(100, kernel_initializer=initializer)
        self.fc5 = tf.keras.layers.Dense(50, kernel_initializer=initializer)
        self.fc6 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer)
        
    
    def call(self, input_tensor, training=False):

        x = input_tensor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.max_pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.max_pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.avg_pool3(x)
        
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        
        #channel_1, channel_2, num_classes = 16, 8, 10

       
        return x
