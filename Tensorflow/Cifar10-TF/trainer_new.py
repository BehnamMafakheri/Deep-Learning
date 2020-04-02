    
import tensorflow as tf
import numpy as np
from data_utils import *

def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1, device = '/cpu:0'):

    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(64)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
  
    with tf.device(device):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # Compute the loss like we did in Part II
        
        model = model_init_fn()
        optimizer = optimizer_init_fn()


        for epoch in range(num_epochs):
            
            # Reset the metrics - https://www.tensorflow.org/alpha/guide/migration_guide#new-style_metrics
            train_loss.reset_states()
            train_accuracy.reset_states()
            train_loss.reset_states()
            train_accuracy.reset_states()

            for images, labels in train_ds:
                train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy)

            for val_images, val_labels in val_ds:
                val_step(val_images, val_labels, model, loss_object, val_loss, val_accuracy)
                
                
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
            print(template.format(epoch+1,
                                  train_loss.result(),
                                  train_accuracy.result()*100,
                                  val_loss.result(),
                                  val_accuracy.result()*100))
        

def train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)
    
    
def val_step (images, labels, model, loss_object, val_loss, val_accuracy):
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)

    val_loss(v_loss)
    val_accuracy(labels, predictions)