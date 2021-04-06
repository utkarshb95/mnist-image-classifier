import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras import optimizers

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


(ds_train, dsvalid, ds_test), ds_info = tfds.load(
    'mnist',
# First 25% and last 25% from training, then validation data is 5%
# from 25% of train data to 30% and test is the usual 10K
    split=['train[:25%]+train[-25%:]','train[25%:30%]', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


dsvalid = dsvalid.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dsvalid = dsvalid.batch(64)
dsvalid = dsvalid.cache()
dsvalid = dsvalid.prefetch(tf.data.experimental.AUTOTUNE)

cnn_model4 = tf.keras.models.Sequential()
cnn_model4.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu', input_shape=(28,28,1)))
cnn_model4.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu'))
cnn_model4.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
cnn_model4.add(tf.keras.layers.Flatten())
cnn_model4.add(tf.keras.layers.Dense(256,activation='relu'))
cnn_model4.add(tf.keras.layers.Dense(10,activation='softmax'))

cnn_model4.summary()

cnn_model4.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
cnn_model4.fit(ds_train, epochs=15, verbose=1, validation_data=dsvalid)

results = cnn_model4.evaluate(ds_test, batch_size=128)
print("test loss, test acc:", results)