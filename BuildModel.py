import tensorflow as tf
from tensorflow import keras
import io
import DataPrepare as dp
import numpy as np
import sys

train_data_path=sys.argv[1]
train_label_path=sys.argv[2]

dev_data_path=sys.argv[3]
dev_label_path=sys.argv[4]

dict_path=sys.argv[5]

pad_index=int(sys.argv[6])
max_length=int(sys.argv[7])
	
train_data=dp.create_data(train_data_path)
train_label=dp.create_label(train_label_path)

dev_data=dp.create_data(dev_data_path)
dev_label=dp.create_label(dev_label_path)

word_index_dict=dp.create_dict(dict_path)
index_word_dict=dict([(value, key) for (key,value) in word_index_dict.items()])

vocab_size=10000

train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=pad_index,padding='post',maxlen=max_length)
dev_data=keras.preprocessing.sequence.pad_sequences(dev_data,value=pad_index,padding="post",maxlen=max_length)

def build_model():
	model=keras.Sequential()
	model.add(keras.layers.Embedding(vocab_size,16))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(16,activation=keras.activations.relu))
	model.add(keras.layers.Dense(1,activation=keras.activations.sigmoid))

	model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
	return model

model=build_model()

model.fit(train_data,train_label, epochs=40, batch_size=512, validation_data=(dev_data,dev_label),verbose=1)


model.save("PosNeg_model.h5")
print("Model is built.")