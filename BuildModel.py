import tensorflow as tf
from tensorflow import keras
import io
import DataPrepare as dp
import numpy as np


train_data_path=r'train_data.txt'
train_label_path=r'train_label.txt'

dev_data_path=r'dev_data.txt'
dev_label_path=r'dev_label.txt'

test_data_path=r'test_data.txt'
test_label_path=r'test_label.txt'

dict_path=r"dict.txt"


	
train_data=dp.create_data(train_data_path)
train_label=dp.create_label(train_label_path)

dev_data=dp.create_data(dev_data_path)
dev_label=dp.create_label(dev_label_path)

test_data=dp.create_data(test_data_path)
test_label=dp.create_label(test_label_path)

word_index_dict=dp.create_dict(dict_path)
index_word_dict=dict([(value, key) for (key,value) in word_index_dict.items()])

vocab_size=10000

train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=word_index_dict["<PAD>"],padding='post',maxlen=256)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value=word_index_dict["<PAD>"],padding="post",maxlen=256)
dev_data=keras.preprocessing.sequence.pad_sequences(dev_data,value=word_index_dict["<PAD>"],padding="post",maxlen=256)

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

results=model.evaluate(test_data,test_label)

print(results)

model=keras.models.load_model("PosNeg_model.h5")
# new_test_data=create_data(r"D:\public\tmp\Data\Post\Last.txt")
# new_test_data=keras.preprocessing.sequence.pad_sequences(new_test_data,value=word_index_dict["<PAD>"],padding="post",maxlen=256)
# pred=model.predict(new_test_data)

# with open("NewTestResultLast.txt",'w+') as f:
# 	for item in pred:
# 		f.write(str(item[0]))
# 		f.write('\n')