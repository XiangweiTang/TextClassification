import tensorflow as tf
from tensorflow import keras
import io
import numpy as np


train_data_path=r'D:\XiangweiTang\Python\TextClassification\train_data.txt'
train_label_path=r'D:\XiangweiTang\Python\TextClassification\train_label.txt'

dev_data_path=r'D:\XiangweiTang\Python\TextClassification\dev_data.txt'
dev_label_path=r'D:\XiangweiTang\Python\TextClassification\dev_label.txt'

test_data_path=r'D:\XiangweiTang\Python\TextClassification\test_data.txt'
test_label_path=r'D:\XiangweiTang\Python\TextClassification\test_label.txt'

dict_path=r"D:\XiangweiTang\Python\TextClassification\dict.txt"

def create_data_itor(file_path):
	with open(file_path,'r',encoding='UTF-8') as f:		
		for line in f:
			split=line.split(' ')
			yield list(map(lambda x:int(x),split))

def create_data(file_path):
	itor=create_data_itor(file_path)
	return np.array(list(itor))

def create_label_itor(file_path):
	with open(file_path,'r',encoding='UTF-8') as f:
		for line in f:
			yield int(line)

def create_label(file_path):
	itor=create_label_itor(file_path)
	return np.array(list(itor))

def create_dict(file_path):
	dictionary={}
	with open(file_path,'r',encoding='UTF-8') as f:
		for line in f:
			key=line.split('\t')[0]
			value=int(line.split('\t')[1])+1
			dictionary[key]=value
	dictionary['<UNK>']=1
	dictionary['<PAD>']=0
	return dictionary
	
train_data=create_data(train_data_path)
train_label=create_label(train_label_path)

dev_data=create_data(dev_data_path)
dev_label=create_label(dev_label_path)

test_data=create_data(test_data_path)
test_label=create_label(test_label_path)

word_index_dict=create_dict(dict_path)
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

#model=build_model()

#model.fit(train_data,train_label, epochs=40, batch_size=512, validation_data=(dev_data,dev_label),verbose=1)


#model.save("PosNeg_model.h5")

#results=model.evaluate(test_data,test_label)

#print(results)

model=keras.models.load_model("PosNeg_model.h5")

pred=model.predict(test_data)

with open("TestResult.txt",'w+') as f:
	for item in pred:
		f.write(str(item[0]))
		f.write('\n')