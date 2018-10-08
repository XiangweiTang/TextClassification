import tensorflow as tf
from tensorflow import keras

#Data preparation
imdb=keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels)=imdb.load_data(num_words=10000)



word_index=imdb.get_word_index()

word_index={k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"]=0
word_index["<START>"]=1
word_index["<UNK>"]=2
word_index["<UNUSED>"]=3

reversed_word_index=dict([(value, key) for (key,value) in word_index.items()])

def decode_review(text):
	return ' '.join([reversed_word_index.get(i,'?') for i in text])

for i in range(10):
	print(decode_review(train_data[i]))
	print(train_labels[i])
	print()

train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post', maxlen=256)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value=word_index['<PAD>'], padding='post', maxlen=256)


vocab_size=10000

def build_model():
	model=keras.Sequential()
	model.add(keras.layers.Embedding(vocab_size,16))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(16,activation=keras.activations.relu))
	model.add(keras.layers.Dense(1,activation=keras.activations.sigmoid))

	model.compile(optimizer=keras.optimizers.Adam() ,loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

	return model

model=build_model()

x_val=train_data[:10000]
partial_x_train=train_data[10000:]

y_val=train_labels[:10000]
partial_y_train=train_labels[10000:]

model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)

model.save("imdb_model.h5")

results=model.evaluate(test_data,test_labels)

print(results)