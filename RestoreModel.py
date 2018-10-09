import tensorflow as tf
from tensorflow import keras
import DataPrepare as dp
import sys
import io

test_path=sys.argv[1]
model_path=sys.argv[2]
pad_index=int(sys.argv[3])
max_length=int(sys.argv[4])
output_path=sys.argv[5]

test_data=dp.create_data(test_path)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value=pad_index,padding="post",maxlen=max_length)

model=keras.models.load_model(model_path)
results=model.predict(test_data)

with open(output_path,'w+',encoding='UTF-8') as f:
	for result in results:
		f.write("{}\n".format(result))
	f.close()