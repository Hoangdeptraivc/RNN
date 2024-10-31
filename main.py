import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Input,LSTM,Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tải dữ liệu đánh giá IMDB
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Tạo bộ phân loại (tokenizer) thủ công
tokenizer = Tokenizer()
train_texts = [text.numpy().decode('utf-8') for text, _ in train_dataset]
tokenizer.fit_on_texts(train_texts)

# Chuyển đổi văn bản thành chuỗi
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_labels = [label.numpy() for _, label in train_dataset]
train_sequences = pad_sequences(train_sequences)

# Chuẩn bị dữ liệu kiểm tra theo cách tương tự
test_texts = [text.numpy().decode('utf-8') for text, _ in test_dataset]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_labels = [label.numpy() for _, label in test_dataset]
test_sequences = pad_sequences(test_sequences)

# Định nghĩa kích thước bộ đệm và kích thước lô
BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Tạo các tập dữ liệu TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).batch(BATCH_SIZE)

# Định nghĩa mô hình RNN
model1 = Sequential()
model1.add(Input(shape=(None,)))  # Sử dụng lớp Input để xác định hình dạng đầu vào
model1.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))  # +1 cho chỉ số đệm
model1.add(SimpleRNN(32, return_sequences=False))
model1.add(Dense(16, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))

# Biên dịch mô hình
#model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Hiển thị tóm tắt mô hình
#model1.summary()
#model1.fit(train_dataset, epochs=10, validation_data=test_dataset)


# lstm
model2 = Sequential()
model2.add(Input(shape=(None,)))
model2.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))
model2.add(LSTM(32, return_sequences=False))
model2.add(Dense(16, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()
model2.fit(train_dataset, epochs=10, validation_data=test_dataset)
#Bidirectional
model3 = Sequential()
model3.add(Input(shape=(None,)))
model3.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))
model3.add(LSTM(32, return_sequences=True))
model3.add(Dense(16, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
#model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model3.summary()
#model3.fit(train_dataset, epochs=10, validation_data=test_dataset)

