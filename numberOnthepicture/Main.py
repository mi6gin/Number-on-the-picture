import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from PIL import Image

# Загрузка данных MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация значений пикселей
train_images, test_images = train_images / 255.0, test_images / 255.0

# Создание модели
model = keras.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(train_images, train_labels, epochs=5)

# Оценка точности на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nТочность на тестовых данных: {test_acc}')

# Загрузка изображения 'example.png' и предсказание цифры
image = Image.open('example.png').convert('L')
image = image.resize((28, 28))
image = np.array(image) / 255.0
predictions = model.predict(np.array([image]))
predicted_digit = np.argmax(predictions)
print(f'Предсказанная цифра: {predicted_digit}')
