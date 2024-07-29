import string

import openpyxl
import pickle

import pandas as pd
from keras.src.utils import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer


model_path = '../Sequentinal/model_seq.h5'
word_list_path = '../Sequentinal/words.txt'

def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        word_list = file.read().splitlines()
    return word_list


def remove_punctuation(text):
    if isinstance(text, str):
        # Удаляем знаки препинания из текста
        cleaned_text = ''.join(char for char in text if char not in string.punctuation).lower()
        return cleaned_text
    else:
        # Если значение не является строкой, просто возвращаем его как есть
        return text

def filter_text(text, word_list):
    if isinstance(text, str):
        text = remove_punctuation(text)
        words = text.split()
        filtered_text = []
        negate = False  # флаг, который будет указывать на ТЕКУЩИЙ отрицательный контекст

        for word in words:
            if word == "не":  # если слово "не" встретилось, то мы меняем флаг отрицания
                negate = True
            elif any(root in word for root in word_list):  # проверяем наличие корня из списка слов в текущем слове
                if negate:  # если флаг отрицания установлен
                    filtered_text.append("не " + word)
                    negate = False
                else:  # если отрицание неактивно, добавляем слово без "не"
                    filtered_text.append(word)
            elif "не" in word:  # если слово само есть отрицание "ненравится" "неоч" или подобное
                filtered_text.append(word)
        return ' '.join(filtered_text)
    else:
        return str(text)



def load_data(file_path):
    workbook = openpyxl.load_workbook(file_path, read_only=True)
    sheet = workbook.active

    data_txt = []

    for row in sheet.iter_rows(min_row=1, max_row=100, values_only=True):
        text = row[0] if row[0] is not None else ""
        data_txt.append(text)

    return data_txt


def for_front_Sequentinal(review):
    loaded_model = load_model(model_path)
    with open('../Sequentinal/tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    word_list = load_word_list(word_list_path)
    # Предварительная обработка тестовых текстов
    filtered_review = filter_text(review, word_list)
    print(filtered_review)
    # Токенизация тестовых текстов и преобразование их в последовательности

    unlabeled_data = []
    unlabeled_data.append(filtered_review)

    test_sequences = loaded_tokenizer.texts_to_sequences(unlabeled_data)

    # Паддинг новых тестовых последовательностей
    padded_test_sequences = pad_sequences(test_sequences, maxlen=500, padding='post', truncating='post')

    # Предсказание на новых тестовых данных
    predictions_test = loaded_model.predict(padded_test_sequences)
    print(predictions_test)
    # predicted_sentiment = "Положительный" if predictions_test[0] > 0.5 else "Отрицательный"
    return 1 if (predictions_test[0] > 0.5) else 0


def for_front_Sequentinal_excel(destination):
    loaded_model = load_model(model_path)
    with open('../Sequentinal/tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    # Загрузка данных из Excel файла
    df = pd.read_excel(destination, usecols=[0, 1, 2], header=None)

    # Добавление нового заголовка
    # df.columns = ['Reviews', 'Predictions_Neuron1', 'Predictions_Functional']
    word_list = load_word_list(word_list_path)

    # Применение фильтрации текста к каждому отзыву
    filtered_reviews = df[0].apply(lambda x: filter_text(x, word_list))

    # Токенизация отфильтрованных отзывов и преобразование их в последовательности
    test_sequences = loaded_tokenizer.texts_to_sequences(filtered_reviews)

    # Паддинг новых тестовых последовательностей
    padded_test_sequences = pad_sequences(test_sequences, maxlen=500, padding='post', truncating='post')

    # Предсказание тональности для каждого отзыва с помощью новой нейронной сети
    predictions_sequential = loaded_model.predict(padded_test_sequences)

    # Преобразование предсказаний в формат 0 и 1
    binary_predictions_sequential = [1 if pred > 0.5 else 0 for pred in predictions_sequential]

    # Добавление предсказаний от новой нейронной сети в новый столбец DataFrame
    df['Sequential'] = binary_predictions_sequential

    # Сохранение DataFrame в Excel файл
    df.to_excel(destination, index=False, header=False)


# for_front_Sequentinal("не работает не нравится не работает")
