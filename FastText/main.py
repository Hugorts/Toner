import pandas as pd
from sklearn.model_selection import train_test_split
import fasttext

def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        word_list = file.read().splitlines()
    return word_list


# Загрузка данных из Excel файла
data = pd.read_excel('reviews_fasttext.xlsx')

# Разделение данных на обучающую и тестовую выборки (все таки это больше нужно, чтобы проверить обучаемость)
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# Сохранение данных в текстовые файлы. фасттекст работает только с ними
train_data[['Отзыв', 'Марка']].to_csv('train_fasttext.txt', header=None, index=None, sep=' ', mode='w')
test_data[['Отзыв', 'Марка']].to_csv('test_fasttext.txt', header=None, index=None, sep=' ', mode='w')

# Обучение модели. Много чего из параметров я не трогал. Модель сама добирает то, что ей надо
model = fasttext.train_supervised('train_fasttext.txt', epoch=50, lr=0.1, wordNgrams=2, dim=2, minCount=5, minn=3, maxn=5)

# Сохранение обученной модели
model.save_model("model.bin")

# Оценка качества модели на тестовой выборке
result = model.test('test_fasttext.txt')

# try-except, чтобы обработать случай деления на ноль, потому что иногда возникает
try:
    precision = result[1]
    recall = result[2]
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Всего: {result[0]}')
    print(f'Точность: {precision:.4f}, Полнота: {recall:.4f}, F1-мера: {f1_score:.4f}')

except ZeroDivisionError:
    print('Error: Произошло деление на ноль. Точность, Полнота, и F1-мера неопределены.')

# Проверка работы модели на отзывах
for review in test_data['Отзыв'].sample(100):  # Выбираем 100 случайных отзывов из тест даты чтобы посмотреть результаты
    # Убираем символы новой строки из отзыва, потому что некоторые отзывы багованные
    review = review.replace('\n', '')

    # Предсказание модели
    prediction = model.predict(review)

    # Вывод результатов
    print(f'Отзыв: {review}\nПрогноз: {prediction[0][0]}, Вероятность: {prediction[1][0]:.4f}\n')

