import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    # Пример предобработки
    df = df.dropna()  # Удаляем пропущенные значения

    # Кодируем целевую переменную
    le = LabelEncoder()
    df['health'] = le.fit_transform(df['health'])

    # Разделяем признаки и целевую переменную
    X = df.drop(columns=['health'])
    y = df['health']

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le