# Tree Health Classification

## Описание проекта

Данный проект предназначен для классификации состояния деревьев на основе данных о деревьях в Нью-Йорке.

## Установка зависимостей
```bash
pip install -r requirements.txt
```
## Запуск API

```bash
uvicorn src.api:app --reload
```

## Примеры использования API

Отправьте POST-запрос на `/predict` с JSON-данными:

```json
{
  "features": "данные"
}
```

## Архитектура модели

Используется простая полносвязная нейронная сеть с двумя скрытыми слоями. Выбор архитектуры основан на простоте и эффективности для данной задачи.
```