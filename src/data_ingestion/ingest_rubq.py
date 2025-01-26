import os
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def ingest_rubq(json_path, db_path, limit):
    """
    Загружает 'limit' вопросов из RuBQ JSON-файла в локальную Chroma DB.

    :param json_path: Путь к JSON (например, D:/_neoflex_bot/data/rubq/RuBQ_2.0_dev.json).
    :param db_path: Папка, где хранить базу Chroma.
    :param limit: Сколько записей проиндексировать (для теста).
    :return: (collection, model) - объект коллекции Chroma и модель эмбеддингов
    """

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    client = chromadb.PersistentClient(path=db_path)
    print("PersistentClient для Chroma инициализирован.")

    # получаем (или создаём) коллекцию "rubq_qa"
    collection = client.get_or_create_collection("rubq_qa")
    print(f"Коллекция 'rubq_qa' создана или подключена.")

    #  модель SentenceTransformer (русская поддержка)
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit and len(data) > limit:
        data = data[:limit]

    for i, item in enumerate(data):
        uid = item.get("uid", f"item_{i}")
        question = item.get("question_text", "")
        answer = item.get("answer_text", "")

        doc_text = f"Вопрос: {question}\nОтвет: {answer}"

        embedding = model.encode(doc_text)

        doc_id = f"rubq_dev_{uid}"

        collection.add(
            documents=[doc_text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{
                "uid": uid,
                "question": question,
                "answer": answer
            }]
        )

        print(f"Indexed item {i+1}/{len(data)} (uid={uid})")

    print(f"\n---\nИндексирование завершено! Всего загружено {len(data)} записей.")
    print(f"Chroma DB сохранена в: {os.path.abspath(db_path)}")

    return collection, model


def test_search(collection, model, query: str, top_k=3):
    """
    Выполняет тестовый поиск в коллекции Chroma.
    :param collection: коллекция (вернулась из ingest_rubq)
    :param model: модель (тоже вернулась из ingest_rubq)
    :param query: текст запроса
    :param top_k: сколько ближайших документов вернуть
    """
    print(f"\nПоиск ближайших документов к запросу: {query}")

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print("\nРезультаты поиска:")
    print(results)

    #  несколько самых близких результатов:
    for i in range(top_k):
        if i >= len(results["documents"][0]):
            break

        doc_id = results["ids"][0][i]
        doc_text = results["documents"][0][i]
        meta = results["metadatas"][0][i]

        print(f"\n--- Результат #{i+1} ---")
        print(f"ID: {doc_id}")
        print(f"Текст: {doc_text}")
        print(f"Метаданные: {meta}")


if __name__ == "__main__":
    """
    Задаём параметры и вызываем функции:
    1) ingest_rubq (загружаем limit вопросов в базу),
    2) test_search (запрос: "Где находится Летний сад?").
    """

    JSON_FILE = "D:/_neoflex_bot/data/rubq/RuBQ_2.0_dev.json"
    DB_PATH = "D:/_neoflex_bot/data/chroma_db"

    # Сколько записей проиндексировать
    LIMIT = None

    collection_obj, model_obj = ingest_rubq(JSON_FILE, DB_PATH, LIMIT)

    SEARCH_QUERY = "Где находится Летний сад?"
    test_search(collection_obj, model_obj, SEARCH_QUERY, top_k=3)
