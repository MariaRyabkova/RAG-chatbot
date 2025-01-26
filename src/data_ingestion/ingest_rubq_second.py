import os
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def ingest_rubq_second(json_path, db_path, limit=50):
    """
    Индексируем 'limit' вопросов из json_path в Chroma,
    используя вторую модель эмбеддингов -- более лёгкую 'cointegrated/rubert-tiny2'.
    """

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    client = chromadb.PersistentClient(path=db_path)
    print("Клиент Chroma инициализирован.")

    collection = client.get_or_create_collection("rubq_qa_model2")

    # Лёгкая русскоязычная модель
    model = SentenceTransformer("cointegrated/rubert-tiny2")
    # model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

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

        doc_id = f"rubq_dev2_{uid}"

        collection.add(
            documents=[doc_text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{"uid": uid, "question": question, "answer": answer}]
        )

        print(f"Second-model indexed {i+1}/{len(data)} (uid={uid})")

    print(f"\n---\nВторая (лёгкая) модель: индексирование завершено! Всего загружено {len(data)} записей.")
    print(f"Chroma DB хранится в: {os.path.abspath(db_path)}")

    return collection, model


def test_search_second(collection, model, query: str, top_k=3):
    """
    Тестовый поиск в коллекции rubq_qa_model2
    """
    print(f"\nПоиск ближайших документов (model2) к запросу: {query}")

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

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
    JSON_FILE = "D:/_neoflex_bot/data/rubq/RuBQ_2.0_dev.json"
    DB_PATH = "D:/_neoflex_bot/data/chroma_db"
    LIMIT = None

    collection2, model2 = ingest_rubq_second(JSON_FILE, DB_PATH, LIMIT)
    test_query = "Где находится Париж?"
    test_search_second(collection2, model2, test_query)