import os
import json
import chromadb
from sentence_transformers import SentenceTransformer


def ingest_rubq_paragraphs(json_path, db_path, limit=None):
    """
    Индексируем (ingest) абзацы из RuBQ_2.0_paragraphs.json в Chroma DB.

    :param json_path: путь к RuBQ_2.0_paragraphs.json
    :param db_path: папка для сохранения/чтения Chroma
    :param limit: сколько абзацев проиндексировать (int), или None для всех
    :return: (collection, model)
    """
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    client = chromadb.PersistentClient(path=db_path)
    print("Chroma PersistentClient инициализирован.")

    collection_name = "rubq_paragraphs"
    collection = client.get_or_create_collection(collection_name)

    #  модель эмбеддингов (пусть пока distiluse-base-multilingual-cased-v2)
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit and len(data) > limit:
        data = data[:limit]

    for i, item in enumerate(data):
        uid = item.get("uid", f"para_{i}")
        text = item.get("text", "")

        embedding = model.encode(text)

        doc_id = f"rubq_para_{uid}"

        metadata = {
            "ru_wiki_pageid": item.get("ru_wiki_pageid", None)
        }

        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata]
        )

        print(f"Indexed paragraph {i + 1}/{len(data)} (uid={uid})")

    print(f"\n---\nИндексирование абзацев завершено! Всего загружено {len(data)}.")
    print(f"Коллекция: {collection_name}, DB_PATH={os.path.abspath(db_path)}")

    return collection, model


def test_search_paragraphs(collection, model, query, top_k=3):
    """
    Пробный поиск по абзацам (rubq_paragraphs).
    :param collection: объект коллекции Chroma
    :param model: SentenceTransformer (та же, что при ingest)
    :param query: строка поиска
    :param top_k: сколько документов вывести
    """
    print(f"\nПоиск по абзацам, запрос: {query}")
    query_emb = model.encode(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    for i in range(top_k):
        if i >= len(results["documents"][0]):
            break
        doc_id = results["ids"][0][i]
        doc_text = results["documents"][0][i]
        meta = results["metadatas"][0][i]

        print(f"\n--- Результат #{i + 1} ---")
        print(f"ID: {doc_id}")
        print(f"Текст: {doc_text}")
        print(f"Метаданные: {meta}")


if __name__ == "__main__":
    JSON_FILE = "D:/_neoflex_bot/data/rubq/RuBQ_2.0_paragraphs.json"
    DB_PATH = "D:/_neoflex_bot/data/chroma_db"

    # Сколько абзацев берём (None - все, или число)
    LIMIT = None

    collection_paras, model_paras = ingest_rubq_paragraphs(JSON_FILE, DB_PATH, LIMIT)

    test_query = "Что известно о хоккейном клубе ЦСКА из Москвы?"
    test_search_paragraphs(collection_paras, model_paras, test_query, top_k=3)
