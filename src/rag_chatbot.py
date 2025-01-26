import chromadb
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

from openai import OpenAI

def init_chroma_and_model(db_path):
    """
    Подключаемся к локальной Chroma, загружаем коллекцию rubq_paragraphs,
    инициализируем модель эмбеддингов (ту же, что при индексации).
    """
    # Инициализируем Chroma
    client = chromadb.PersistentClient(path=db_path)
    collection_name = "rubq_paragraphs"
    #collection_name = "rubq_qa"
    #collection_name = "rubq_qa_model2"
    collection = client.get_collection(collection_name)

    # Ту же модель, что при ingest (distiluse-base-multilingual-cased-v2).
    #model = SentenceTransformer("cointegrated/rubert-tiny2")
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

    return collection, model


def search_in_collection(query, collection, model, top_k=3):
    """
    Поиск по нашей коллекции .
    Возвращает список (text, metadata, doc_id).
    """
    query_emb = model.encode(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    matched = []
    for i in range(len(results["documents"][0])):
        doc_text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        doc_id = results["ids"][0][i]
        matched.append((doc_text, meta, doc_id))
    return matched


def duckduckgo_search(query, max_results=3):
    """
    Поиск с помощью DuckDuckGo, используя DDGS.
    Возвращаем список (snippet, url).
    """
    snippets = []
    with DDGS() as ddgs:
        results_gen = ddgs.text(query, safesearch="Off", timelimit=None)
        for i, r in enumerate(results_gen):
            if i >= max_results:
                break
            snippet = r.get("body", "")
            url = r.get("href", "")
            snippets.append((snippet, url))
    return snippets


def call_local_llm(prompt):
    """
    Вызываем локальную LLM (llama3.1)
    """

    client = OpenAI(base_url="http://localhost:11434/v1", api_key='1')
    response = client.chat.completions.create(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content


def run_chat_loop(collection, model):
    """
    Основной цикл диалога в консоли.
    """
    conversation_history = []

    while True:
        user_input = input("\nПользователь: ")
        if user_input.strip().lower() in ["exit", "quit", "выход"]:
            print("Чат завершён.")
            break

        # 1. Поиск по локальной базе
        matched_docs = search_in_collection(user_input, collection, model, top_k=2)

        # Простейшая логика определения «релевантности»
        fallback_to_duck = False
        if not matched_docs:
            fallback_to_duck = True
        else:
            first_doc_text = matched_docs[0][0]
            if len(first_doc_text.strip()) < 20:  # очень грубо
                fallback_to_duck = True

        if fallback_to_duck:
            ddg_snippets = duckduckgo_search(user_input, max_results=3)
            context_text = ""
            for snippet, url in ddg_snippets:
                context_text += f"Snippet: {snippet}\nUrl: {url}\n\n"
            if not context_text.strip():
                context_text = "Не удалось найти информацию в DuckDuckGo."
        else:
            context_text = ""
            for doc, meta, doc_id in matched_docs:
                context_text += f"DocID: {doc_id}\n{doc}\n\n"

        # История диалога
        history_str = "\n".join(conversation_history[-6:])

        prompt = (
            f"История диалога:\n{history_str}\n\n"
            f"Пользователь спрашивает: {user_input}\n\n"
            f"Вот найденные материалы:\n{context_text}\n\n"
            "Сформулируй развёрнутый ответ. Если нет уверенности, скажи, что не знаешь.\n"
        )

        try:
            answer = call_local_llm(prompt)
        except Exception as e:
            print("Ошибка при вызове локальной LLM:", e)
            answer = "Извините, не могу сейчас ответить."

        print(f"Бот: {answer}")

        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Bot: {answer}")


if __name__ == "__main__":
    DB_PATH = "D:/_neoflex_bot/data/chroma_db"
    coll, emb_model = init_chroma_and_model(DB_PATH)
    run_chat_loop(coll, emb_model)
