import os
import chromadb
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

import operator
from typing import TypedDict, Annotated

from openai import OpenAI

def call_local_llm(messages, base_url="http://localhost:11434/v1", model="llama3.1"):
    """
    Вызывает локальную LLM
    messages: список словарей вида {"role": "...", "content": "..."}
    Возвращает content строки (ответ бота).
    """
    client = OpenAI(base_url=base_url, api_key="1")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=128,    # м варьировать
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM Error: {e}]"


def init_chroma_collection(db_path, collection_name="rubq_paragraphs"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(collection_name)

def init_embedding_model():
    return SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
    #return SentenceTransformer("cointegrated/rubert-tiny2")

def search_in_collection(query, collection, emb_model, top_k=2):
    """
    Векторный поиск (возвращаем список текстов).
    """
    query_emb = emb_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    docs = results["documents"][0]
    return docs  # list[str]


# Fallback: DuckDuckGo
def duckduckgo_search(query, max_results=3):
    """
    Возвращаем короткие сниппеты.
    """
    snippets = []
    with DDGS() as ddgs:
        gen = ddgs.text(query, safesearch="Off", timelimit=None)
        for i, r in enumerate(gen):
            if i >= max_results:
                break
            snippet = r.get("body", "")
            url = r.get("href", "")
            snippets.append(f"{snippet} (Source: {url})")
    if not snippets:
        snippets = ["No results from DuckDuckGo."]
    return snippets

class BotState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # история (HumanMessage, AIMessage)


def user_input_node(state: BotState):
    """
    Узел-обработчик для чтения пользовательского ввода
    """
    return {}

def bot_logic_node(state: BotState):
    """
    Узел, в котором весь RAG-процесс:
     - смотрим последний HumanMessage,
     - делаем локальный поиск (или DuckDuckGo),
     - формируем финальный промпт,
     - вызываем LLM,
     - добавляем AIMessage в state.
    """
    messages = state["messages"]
    last_msg = messages[-1]  # должен быть HumanMessage
    query = last_msg.content

    local_docs = search_in_collection(query, collection_obj, emb_model, top_k=2)

    fallback = False
    if not local_docs:
        fallback = True
    else:
        if len(local_docs[0].strip()) < 20:
            fallback = True

    if fallback:
        ddg_results = duckduckgo_search(query, max_results=2)
        context_text = "\n".join(ddg_results)
    else:
        context_text = "\n".join(local_docs)

    user_role_msg = {"role": "user", "content": f"Контекст:\n{context_text}\n\nПользователь спросил: {query}\nОтвет по-русски:"}

    new_ai_text = call_local_llm([user_role_msg], model="llama3.1")

    ai_msg = AIMessage(content=new_ai_text)
    return {"messages": [ai_msg]}


from langgraph.graph import StateGraph, END

graph = StateGraph(BotState)

graph.add_node("user_input", user_input_node)
graph.add_node("bot_logic", bot_logic_node)

graph.add_edge("user_input", "bot_logic")
graph.add_edge("bot_logic", END)

graph.set_entry_point("user_input")

compiled_graph = graph.compile()


if __name__ == "__main__":
    # инициализация Chroma и эмбеддинги
    DB_PATH = "D:/_neoflex_bot/data/chroma_db"
    collection_obj = init_chroma_collection(DB_PATH, "rubq_paragraphs")
    #collection_obj = init_chroma_collection(DB_PATH, "rubq_qa")
    #collection_obj = init_chroma_collection(DB_PATH, "rubq_qa_model2")
    emb_model = init_embedding_model()

    print(" RAG-бот ")
    print("Введите 'exit', чтобы выйти.\n")

    # память (история) будем хранить сами: список AnyMessage
    memory: list[AnyMessage] = []

    while True:
        user_text = input("Пользователь: ")
        if user_text.strip().lower() in ["exit", "quit", "выход"]:
            print("Чат завершён.")
            break

        memory.append(HumanMessage(content=user_text))

        state = {"messages": memory}
        final_state = compiled_graph.invoke(state)

        memory = final_state["messages"]
        bot_msg = memory[-1].content
        print("Бот:", bot_msg)
