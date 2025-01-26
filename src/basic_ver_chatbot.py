import os
import sys
from pathlib import Path

from langgraph.graph import StateGraph, END
import operator
from typing import TypedDict, Annotated
import subprocess

class MyState(TypedDict):
    step: str

def check_db_node(state: MyState):
    """Проверяем, есть ли Chroma DB"""
    db_path = "D:/_neoflex_bot/data/chroma_db"
    p = Path(db_path)
    if not p.exists() or not any(p.iterdir()):
        print("Chroma DB не найдена или пуста. Запустите ingestion-скрипт!")
    else:
        print("Chroma DB присутствует, всё ок.")
    return {"step": "run_chat"}

def run_chat_node(state: MyState):
    """Запускаем rag_chatbot.py через subprocess."""
    chatbot_script = os.path.join("D:/_neoflex_bot/src", "rag_chatbot.py")
    subprocess.run([sys.executable, chatbot_script], check=False)
    return {}

graph = StateGraph(MyState)
graph.add_node("check_db", check_db_node)
graph.add_node("run_chat", run_chat_node)

graph.add_edge("check_db", "run_chat")
graph.add_edge("run_chat", END)
graph.set_entry_point("check_db")

compiled_graph = graph.compile()

def main():
    st = {"step": ""}
    compiled_graph.invoke(st)

if __name__ == "__main__":
    main()
