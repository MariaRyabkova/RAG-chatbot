from openai import OpenAI

def test_local_llm(prompt: str):

    # Укажем адрес API
    # Подключение к локальной модели
    client = OpenAI(base_url="http://localhost:11434/v1", api_key='1')

    response = client.chat.completions.create(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.7
    )

    print("LLM Ответ:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    user_prompt = "Привет! Расскажи о себе, что ты за модель?"
    test_local_llm(user_prompt)