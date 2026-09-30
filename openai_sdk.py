import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="Groq API Key"
)

# 1. HESABINDAKİ TÜM AKTİF MODELLERİ SORGULA
try:
    models_list = client.models.list()
    available_models = [m.id for m in models_list.data]
    
    # Sadece Sohbet/Tool Calling destekleyen LLM'leri filtrele
    valid_llms = [
        m for m in available_models 
        if any(keyword in m for keyword in ["qwen", "gpt-oss", "llama", "mixtral"])
        and not any(ignore in m for ignore in ["guard", "whisper", "orpheus"])
    ]
    
    if not valid_llms:
        raise Exception("Tool calling destekleyen uygun bir LLM bulunamadı!")

    # Uygun ilk modeli seç (Örn: qwen/qwen3.8-27b veya openai/gpt-oss-120b)
    WORKING_MODEL = valid_llms[0]
    print(f"[SEÇİLEN MODEL]: {WORKING_MODEL}\n")

except Exception as e:
    print(f"Model seçimi hatası: {e}")
    exit()

# 2. SEÇİLEN MODEL İLE İSTEK AT
tools = [
    {
        "type": "function",
        "function": {
            "name": "topla",
            "description": "İki tam sayıyı toplar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

response = client.chat.completions.create(
    model=WORKING_MODEL,
    messages=[
        {"role": "system", "content": "Sen bir matematik ajanısın."},
        {"role": "user", "content": "150 + 250 kaç eder?"}
    ],
    tools=tools,
    tool_choice="auto"
)

msg = response.choices[0].message

# 1. Model metin yanıtı verdiyse:
if msg.content:
    print(f"Model Cevabı: {msg.content}")

# 2. Model bir araç (tool) çalıştırmak istediyse:
if msg.tool_calls:
    for tool in msg.tool_calls:
        print(f"--> Çağrılacak Fonksiyon : {tool.function.name}")
        print(f"--> Gönderilen Parametreler: {tool.function.arguments}")
        print(f"--> Çağrı ID              : {tool.id}")