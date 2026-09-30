
import json
from openai import OpenAI


GROQ_API_KEY = "Groq API Key"  
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

client = OpenAI(
    base_url=GROQ_BASE_URL,
    api_key=GROQ_API_KEY
)


MODEL_NAME = "openai/gpt-oss-120b"



def topla(a: int, b: int) -> int:
    """İki tam sayıyı toplayan fonksiyon."""
    print(f"\n   [>>> PYTHON TOOL'U ÇALIŞTI <<<]: {a} + {b} hesaplanıyor...")
    return a + b

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "topla",
            "description": "İki tam sayıyı toplar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "Birinci sayı"},
                    "b": {"type": "integer", "description": "İkinci sayı"}
                },
                "required": ["a", "b"]
            }
        }
    }
]



class OpenAIAgent:
    def __init__(self, name: str, instructions: str, model: str, tools: list, tools_schema: list):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools_map = {func.__name__: func for func in tools}
        self.tools_schema = tools_schema


class AgentRunner:
    @staticmethod
    def run(agent: OpenAIAgent, user_input: str) -> str:
        print(f"--- [{agent.name}] Başlatıldı (Aktif Model: {agent.model}) ---")
        print(f"Kullanıcı Görevi: {user_input}\n")

        messages = [
            {"role": "system", "content": agent.instructions},
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model=agent.model,
            messages=messages,
            tools=agent.tools_schema,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            print("   [Ajan Kararı]: Model otonom olarak 'topla' aracını çağırmaya karar verdi.")
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name in agent.tools_map:
                    python_function = agent.tools_map[func_name]
                    tool_result = python_function(**func_args)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": str(tool_result)
                    })

            final_response = client.chat.completions.create(
                model=agent.model,
                messages=messages
            )
            return final_response.choices[0].message.content
        else:
            return response_message.content



if __name__ == "__main__":
    hesap_ajani = OpenAIAgent(
        name="Hesaplayıcı Kartal",
        instructions="Sen sana verilen matematiksel görevleri elindeki araçları kullanarak çözen bir ajansın.",
        model=MODEL_NAME,
        tools=[topla],
        tools_schema=tools_schema
    )

    nihai_cevap = AgentRunner.run(
        agent=hesap_ajani,
        user_input="Depoda 150 ürün vardı, 250 ürün daha eklendi. Toplam kaç ürün oldu?"
    )

    print("\n--- Ajanın Nihai Cevabı ---")
    print(nihai_cevap)