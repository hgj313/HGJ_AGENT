from backend.llm_model.reasoning_model.minimax import minimax_reasoning_model
from langchain.messages import SystemMessage, HumanMessage

minimax = minimax_reasoning_model(provider="openai")
model = minimax.get_model()
sys_prompt = SystemMessage(content="用户使用中文时：\n"
        "1. 最终回答必须使用中文；\n"
        "2. 若输出 thinking/reasoning，也必须使用中文；\n"
        "3. 除专有名词外，不要混用英文；\n"
        "4. 保持术语准确、表达自然。")
human_input = HumanMessage(content="给我写一篇一千字的抒情散文")
messages = [sys_prompt,human_input]
chunks = model.stream(messages, config={"stream_mode": "messages"})
minimax.stream_print(chunks)