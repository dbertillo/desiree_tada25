from .registry import register_client
from .model_adapters import Llama3Client, Claude3Client, Mixtral8x7Client, DeepseekR1Client, ChatGPT4Client

@register_client(['llama3'])  # e.g. "LLAMA3"
def create_llama3(model_id: str, **kwargs):
    return Llama3Client(model_id, **kwargs)

@register_client(['claude3'])  # e.g. "CLAUDE3"
def create_claude3(**kwargs):
    return Claude3Client(**kwargs)

@register_client(['mixtral8x7b'])  # e.g. "MIXTRAL8X7B"
def create_mixtral8x7b(model_id: str, **kwargs):
    return Mixtral8x7Client(model_id, **kwargs)

@register_client(['deepseek-r1'])  # e.g. "MIXTRAL8X7B"
def create_mixtral8x7b(model_id: str, **kwargs):
    return DeepseekR1Client(model_id, **kwargs)

@register_client(['chatgpt4'])
def create_chatgpt4(model_id: str, **kwargs):
    return ChatGPT4Client(model_id, **kwargs)