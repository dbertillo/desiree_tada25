# prompt_formatters.py

from .registry import register_formatter
from termcolor import colored
import json

@register_formatter(["claude3"])
def format_claude3(messages) -> list:
    system_message = {}
    other_messages = []
    for message in messages: 
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            system_message = {"role": role, "content": content}
        else:
            other_messages.append({"role": role, "content": content})
    return [system_message, other_messages]


@register_formatter(["llama3", "llama3-8b", "llama3-70b"])
def format_llama3(messages) -> str:
    def encapsulate(role, content):
        return f"<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>"

    prompt = "<|begin_of_text|>"
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        prompt += encapsulate(role, content)
    prompt += encapsulate("assistant", "")
    return prompt


@register_formatter(['mixtral8x7b'])
def format_mixtral(messages, bos_token="<s>", eos_token="</s>") -> str:

    prompt = bos_token
    loop_messages = messages
    system_message = None

    # Handle optional system message
    if loop_messages[0]['role'] == 'system':
        system_message = loop_messages[0]['content']
        loop_messages = loop_messages[1:]

    merged_messages = []

    if len(loop_messages) == 2:
        # Only one message, make it assistant
        merged_messages.append({'role': 'user', 'content': loop_messages[0]['content']})
        merged_messages.append({'role': 'assistant', 'content': loop_messages[1]['content']})

    elif len(loop_messages) > 2:
        if loop_messages[-1]['role'] == 'user':
            loop_messages[-1]['role'] = 'assistant'
        current_user_message = None
        current_assistant_message = None
        for i, message in enumerate(loop_messages):
            role = message['role']
            content = message['content']
            if role == 'user':
                if current_assistant_message is not None:
                    merged_messages.append({'role': 'assistant', 'content': current_assistant_message})
                    current_assistant_message = None
                if current_user_message is None:
                    current_user_message = content
                else:
                    current_user_message += "\n" + content
            elif role == 'assistant':
                if current_user_message is not None:
                    merged_messages.append({'role': 'user', 'content': current_user_message})
                    current_user_message = None
                if i == len(loop_messages) - 1:
                    merged_messages.append({'role': 'assistant', 'content': content})


    # Build prompt
    for idx, message in enumerate(merged_messages):
        role = message['role']
        content = message['content']

        if role == 'user':
            if idx == 0 and system_message is not None:
                prompt += f" [INST] {system_message}\n\n{content} [/INST]"
            else:
                prompt += f" [INST] {content} [/INST]"
        elif role == 'assistant':
            prompt += f" {content}{eos_token}"

    return prompt

@register_formatter(['deepseek-r1'])
def format_deepseekr1(messages) -> str:
    being_of_sentence = "<|begin_of_sentence|>"
    user = "<|User|>"
    assistant = "<|Assistant|>"
    think = "<think>\n"

    prompt = being_of_sentence
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            prompt += f"{user}{content}"
        elif role == "assistant":
            prompt += f"{assistant}{content}"
    prompt += f"{think}"
    return prompt


@register_formatter(['chatgpt4'])
def format_chatgpt4(messages) -> str:
    """
    Format the messages for ChatGPT-4.
    """
    return messages
    prompt = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    return prompt