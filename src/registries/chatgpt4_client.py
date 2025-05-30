import os
from openai import AzureOpenAI
from termcolor import colored
import json
class AzureOpenAIChatGPT4Wrapper:

    def __init__(self, model_id, **kwargs):
        self.endpoint = "https://dicita-merialdo-openai.openai.azure.com/"
        self.model_name = "gpt-4o"
        self.deployment = "gpt-4o"
        subscription_key = os.getenv("CHATGPT4_API_KEY")

        if not subscription_key:
            raise ValueError("Please set the CHATGPT4_API_KEY environment variable.")
        
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.endpoint,
            api_key=subscription_key,
        )


    def invoke(self, prompt, temperature=0.1, max_tokens=4096):

        #response = self.client.chat.completions.create(
        #    messages=[
        #        {
        #            "role": "system",
        #            "content": "You are a helpful assistant.",
        #        },
        #        {
        #            "role": "user",
        #            "content": "I am going to Paris, what should I see?",
        #        }
        #    ],
        #    max_tokens=max_tokens,
        #    temperature=temperature,
        #    top_p=1.0,
        #    model=self.deployment
        #)

        response = self.client.chat.completions.create(
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            model=self.deployment
        )

        print(response.choices[0].message.content)
        # response it type: openai.chat.completions.ChatCompletionResponse
        return response.choices[0].message.content, response