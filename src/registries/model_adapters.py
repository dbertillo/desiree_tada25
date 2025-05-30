
from abc import ABC, abstractmethod
from bedrock_claude_wrapper import BedrockClaudeWrapper, BedrockMixtralWrapper, BedrockDeepseekR1Wrapper
from registries.chatgpt4_client import AzureOpenAIChatGPT4Wrapper
from langchain_aws import BedrockLLM 
from botocore.exceptions import EventStreamError
from termcolor import colored

class BaseModel(ABC):
    @abstractmethod
    def predict(self, prompt: str, temperature: float, max_gen_lex: int):
        """
        Generate a response from the model given a prompt, temperature, and maximum generation length.
        
        Returns:
            A tuple of (response, metadata)
        """
        pass

# adapter for BedrockLLM (used for LLAMA3)
class Llama3Client(BaseModel):
    def __init__(self, model_id: str, **kwargs):
        self.llm = BedrockLLM(credentials_profile_name="default", model_id=model_id, **kwargs)
    
    def _call_stream(self, prompt, temperature, max_gen_len):

        stream = self.llm._prepare_input_and_invoke_stream(prompt=prompt,
                                                                temperature=temperature,
                                                                max_gen_len=max_gen_len)
        response = ''
        current_response_chunk = ''
        metadata = None
        try:
            for chunk in stream:
                current_response_chunk += chunk.text
                response += chunk.text
                if 'stop_reason' in chunk.generation_info.keys():
                    if chunk.generation_info['stop_reason'] == 'stop':
                        metadata = chunk.generation_info
                elif 'usage_metadata' in chunk.generation_info.keys():
                    metadata['amazon-bedrock-invocationMetrics']['total_tokens'] = chunk.generation_info['usage_metadata']['total_tokens']
                elif 'stop_reason' not in chunk.generation_info.keys():
                    print(colored(chunk.generation_info, 'red'))
            
            return response, metadata
        except EventStreamError as e:
            return 'FAILED EXTRACTION!', e


    def predict(self, prompt: str, temperature = 0.0, max_gen_lex=3000):
        # Calls the method that handles streaming for LLAMA3
        response, metadata = self._call_stream(prompt, temperature, max_gen_lex)
        return response, metadata

# adapter for BedrockClaudeWrapper (used for CLAUDE3)
class Claude3Client(BaseModel):
    def __init__(self, model_id, **kwargs):
        self.llm = BedrockClaudeWrapper(**kwargs)
    
    def predict(self, prompt: str, temperature=0.0, max_gen_lex=3000):
        # Uses a different invocation (here, invoke) for CLAUDE3
        response, metadata = self.llm.invoke(prompt, temperature, max_gen_lex)
        return response, metadata


class Mixtral8x7Client(BaseModel):
    def __init__(self, model_id, **kwargs):
        self.llm = BedrockMixtralWrapper(model_id, **kwargs)
        
    
    def predict(self, prompt: str, temperature=0.0, max_gen_lex=3000):
        # Uses a different invocation (here, invoke) for CLAUDE3
        response, metadata = self.llm.invoke(prompt, temperature, max_gen_lex)
        return response, metadata


class DeepseekR1Client(BaseModel):
    def __init__(self, model_id, **kwargs):
        self.llm = BedrockDeepseekR1Wrapper(model_id, **kwargs)

    def predict(self, prompt: str, temperature=0.0, max_gen_lex=3000):
        # Uses a different invocation (here, invoke) for CLAUDE3
        response, metadata = self.llm.invoke(prompt, temperature, max_gen_lex)
        return response, metadata
    

class ChatGPT4Client(BaseModel):
    def __init__(self, model_id, **kwargs):
        self.llm = AzureOpenAIChatGPT4Wrapper(model_id, **kwargs)
        self.model_id = model_id

    def predict(self, prompt, temperature=0.1, max_gen_lex=4096):
        # Uses a different invocation (here, invoke) for CLAUDE3
        response, metadata = self.llm.invoke(prompt, temperature, max_gen_lex)
        return response, metadata