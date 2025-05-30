import boto3
from botocore.exceptions import ClientError
import json
import os
from termcolor import colored

class BedrockClaudeWrapper:
    """
        This class uses the Messages API to interact with the Bedrock API.
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
    """

    def __init__(self):
        self.client = boto3.client(
            service_name='bedrock-runtime'
        )
        self.model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.model_name = "claude-3-5-sonnet"
        
            
    def print_setup(self):
        print(colored("------------###------------", 'cyan'))
        print(f"Using the Bedrock API to call claude 3.5 sonnet for extraction.")
        print(f"\tModel ID: {self.model_id}")
        print(f"\tModel Name: {self.model_name}")
        # print(f"Client: {self.client}")
        print(colored("------------###------------", 'cyan'))
        print("\n")
        return

    def invoke(self, message: str, temperature: int, max_tokens: int):
        # construct the body from the message
        # list of parameters for this specific model and API: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

        # Check if messagge arrives as list or string
        
        if isinstance(message, list):
            system = message[0]
            if type(system) != str:
                system = str(system)
            messages = message[1]
        else:
            raise ValueError("Message must be a list with two elements: system and messages")
        
        body = json.dumps({
            "max_tokens": max_tokens,
            "system": system, #prompt[0]
            "messages": messages, #prompt[1]
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": temperature,
        })

        # invoke the model and generate the response
        response = self.client.invoke_model(
            body=body,
            #contentType='string',
            #accept='string',
            modelId=self.model_id,
            #trace='ENABLED'|'DISABLED',
            #guardrailIdentifier='string',
            #guardrailVersion='string',
            #performanceConfigLatency='standard'|'optimized'
        )
        metadata = response
        response_body = json.loads(response.get('body').read())
        # print(colored(response_body.get('content')[0].get('text'), 'green'))
        # get the HTTP response body
        # list of parameters for this specific model and API: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        return response_body.get('content')[0].get('text'), metadata


class BedrockMixtralWrapper:
    def __init__(self, model_id):
        self.client = boto3.client(
            service_name='bedrock-runtime'
        )

        self.model_id = model_id

        # self.bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name='us-west-2')

    def invoke(self, prompt, temperature: int, max_tokens: int):

        body = json.dumps({
            "prompt":prompt,
            "max_tokens": 250,
            "temperature": 0.5 #Temperature controls randomness; higher values increase diversity, lower values boost predictability.
        })

        response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json", 
                contentType="application/json"
            )

        # print(colored(response, 'cyan'))
        response_output = json.loads(response.get('body').read())
        mixtral_parse_text = response_output['outputs'][0]['text']
        mixtral_parse_text = mixtral_parse_text.replace('\n', ' ')
        mixtral_output = mixtral_parse_text.strip()

        # Print the output with new lines wherever "\n" is encountered
        # print(colored(mixtral_output, 'green'))
        # exit()
        metadata = response
        return mixtral_output, metadata
'''
# test the wrapper
if __name__ == "__main__":

    # create the wrapper
    bedrock_client = BedrockClaudeWrapper()

    
    # loop until the user enters "exit" to exit the program and invoke the model with the user input
    while True:
        message = input("Enter a message: ")
        if message == "exit":
            break
        output = bedrock_client.invoke(message)
        print(output)
        print("\n")
'''




class BedrockDeepseekR1Wrapper():
    def __init__(self, model_id):
        self.client = boto3.client("bedrock-runtime")
        self.model_id = model_id



    def invoke(self, prompt, temperature: int, max_tokens: int): 
# Embed the prompt in DeepSeek-R1's instruction format.

        body = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        })

        try:
            # Invoke the model with the request.
            response = self.client.invoke_model(modelId=self.model_id, body=body)

            # Read the response body.
            model_response = json.loads(response["body"].read())
            
            # Extract choices.
            choices = model_response["choices"]
            
            # Print choices.
            for index, choice in enumerate(choices):
                print(f"Choice {index + 1}\n----------")
                print(f"Text:\n{choice['text']}\n")
                print(f"Stop reason: {choice['stop_reason']}\n")
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            exit(1)

        exit()

if __name__ == "__main__":
    # create the wrapper
    client = boto3.client('bedrock')


    response = client.list_foundation_models(
    # byProvider='string',
    # byCustomizationType='FINE_TUNING'|'CONTINUED_PRE_TRAINING'|'DISTILLATION',
    # byOutputModality='TEXT'|'IMAGE'|'EMBEDDING',
    # byInferenceType='ON_DEMAND'|'PROVISIONED'
    )

    response = client.list_inference_profiles()
    print(response)
    # print(json.dumps(response, indent=4))




