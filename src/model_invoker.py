from registries.registry import CLIENT_REGISTRY
import registries.clients_registry


class ModelInvoker():

    def __init__(self, model_type, model_id):
        self.model_type = model_type
        self.model_id = model_id

        self.setup_model()
    
    def setup_model(self):
        try:
            create_fn = CLIENT_REGISTRY[self.model_type]
            # For models that require a model_id, pass it along; others can ignore it.
            self.client = create_fn(model_id=self.model_id)
        except KeyError:
            raise ValueError(f"No registered model for type: {self.model_type}")


    def invoke(self, formatted_prompt):
        response, metadata = self.client.predict(formatted_prompt)
        return response, metadata
    

if __name__ == "__main__":
    # Example usage
    model_type = "llama3"
    model_id = "llama3-7b"
    input_data = "Hello, how are you?"
    
    invoker = ModelInvoker(model_type, model_id)
    output = invoker.invoke(input_data)
    print(output)