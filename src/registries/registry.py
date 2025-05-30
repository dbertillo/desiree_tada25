# registry.py

PROMPT_FORMATTERS = {}

def register_formatter(model_types):
    if isinstance(model_types, str):
        model_types = [model_types]

    def decorator(fn):
        for model_type in model_types:
            PROMPT_FORMATTERS[model_type] = fn
        return fn
    return decorator


CLIENT_REGISTRY = {}

def register_client(model_types):
    if isinstance(model_types, str):
        model_types = [model_types]
        
    def decorator(fn):
        for model_type in model_types:
            CLIENT_REGISTRY[model_type] = fn
        return fn
    return decorator