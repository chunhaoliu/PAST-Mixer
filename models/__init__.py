from importlib import import_module


MODEL_REGISTRY = {
    'pastmixer': 'PASTMixer',
    'tsmixer': 'TSMixer',
    'lstm': 'LSTM',
    'transformer': 'Transformer',
    'timemixer': 'TimeMixer',
    'itransformer': 'iTransformer',
}


def resolve_model_name(name):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        available = ', '.join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[key]


def build_model(configs):
    module_name = resolve_model_name(configs.model)
    module = import_module(f'models.{module_name}')
    from models.io_adapter import IOAdapter, needs_adapter

    if needs_adapter(configs):
        return IOAdapter(configs, module.Model, module_name)
    return module.Model(configs)
