from importlib import import_module


def build_model(configs):
    if str(configs.model).lower() != 'pastmixer':
        raise ValueError("This release contains only the PAST-Mixer model.")
    return import_module('models.PASTMixer').Model(configs)
