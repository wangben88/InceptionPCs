import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, recursive = False, **kwargs):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if recursive:
        for k, v in config["params"].items():
            try:
                if "target" in v:
                    obj = instantiate_from_config(v)
                    kwargs[k] = obj
                    config["params"].pop(k, None)
            except TypeError:
                pass
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)