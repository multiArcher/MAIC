def __getattr__(name):
    if name == "REGISTRY":
        from utils.maker import CriticMaker
        return CriticMaker

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
