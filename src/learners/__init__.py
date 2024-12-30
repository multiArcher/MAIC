def __getattr__(name):
    if name == "REGISTRY":
        from utils.maker import LearnerMaker
        return LearnerMaker

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
