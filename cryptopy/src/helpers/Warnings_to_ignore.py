import warnings

warnings.filterwarnings(
    "ignore", message="JSON format is not registered with bravado-core!"
)
warnings.filterwarnings(
    "ignore", message="guid format is not registered with bravado-core!"
)
warnings.filterwarnings("ignore", category=UserWarning, module="swagger_spec_validator")
