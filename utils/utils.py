import yaml

def load_property_config(filepath="properties.yaml"):
    """
    Loads the property configuration from a YAML file.
    
    How to use:
    PROPERTY_CONFIG = load_property_config("properties.yaml")
    """
    with open(filepath, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error loading YAML: {exc}")
            return None
    
    print(config)

