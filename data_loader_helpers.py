import importlib

def load_data_from_data_loader(data_loader_definition): 
    # if data_loader_definition is a string, import the module and call load_data
    if isinstance(data_loader_definition, str):
        data_loader = importlib.import_module(data_loader_definition)
        return data_loader.load_data()
    # if data_loader_definition is a dictionary, call load_data with the arguments in the dictionary
    elif isinstance(data_loader_definition, dict):
        data_loader = importlib.import_module(data_loader_definition['name'])
        return data_loader.load_data(**data_loader_definition['args'])
