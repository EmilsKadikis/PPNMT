from itertools import product
import json

def _product_dict(**kwargs):
    """
    Given a dictionary such as {a: [1,2,3], b: [4,5,6]}
    Iterates through all possbiel combinations: [{a: 1, b: 4}, {a: 1, b: 5}, {a: 1, b: 6}, ....]

    Code taken from: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

def _merge_configs(base_config, extra_config):
    merged_config = base_config.copy()

    for key, value in extra_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            # Recursively merge nested dictionaries
            merged_config[key] = _merge_configs(base_config[key], value)
        else:
            # Override or add the key-value pair to the merged configuration
            merged_config[key] = value

    return merged_config

def _extend_experiment_definition_from_base(experiment_definition):
    base_file_name = experiment_definition.pop("base", None)
    if base_file_name is None:
        return experiment_definition
    
    with open(base_file_name, "r") as f:
        base_config = json.load(f)

    return _merge_configs(base_config, experiment_definition)

def expand_experiments(experiments):
    """
    If the experiment has the key "base" defined, it is used as the base for all the configurations defined in "experiments".

    For every experiment that has 'try_combinations': [column_a, column_b, ...] defined, 
    expands it into multiple separate experiments, trying all possible combinations of the values given in column_a, column_b, ....
    """
    if type(experiments) is not list:
        experiments = [experiments]
    experiments = [_extend_experiment_definition_from_base(experiment) for experiment in experiments]

    expanded_experiments = []
    for experiment_definition in experiments:
        if 'try_combinations' in experiment_definition:
            column_names = experiment_definition['try_combinations']
            all_possible_values = {column_name: experiment_definition['hyperparameters'][column_name] for column_name in column_names}
            all_possible_value_combinations = _product_dict(**all_possible_values)

            # create a new experiment definition for each combination
            for combination in all_possible_value_combinations:
                new_experiment_definition = experiment_definition.copy()
                new_experiment_definition['hyperparameters'] = experiment_definition['hyperparameters'].copy()

                new_experiment_definition.pop('try_combinations')
                # change the value of each hyperparameter that are being varied
                for key, value in combination.items():
                    new_experiment_definition['hyperparameters'][key] = value
                expanded_experiments.append(new_experiment_definition)
        else:
            expanded_experiments.append(experiment_definition)
    return expanded_experiments

