import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME

def map_model_names(model_names):
    model_name_map = {
        'probability': MODEL_NAME,
        'nt': MODEL_NAME,
        'NT': MODEL_NAME,
        'E_norm': 'IntaRNA 2',
        'INTARNA': 'IntaRNA 2',
        'intarna': 'IntaRNA 2',
        'IntaRNA': 'IntaRNA 2',
        'priblast': 'pRIblast',
        'rnaplex': 'RNAplex',
        'rnacofold': 'RNAcofold',
        'risearch2': 'RIsearch2',
        'assa': 'ASSA',
        'RNAhybrid': 'RNAhybrid',
        'rnahybrid': 'RNAhybrid',
        'rnaup': 'RNAup',
    }

    def map_single_name(name):
        if name.startswith('ens'):
            return 'Ensemble'
        return model_name_map.get(name, name)

    if isinstance(model_names, list):
        return [map_single_name(name) for name in model_names]
    else:
        return map_single_name(model_names)

def map_experiment_names(experiment_names):
    experiment_name_map = {
        'psoralen':'Psoralen-based',
        'ricseq':'RIC-seq',
        'mario':'MARIO',
    }

    def map_single_name(name):
        return experiment_name_map.get(name, name)

    if isinstance(experiment_names, list):
        return [map_single_name(name) for name in experiment_names]
    else:
        return map_single_name(experiment_names)
    
    
    
# Examples of usage:
# print(map_model_names('probability'))  # Output: 'RNARNet'
# print(map_model_names(['probability', 'rnaplex']))  # Output: ['RNARNet', 'RNAplex']
# print(map_model_names('ensSample'))  # Output: 'Ensemble'
