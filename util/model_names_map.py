def map_model_names(model_names):
    model_name_map = {
        'probability': 'RNARNet',
        'nt': 'RNARNet',
        'NT': 'RNARNet',
        'E_norm': 'IntaRNA',
        'INTARNA': 'IntaRNA',
        'intarna': 'IntaRNA',
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

    
    
# Examples of usage:
# print(map_model_names('probability'))  # Output: 'RNARNet'
# print(map_model_names(['probability', 'rnaplex']))  # Output: ['RNARNet', 'RNAplex']
# print(map_model_names('ensSample'))  # Output: 'Ensemble'
