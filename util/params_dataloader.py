# dataloader_hyperparameters


dl_hparams = {
    '200': {
        'paris': {
            'all_species': {
                'train_hq': {
                    'multipliers': {
                        1.5: 0.01, 
                        5:0.1,
                        7:0.1,
                        10:0.45,
                        15:0.05,
                        20:0.1,
                        27:0.1,
                        35:0.08,
                        1_000:0.01,
                    },
                    'windows': {
                        (30, 31): 0.005,
                        (31, 80): 0.045,
                        (80, 150): 0.08, 
                        (150, 180): 0.02, 
                        (180, 220): 0.13, 
                        (220, 250): 0.13,
                        (250, 350): 0.1,
                        (350, 400): 0.08, 
                        (400, 450): 0.07, 
                        (450, 550): 0.07, 
                        (550, 1_000): 0.16, 
                        (1_000, 5_970):0.08, 
                        (5_970, 5_971): 0.03
                    },
                },
                'full_train': {
                    'multipliers': {
                        1.5: 0.04, 
                        3: 0.05,
                        5:0.05,
                        7:0.05,
                        10:0.4,
                        15:0.05,
                        20:0.15,
                        27:0.1,
                        35:0.1,
                        1_000:0.01,
                    },
                    'windows': {
                        (30, 31): 0.025,
                        (31, 80): 0.045,
                        (80, 150): 0.1, 
                        (150, 180): 0.07, 
                        (180, 220): 0.15, 
                        (220, 250): 0.13,
                        (250, 350): 0.1,
                        (350, 400): 0.04, 
                        (400, 450): 0.03, 
                        (450, 550): 0.1, 
                        (550, 1_000): 0.13, 
                        (1_000, 5_970):0.04, 
                        (5_970, 5_971): 0.04,
                    },
                },
            },
        },
        'paris_finetuning': {
            'all_species': {
                'train_hq': {
                    'multipliers': {
                        1.5: 0.04, 
                        3: 0.05,
                        5:0.05,
                        7:0.05,
                        10:0.4,
                        15:0.05,
                        20:0.15,
                        27:0.1,
                        35:0.1,
                        1_000:0.01,
                    },
                    'windows': {
                        (30, 31): 0.005,
                        (31, 80): 0.045,
                        (80, 150): 0.08, 
                        (150, 180): 0.02, 
                        (180, 220): 0.13, 
                        (220, 250): 0.13,
                        (250, 350): 0.1,
                        (350, 400): 0.08, 
                        (400, 450): 0.07, 
                        (450, 550): 0.07, 
                        (550, 1_000): 0.16, 
                        (1_000, 5_970):0.08, 
                        (5_970, 5_971): 0.03
                    },
                },
                'full_train': {
                    'multipliers': {
                        1.5: 0.04, 
                        3: 0.05,
                        5:0.05,
                        7:0.05,
                        10:0.4,
                        15:0.05,
                        20:0.15,
                        27:0.1,
                        35:0.1,
                        1_000:0.01,
                    },
                    'windows': {
                        (30, 31): 0.025,
                        (31, 80): 0.045,
                        (80, 150): 0.1, 
                        (150, 180): 0.07, 
                        (180, 220): 0.15, 
                        (220, 250): 0.15,
                        (250, 350): 0.1,
                        (350, 400): 0.04, 
                        (400, 450): 0.03, 
                        (450, 550): 0.1, 
                        (550, 1_000): 0.13, 
                        (1_000, 5_970):0.04, 
                        (5_970, 5_971): 0.02,
                    },
                },
            },
        },
        'splash': {
            'all_species': {
                'multipliers': {
                        # TODO
                },
                'windows': {
                        # TODO
                },
            },
        },
    },
}


def parisdataset_name_map(dataset, finetuning):
    assert dataset == 'paris'
    if finetuning:
        return 'paris_finetuning'
    else:
        return 'paris'
    
def specie_name_map(specie):
    assert specie in ['all', 'human', 'mouse']
    if (specie == 'all'):
        return 'all_species'
    else:
        raise NotImplementedError
        
def train_hq_name_map(train_hq):
    if train_hq:
        return 'train_hq'
    else:
        return 'full_train'

def load_windows_and_multipliers(dimension, dataset, specie, train_hq, finetuning):
    
    dimension = str(dimension)
    
    assert dimension in dl_hparams.keys()
    
    specie = specie_name_map(specie)
    
    if dataset == 'paris':
        dataset = parisdataset_name_map(dataset, finetuning)
        train_hq = train_hq_name_map(train_hq)
        inner_dict = dl_hparams[dimension][dataset][specie][train_hq]
    else:
        inner_dict = dl_hparams[dimension][dataset][specie]
        
    return inner_dict['multipliers'], inner_dict['windows']


    # '500':{
    #     'easypretrain':{
    #         'pos_multipliers':{15:0.2, 25:0.3,50:0.2, 100:0.23, 10_000_000: 0.07},
    #         'neg_multipliers': {15:0.05, 28:0.15, 40:0.08, 50:0.05, 60:0.1, 80:0.03, 90:0.03, 100:0.05, 110:0.05, 120:0.1, 140:0.05, 160:0.03, 180:0.03, 200:0.03, 220:0.02, 240:0.01, 260:0.01, 10_000_000:0.1},
    #         'neg_windows':{(280, 800): 0.4, (800, 1_500): 0.15, (1_500, 2_000): 0.1, (2_000, 2_300): 0.1, (2_300, 5_970): 0.15, (5_970, 5_971): 0.1},
    #     },
    #     'nopretrain':{
    #         'per_sample_p':0.25,
    #         'proportion_sn':0.55,
    #         'proportion_hn':0.0,
    #         'proportion_en':0.1,
    #         'pos_multipliers':{15:0.2, 25:0.3,50:0.2, 100:0.23,100_000_000:0.07},
    #         'neg_multipliers':{15:0.2, 25:0.3,50:0.2, 100:0.23,100_000_000:0.07},
    #         'neg_windows':{(280, 800): 0.4, (800, 1_500): 0.15, (1_500, 2_000): 0.1, (2_000, 2_300): 0.1, (2_300, 5_970): 0.15, (5_969, 5_971): 0.1},
    #     },
    # }