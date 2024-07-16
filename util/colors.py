#COLORS##
COLOR_NT_AUC = '#3C00FF'
COLOR_NT_PREC = '#747DC6'
COLOR_NT_NPV = '#281B7C'


COLOR_NT1 = '#2B2D63' #if you have more than 1 NT model to show
COLOR_NT2 = '#80B2FF'
COLOR_NT3 = '#3864AC'
COLOR_NT4 = '#2B2D63' #if you have more than 1 NT model to show
COLOR_NT5 = '#8A6642'
COLOR_NT6 = '#B8B799'
COLOR_NT7 = '#6A5F31' #if you have more than 1 NT model to show
COLOR_NT8 = '#412227' #if you have more than 1 NT model to show

COLOR_INTARNA_AUC = 'orange'
COLOR_INTARNA_PREC = '#FFB450'
COLOR_INTARNA_NPV = '#A57617'

COLOR_ENS_AUC = 'green'
COLOR_ENS_PREC = '#70DC6C'
COLOR_ENS_NPV = '#18791B'

# other models
COLOR_PRIBLAST = 'gray'
COLOR_RNAUP = 'red'
COLOR_RNAPLEX = 'pink'
COLOR_RNAHYBRID = 'purple'
COLOR_RNACOFOLD = 'brown'
COLOR_RISEARCH2 = 'olive'
COLOR_ASSA = 'cyan'


model_colors_dict = {
    
    'priblast': COLOR_PRIBLAST,
    'RNAup': COLOR_RNAUP,
    'RNAplex': COLOR_RNAPLEX,
    'RNAhybrid': COLOR_RNAHYBRID,
    'rnacofold': COLOR_RNACOFOLD,
    'risearch2': COLOR_RISEARCH2,
    'assa': COLOR_ASSA,
    
    'E_norm': COLOR_INTARNA_AUC,
    'INTARNA': COLOR_INTARNA_AUC,
    
    'NT': COLOR_NT_AUC,
    'nt': COLOR_NT_AUC,
    
    'nt1': COLOR_NT1,
    'nt2': COLOR_NT2,
    'nt3': COLOR_NT3,
    'nt4': COLOR_NT4,
    'nt5': COLOR_NT5,
    'nt6': COLOR_NT6,
    'nt7': COLOR_NT7,
    'nt8': COLOR_NT8,
    
    'ensemble': COLOR_ENS_AUC,
    'ensemble_score': COLOR_ENS_AUC,
}