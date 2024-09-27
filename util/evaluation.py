import pandas as pd
import numpy as np
import os
import seaborn as sns
import pickle
from tqdm.notebook import tqdm
from scipy import stats
import sys
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc, precision_score, recall_score, confusion_matrix
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mutual_info_score
from .plot_utils import calc_prec_rec_sens_npv
from .model_names_map import map_model_names
from .misc import balance_df, undersample_df, is_unbalanced, obtain_majority_minority_class

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def calculate_correlations(series_list, method='pearson', plot=False):
    """
    Calculate the correlations between all combinations of series in the list.
    
    Parameters:
    - series_list: List of pandas Series.
    - method: Correlation method ('pearson', 'spearman', 'kendall', 'mutual_info').
    - plot: Boolean indicating whether to plot scatter plots of the correlations.
    
    Returns:
    - DataFrame of correlation coefficients.
    """
    num_series = len(series_list)
    series_names = [s.name for s in series_list]
    correlations = pd.DataFrame(np.ones((num_series, num_series)), index=series_names, columns=series_names)

    for i in range(num_series):
        for j in range(i + 1, num_series):
            if method == 'pearson':
                corr = series_list[i].corr(series_list[j])
            elif method == 'spearman':
                corr, _ = spearmanr(series_list[i], series_list[j])
            elif method == 'kendall':
                corr, _ = kendalltau(series_list[i], series_list[j])
            elif method == 'mutual_info':
                corr = mutual_info_score(series_list[i], series_list[j])
            else:
                raise ValueError(f"Unsupported method: {method}")

            correlations.iloc[i, j] = corr
            correlations.iloc[j, i] = corr

            if plot:
                plt.figure(figsize=(6, 4))
                plt.scatter(series_list[i], series_list[j], alpha=0.75)
                plt.title(f'{method.capitalize()} Correlation between {series_names[i]} and {series_names[j]}: {corr:.2f}')
                plt.xlabel(series_names[i])
                plt.ylabel(series_names[j])
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    return correlations


def calculate_metrics_mean_over_iterations(res, n_iterations=10, task = 'all', column = 'probability', threshold = 0.5):
    """
    Perform iterative resampling and calculate the mean metrics over multiple iterations.
    
    Args:
        res (pd.DataFrame): The input dataframe containing 'probability' and 'interacting' columns.
        n_iterations (int): Number of iterations to perform (default is 10).
        task (str): 'patches', 'interactors', 'all'
        column (str): column of the model prediction
        threshold (float): if > threshold the prediction is 1, if < threshold the prediction is 0
    
    Returns:
        dict: A dictionary containing the mean metrics (auc, acc, precision, recall, specificity, npv).
    """
    auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    npv_scores = []

    for i in range(n_iterations):
        # Resample and calculate metrics
        res_balanced = balancing_only_for_one_task(res, task = task)
        
        # Calculate precision, recall, specificity, npv
        precision, recall, specificity, npv = calc_prec_rec_sens_npv(res_balanced, column)
        
        # Calculate accuracy
        acc = ((res_balanced[column] > threshold_model) == res_balanced.interacting).sum() / res_balanced.shape[0]
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(res_balanced.interacting, res_balanced[column])
        auc_score = auc(fpr, tpr)
        
        # Store results for this iteration
        auc_scores.append(auc_score)
        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        specificity_scores.append(specificity)
        npv_scores.append(npv)

    # Calculate the mean of each metric
    metrics_mean = {
        'auc': np.mean(auc_scores),
        'acc': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'specificity': np.mean(specificity_scores),
        'npv': np.mean(npv_scores)
    }

    return metrics_mean


def filter_hq_data_by_interaction_length(res, hq_threshold):

    condition_1 = res.policy.isin(['easypos', 'smartneg'])
    condition_2 = abs(res.seed_x1 - res.seed_x2) >= hq_threshold
    condition_3 = abs(res.seed_y1 - res.seed_y2) >= hq_threshold
    condition_4 = res.policy.isin(['easyneg', 'hardneg'])
    condition_final = (condition_1 & condition_2 & condition_3) | condition_4

    return res[condition_final].reset_index(drop = True)

def balancing_only_for_one_task(res: pd.DataFrame, task: 'str') -> pd.DataFrame:
    """
    Undersample negative classes accorting to the task.
    
    Args:
        res (pd.DataFrame): DataFrame containing the 'policy' column which includes different classes.
        task (str): it can be 'patches' or 'interactors'
        
    Returns:
        pd.DataFrame: A new balanced DataFrame for the specific task
    """
    assert task in ['patches', 'interactors', 'all']
    
    # Split the dataframe into different classes
    ep = res[res.policy == 'easypos']
    hn = res[res.policy == 'hardneg']
    en = res[res.policy == 'easyneg']
    sn = res[res.policy == 'smartneg']
    
    if task == 'patches':
        
        size_sampling_neg = min(ep.shape[0]//2, hn.shape[0], en.shape[0])
        
        ep_sampled = ep.sample(size_sampling_neg*2)
        hn_sampled = hn.sample(size_sampling_neg)
        en_sampled = en.sample(size_sampling_neg)
    
        res_sampled = pd.concat([ep_sampled, hn_sampled, en_sampled], axis=0).reset_index(drop=True)
        
        
    elif task == 'interactors':
        
        size_sampling_neg = min(ep.shape[0], sn.shape[0])
        
        ep_sampled = ep.sample(size_sampling_neg)
        sn_sampled = sn.sample(size_sampling_neg)

        res_sampled = pd.concat([ep_sampled, sn_sampled], axis=0).reset_index(drop=True)
        
    elif task == 'all':
        size_sampling_neg = min(ep.shape[0]//3, hn.shape[0], en.shape[0], sn.shape[0])
    
        # Undersample each class to have the same number of samples as the smallest class
        ep_sampled = ep.sample(size_sampling_neg*3)
        hn_sampled = hn.sample(size_sampling_neg)
        en_sampled = en.sample(size_sampling_neg)
        sn_sampled = sn.sample(size_sampling_neg)

        res_sampled = pd.concat([ep_sampled, hn_sampled, en_sampled, sn_sampled], axis=0).reset_index(drop=True)
    
    assert np.round(
        (res_sampled[res_sampled.ground_truth == 0].shape[0] / res_sampled[res_sampled.ground_truth == 1].shape[0]), 
        1) == 1
    
    return res_sampled



class ModelResultsManager:
    def __init__(self, model_name: str, dimension:int, chkpt_directory: str, rna_rna_files_dir:str, test_info_directory:str, other_tools:list, other_tools_dir:str):
        """
        Initializes the ModelResultsManager.
        
        Parameters:
        model_name (str): The name of the model.
        dimension (int): The length of the rna in the test set. For instance, if 200 it means that the contact matrix is 200X200.
        chkpt_directory (str): The directory where the models are stored
        rna_rna_files_dir (str): The directory where the train/test/val list of couples are stored.
        test_info_directory (str): The directory where the test500.csv (or test200.csv) file is stored
        other_tools (list): List with the name of all the thermodynamic tools
        other_tools_dir (str): The directory where the results of all the thermodynamic tools results are stored
        """
        
        self.dimension = str(dimension)
        self.model_name = model_name
        self.chkpt_directory = chkpt_directory
        self.rna_rna_files_dir = rna_rna_files_dir
        self.test_info_directory = test_info_directory
        self.other_tools = other_tools
        self.other_tools_dir = other_tools_dir
        
        # Load all couples used for training/testing the model
        self._load_training_testing_couples()
        
        self.dataframes = {}
        # Load all dataframes related to the model
        self._load_dataframes()
        
        
    def _load_dataframes(self):
        """
        Internal method to load the dataframes for all experiments.
        The method assumes that the files are stored in CSV format in the provided directory.
        """
        experiments = ['test_HQ', 'test', 'val_HQ', 'val', 'splash', 'ricseq', 'mario']
        for experiment in experiments:
            try:
                self.dataframes[experiment] = self._load_res_for_experiment(experiment)
            except FileNotFoundError:
                continue
                
                
    def _load_training_testing_couples(self):
        """
        Loads RNA pairs used for training and validation.
        """
        self.couples_paris_training = self._load_pickle(os.path.join(self.rna_rna_files_dir, "gene_pairs_training.txt"))
        self.couples_paris_val = self._load_pickle(os.path.join(self.rna_rna_files_dir, "gene_pairs_val.txt"))
        self.couples_splash_training = self._load_pickle(os.path.join(self.rna_rna_files_dir, "splash", "gene_pairs_training.txt"))
        self.couples_splash_test = self._load_pickle(os.path.join(self.rna_rna_files_dir, "splash", "gene_pairs_test.txt"))
    
    def _load_single_experiment_data(
        self,
        experiment: str, 
        paris_test: bool, 
        paris_finetuned_model: bool, 
        specie_paris: str,
        paris_hq: bool, 
        paris_hq_threshold: int, 
        n_reads_paris: int, 
        splash_trained_model: bool,
        only_test_splash_ricseq_mario: bool, 
        n_reads_ricseq: int
    ) -> pd.DataFrame:
        """
        See get_experiment_data()
        """
        
        assert experiment in ['paris', 'splash', 'ricseq', 'mario']
        
        if experiment == 'paris':
            
            experiment_name = self._get_paris_experiment_name(paris_test, paris_hq)
            
            res = self.dataframes[experiment_name]
            
            assert specie_paris in ['all', 'mouse', 'human']
            if specie_paris in ['mouse', 'human']:
                res = res[res.specie == specie_paris].reset_index(drop = True)
                
            res = filter_hq_data_by_interaction_length(res, paris_hq_threshold)
            
            res = res[
                    (res.policy.isin(['hardneg', 'easyneg', 'smartneg'])) |
                    ((res.n_reads >= n_reads_paris) & (res.policy == 'easypos'))
                ].reset_index(drop = True)
        
        elif experiment in ['splash', 'ricseq', 'mario']:
            
            res = self.dataframes[experiment]
            
            if only_test_splash_ricseq_mario:
                if experiment == 'splash':
                    res = res[res.couples.isin(self.couples_splash_test)].reset_index(drop = True)
                else:
                    raise NotImplementedError
                    
            if experiment == 'ricseq':
                res = res[
                    (res.policy.isin(['hardneg', 'easyneg', 'smartneg'])) |
                    ((res.n_reads >= n_reads_ricseq) & (res.policy == 'easypos'))
                ].reset_index(drop = True)
        
        if paris_finetuned_model:
            res = res[~res.couples.isin(self.couples_paris_val)].reset_index(drop = True)
            
        if splash_trained_model:
            res = res[~res.couples.isin(self.couples_splash_training)].reset_index(drop = True)
        
        res = self._format_dataframe(res)
        self._validate_data_integrity(res)
        
        return res
    
    def get_experiment_data(
        self, 
        experiment: str, 
        paris_test: bool, 
        paris_finetuned_model: bool, 
        specie_paris: str,
        paris_hq: bool, 
        paris_hq_threshold: int, 
        n_reads_paris: int,
        splash_trained_model: bool,
        only_test_splash_ricseq_mario: bool,
        n_reads_ricseq: int,
        logistic_regression_models:dict,
    ) -> pd.DataFrame:
        """
        Returns the dataframe for the specified experiment.
        
        Parameters:
        experiment (str): The name of the experiment ('paris', 'splash', 'ricseq', 'mario', 'psoralen').
        paris_test (bool): If True, I will use paris test set, otherwise the validation set.
        paris_finetuned_model (bool): If True, I will drop all the pairs that are inside the validation set
        specie_paris (str): 'all', 'human' or 'mouse'.
        paris_hq (bool): If True, I load the paris HQ results (with region of interaction > 20)
        paris_hq_threshold (int): minimum length of the region of interaction supporting positive or smartneg data
        n_reads_paris (int): minimum number of reads (included) supporting paris data
        splash_trained_model (bool): If true, I will exclude from all the test set results the couple inside the splash training set.
        only_test_splash_ricseq_mario (bool): If true, I will keep only the data of the test set (this is valid only for 'splash', 'ricseq', 'mario' since paris is a test set).
        n_reads_ricseq (int): minimum number of reads (included) supporting ricseq data
        
        Returns:
        pd.DataFrame: The dataframe for the requested experiment.
        """
        
        assert experiment in ['paris', 'splash', 'ricseq', 'mario', 'psoralen']
        
        if experiment == 'psoralen':
            paris = self._load_single_experiment_data('paris', paris_test, paris_finetuned_model, specie_paris, paris_hq, paris_hq_threshold, n_reads_paris, splash_trained_model,only_test_splash_ricseq_mario, n_reads_ricseq)
            splash = self._load_single_experiment_data('splash', paris_test, paris_finetuned_model, specie_paris,paris_hq, paris_hq_threshold, n_reads_paris, splash_trained_model,only_test_splash_ricseq_mario, n_reads_ricseq)
            splash['n_reads'] = np.nan
            assert set(paris.columns) == set(splash.columns)
            splash = splash.filter(list(paris.columns), axis = 1) # so that they have the same order
            res = pd.concat([paris, splash], axis = 0).reset_index(drop=True)
        else:
            res = self._load_single_experiment_data(experiment, paris_test, paris_finetuned_model, specie_paris,paris_hq, paris_hq_threshold, n_reads_paris, splash_trained_model,only_test_splash_ricseq_mario, n_reads_ricseq)
            
        if logistic_regression_models:
            res = map_thermodynamic_columns(res, self.other_tools, logistic_regression_models)
        
        return res
    
    def _load_res_for_experiment(self, experiment_name:str) -> pd.DataFrame:
        
        res = pd.read_csv(os.path.join(self.chkpt_directory, self.model_name, f'{experiment_name}_results{self.dimension}.csv'))
        
        for tool_name in self.other_tools:
            tool = pd.read_csv(os.path.join(self.other_tools_dir, f'{tool_name}_{experiment_name}{self.dimension}.csv'))
            tool['value'] = tool['value'].astype(float)
            assert set(tool.couples) == set(res.id_sample)
            assert (tool.minimum == True).all()
            assert len(set(tool.couples)) == len(set(res.id_sample)) == res.shape[0] == tool.shape[0]
            res = res.merge(tool[['value', 'couples']].rename({'couples': 'id_sample', 'value': tool_name}, axis=1), on='id_sample', how='left').fillna(0)
        
        # Drop all the pairs (they should be few) that are present in the training set.
        res = res[~res.couples.isin(self.couples_paris_training)].reset_index(drop = True)
        
        test500 = pd.read_csv(os.path.join(self.test_info_directory, f'{experiment_name}{self.dimension}.csv'))
        test500['distance_from_site'] = ( (test500['distance_x'] ** 2) + (test500['distance_y']** 2) )**(0.5) #pitagora
        test500['distance_from_site_embedding'] = ( (test500['distance_embedding_x'] ** 2) + (test500['distance_embedding_y']** 2) )**(0.5) #pitagora
                
        if experiment_name == 'ricseq':
            df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_ricseq.csv'))
            test500 = df_nt[['couples', 'n_reads']].rename({'couples':'df_nt_id'}, axis = 1).merge(test500, on = 'df_nt_id')
        
        res = res.merge(test500.drop(['policy', 'g1', 'g2'], axis = 1).rename({'couples':'id_sample'}, axis = 1), on = 'id_sample').reset_index(drop = True) # policy, g1, g2 are already in res
        
        return res

    def _load_pickle(self, file_path: str):
        """
        Helper function to load pickle files.

        Args:
            file_path (str): Path to the pickle file.
        Returns:
            The loaded object from the pickle file.
        """
        with open(file_path, "rb") as fp:
            return pickle.load(fp)
        
    def _validate_data_integrity(self, res: pd.DataFrame) -> pd.DataFrame:
        """
        Validates data integrity

        Args:
            res (pd.DataFrame): Dataframe to validate.
        """
        
        assert res[res.ground_truth.isna()].shape[0] == 0
        assert (res['ground_truth'] == res['interacting'].astype(int)).all()
        
        
    def _format_dataframe(self, res: pd.DataFrame):
        """
        Prepare properly the dataframe

        Args:
            res (pd.DataFrame): Dataframe to validate.
        """
            
            
        res['interacting'] = True
        res.loc[res.policy.isin(['easyneg', 'hardneg', 'smartneg']), 'interacting'] = False
        res = res.drop('Unnamed: 0', axis = 1).reset_index(drop = True)
        return res
        
        
        
    def _get_paris_experiment_name(self, paris_test: bool, paris_hq: bool) -> str:
        """
        Determines the Paris experiment name based on test and HQ flags.

        Args:
            paris_test (bool): Whether it's a test set.
            paris_hq (bool): Whether it's a HQ dataset.

        Returns:
            str: The experiment name ('test_HQ', 'test', 'val_HQ', 'val').
        """
        if paris_test and paris_hq:
            return 'test_HQ'
        elif paris_test:
            return 'test'
        elif paris_hq:
            return 'val_HQ'
        return 'val'
    
    
def obtain_all_model_auc(subset, tools, n_runs=50):
    if is_unbalanced(subset):
        list_to_append = ['NT']
        
        aucs_dict = {tool_name: [] for tool_name in list_to_append + tools}

        for _ in range(n_runs):
            # Perform undersampling to create a balanced subset
            majority_class, minority_class = obtain_majority_minority_class(subset)

            # Undersample majority class
            majority_undersampled = resample(majority_class, 
                                             replace=False, 
                                             n_samples=len(minority_class), 
                                             random_state=np.random.randint(10000))

            # Combine minority class with undersampled majority class
            balanced_subset = pd.concat([minority_class, majority_undersampled])

            # Calculate AUC for 'NT'
            fpr, tpr, _ = roc_curve(balanced_subset.ground_truth, balanced_subset['probability'])
            roc_auc = auc(fpr, tpr)
            aucs_dict['NT'].append(roc_auc)

            # Calculate AUC for each tool
            for tool_name in tools:
                fpr, tpr, _ = roc_curve(balanced_subset.ground_truth, abs(balanced_subset[tool_name]))
                roc_auc = auc(fpr, tpr)
                aucs_dict[tool_name].append(roc_auc)

        # Calculate mean AUC for each model
        mean_aucs = {tool_name: np.mean(aucs_dict[tool_name]) for tool_name in aucs_dict}
    else:
        # Calculate AUC directly without undersampling
        mean_aucs = {}
        
        fpr, tpr, _ = roc_curve(subset.ground_truth, subset['probability'])
        mean_aucs['NT'] = auc(fpr, tpr)
        
        for tool_name in tools:
            fpr, tpr, _ = roc_curve(subset.ground_truth, abs(subset[tool_name]))
            mean_aucs[tool_name] = auc(fpr, tpr)
    
    # Create DataFrame with results
    df_out = pd.DataFrame({
        'model_name': list(mean_aucs.keys()),
        'auc': [round(auc, 2) for auc in mean_aucs.values()]
    })
    
    return df_out


def remove_outliers(df, column, threshold = 3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()

    # Filter data based on Z-score
    filtered_data = df[column][abs(z_scores) < threshold]
    
    df = df[abs(z_scores) < threshold]
    
    return df

def map_thermodynamic_columns(res, energy_columns, logistic_regression_models):
    for model_column in energy_columns:
        lrm_dict = logistic_regression_models[model_column]
        X_test = np.expand_dims(res[model_column], 1)
        y_pred = log_func(lrm_dict['intercept'], lrm_dict['coef'], X_test)
        res[model_column] = pd.Series(y_pred.flatten()) #modify column according to the model mapping
    return res

def map_dataset_to_hp(dataset):
    assert dataset in ['parisHQ', 'paris_mouse_HQ', 'ricseqHQ', 'psoralen', 'paris', 'paris_mouse', 'ricseq', 'mario', 'splash']
    
    if dataset == 'parisHQ':
        experiment = 'paris'
        specie_paris = 'human'
        paris_hq_threshold = 35
        n_reads_ricseq = np.nan

    elif dataset == 'paris_mouse_HQ':
        experiment = 'paris'
        specie_paris = 'mouse'
        paris_hq_threshold = 35
        n_reads_ricseq = np.nan

    elif dataset == 'ricseqHQ':
        experiment = 'ricseq'
        specie_paris = np.nan
        paris_hq_threshold = np.nan
        n_reads_ricseq = 5

    elif dataset == 'psoralen':
        experiment = 'psoralen'
        specie_paris = 'all'
        paris_hq_threshold = 1
        n_reads_ricseq = np.nan

    elif dataset == 'paris_mouse':
        experiment = 'paris'
        specie_paris = 'mouse'
        paris_hq_threshold = 1
        n_reads_ricseq = np.nan

    else:
        experiment = dataset
        specie_paris = 'human'
        paris_hq_threshold = 1
        n_reads_ricseq = 1
        
    return experiment, specie_paris, paris_hq_threshold, n_reads_ricseq

def obtain_df_auc(model, paris_finetuned_model, energy_columns, splash_trained_model, list_of_datasets = ['parisHQ', 'paris_mouse_HQ', 'ricseqHQ', 'psoralen', 'paris', 'paris_mouse', 'ricseq', 'mario', 'splash'], logistic_regression_models = {}):
    assert set(list_of_datasets).intersection(set(['parisHQ', 'paris_mouse_HQ', 'ricseqHQ', 'psoralen', 'paris', 'paris_mouse', 'ricseq', 'mario', 'splash'])) == set(list_of_datasets)
    

    dfs = [] 
    for dataset in tqdm(list_of_datasets):

        experiment, specie_paris, paris_hq_threshold, n_reads_ricseq = map_dataset_to_hp(dataset)
        
        res = model.get_experiment_data(
            experiment = experiment, 
            paris_test = True, 
            paris_finetuned_model = paris_finetuned_model, 
            specie_paris = specie_paris,
            paris_hq = False,
            paris_hq_threshold = paris_hq_threshold,
            n_reads_paris = 1,
            splash_trained_model = splash_trained_model,
            only_test_splash_ricseq_mario = False,
            n_reads_ricseq = n_reads_ricseq,
            logistic_regression_models = logistic_regression_models
        )

        easypos_smartneg = res[res.policy.isin(['easypos', 'smartneg'])].reset_index(drop = True)
        enhn = res[res.policy.isin(['easypos', 'easyneg', 'hardneg'])].reset_index(drop = True)

        dfs.append(obtain_all_model_auc(easypos_smartneg, energy_columns).rename({'auc': f'auc_interactors_{dataset}'}, axis = 1))
        dfs.append(obtain_all_model_auc(enhn, energy_columns).rename({'auc': f'auc_patches_{dataset}'}, axis = 1))


    df_auc = pd.concat(dfs, axis = 1)
    df_auc = df_auc.loc[:,~df_auc.columns.duplicated()].copy()
    df_auc['model_name'] = df_auc['model_name'].apply(map_model_names)
    df_auc = df_auc.drop_duplicates().reset_index(drop=True)
    
    return df_auc

def log_func(i, c, x):
    z = i + np.dot(x, c.T)  # Compute the linear combination
    return 1 / (1 + np.exp(-z))  # Apply the logistic function

def replace_outliers_with_nan_and_make_positive(df, columns):
    for column in columns:
        df[column] = df[column].abs()
        # Calculate the z-scores
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        # Set a threshold for z-score
        threshold = 4
        # Replace outliers with NaN
        df.loc[np.abs(z_scores) > threshold, column] = np.nan
    return df

def obtain_sr_nosr(res, both_sr_condition, filtered_policies):

    if both_sr_condition:
        sr = res[res['simple_repeat1'] & res['simple_repeat2']]
    else:
        sr = res[res['simple_repeat1'] | res['simple_repeat2']]

    no_sr = res[(res['simple_repeat1'] == False) & (res['simple_repeat2'] == False)] #res[res['none1'] & res['none2']] #res[(res['simple_repeat1'] == False) & (res['simple_repeat2'] == False)]

    sr = sr[sr.policy.isin(filtered_policies)].reset_index(drop = True)
    no_sr = no_sr[no_sr.policy.isin(filtered_policies)].reset_index(drop = True)

    return sr, no_sr