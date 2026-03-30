import pickle
from pathlib import Path
from tools import plot_combined_distributions, plot_rank_distributions_combined, plot_mse_trends, generate_latex_table, generate_single_table
import matplotlib.pyplot as plt
import pandas as pd

def load_nested_case_infos(base_dir):
    """
    Load case_infos data from nested directory structure
    Returns: {coeff_n: {seed: {test: {model_name: data}}}}
    where test is True or False
    """
    base_path = Path(base_dir)
    all_data = {}
    
    # Iterate through coeff directories (coeff_1, coeff_2, etc.)
    for coeff_dir in base_path.glob('coeff_*'):
        coeff_name = coeff_dir.name
        all_data[coeff_name] = {}
        
        # 遍历model目录
        for model_dir in coeff_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            # 递归查找所有包含case_infos的目录
            for case_infos_dir in model_dir.rglob('case_infos'):
                if not case_infos_dir.is_dir():
                    continue
                
                # Iterate through seed directories
                for seed_dir in case_infos_dir.iterdir():
                    if not seed_dir.is_dir():
                        continue
                        
                    seed = seed_dir.name
                    
                    # Initialize seed dictionary if not exists
                    if seed not in all_data[coeff_name]:
                        all_data[coeff_name][seed] = {'True': {}, 'False': {}}
                    
                    # Check both True and False subdirectories
                    for test_value in [True, False]:
                        pkl_file = seed_dir / str(test_value) / f'case_infos_{model_name}.pkl'
                        
                        if pkl_file.exists():
                            with open(pkl_file, 'rb') as f:
                                data = pickle.load(f)
                                # Assume the pickle file has a single key
                                key = list(data.keys())[0]
                                all_data[coeff_name][seed][str(test_value)][model_name] = data[key]
    
    return all_data


data = load_nested_case_infos('../../my_outputs/cancer_sim_cont/22')
print(data.keys())

merged_data = data['coeff_4']['10']['False']
fig = plot_combined_distributions(merged_data, vi=False)
plt.savefig('False/coeff4.pdf')

merged_data = data['coeff_4']['10']['True']
fig = plot_combined_distributions(merged_data, vi=False)
plt.savefig('True/coeff4.pdf')


def load_nested_mse(base_dir):
    """
    Load mse.csv data from nested directory structure
    Returns: {coeff_n: {test: {model_name: dataframe}}}
    where test is True or False
    """
    base_path = Path(base_dir)
    all_data = {}
    
    # Iterate through coeff directories (coeff_1, coeff_2, etc.)
    for coeff_dir in base_path.glob('coeff_*'):
        coeff_name = coeff_dir.name
        all_data[coeff_name] = {'True': {}, 'False': {}}
        
        for model_dir in coeff_dir.iterdir():
            # print(model_dir)
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            
            print(model_name)
            for train_dir in model_dir.rglob('train/True'):
                if not train_dir.is_dir():
                    continue
                
                parent_dir = train_dir.parent
                print(parent_dir)
                for test_value in [True, False]:
                    mse_file = train_dir / str(test_value) / 'mse.csv'
                    
                    if mse_file.exists():
                        df = pd.read_csv(mse_file, index_col=False)
                        df = df.drop('seed', axis=1)
                        all_data[coeff_name][str(test_value)][model_name] = df
    
    return all_data

data = load_nested_mse('../../my_outputs/cancer_sim_cont/22')
print(data['coeff_2'])
plot_mse_trends(data, False, [1,2,3])

coeff_list = [1,2,3,4]
tau_list = [1, 2, 3, 4, 5, 6]
models = ["ACTIN", "CT", "CRN", "RMSN", "VCIP"]
generate_latex_table(data, coeff_list, tau_list, models, "all_mse_table.txt")
generate_latex_table(data, coeff_list, tau_list, models, "all_mse_table_true.txt", True)

tau_list = [1, 2, 4, 6, 8, 9, 10, 11, 12]
coeff = 4
generate_single_table(data, coeff, tau_list, models, "mse_table.txt", True)