import torch
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scipy.stats import spearmanr
import pandas as pd
from scipy.stats import rankdata

def compute_kl_divergence(q_mu, q_logvar, p_mu, p_logvar):
    kl_div = 0.5 * torch.sum(
        p_logvar - q_logvar - 1 + (q_logvar.exp() + (q_mu - p_mu).pow(2)) / p_logvar.exp()
    )
    return kl_div

class ExperimentManager:
    def __init__(self, base_path="./experiments"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
    def get_config_hash(self, config):
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def find_or_create_experiment_dir(self, config):
        config_hash = self.get_config_hash(config)
        

        for exp_dir in os.listdir(self.base_path):
            exp_path = os.path.join(self.base_path, exp_dir)
            config_path = os.path.join(exp_path, "config.json")
            
           
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    saved_hash = self.get_config_hash(saved_config)
                    
                    if saved_hash == config_hash:
                        return exp_path, False  
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_path = os.path.join(self.base_path, timestamp)
        os.makedirs(exp_path, exist_ok=True)
        
       
        config_path = os.path.join(exp_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        return exp_path, True 

def write_csv(results, csv_dir, config):
    results_dict = {
        int(tau): float(value)
        for tau, value in re.findall(r'tau=(\d+).*?: ([\d.]+)', results)
    }
    seed = config.exp.seed
    csv_path = csv_dir + '/mse.csv'
    os.makedirs(csv_dir, exist_ok=True)

    data = {'seed': seed}
    data.update({f'tau={k}': v for k, v in results_dict.items()})
    df_row = pd.DataFrame([data])

    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)

def check_csv(csv_dir, config):
    """
    Check if the result for the given seed already exists in the CSV file.
    如果文件不存在：直接返回 True\（表示“需要执行/可以写入”，因为还没有任何结果)
    如果文件存在：读取 CSV 文件，检查是否有对应 seed 的结果
        如果找到了对应 seed 的结果：返回 False（表示“不需要执行/不应该写入”，因为已经有结果了)
        如果没找到对应 seed 的结果：返回 True（表示“需要执行/可以写入”，因为这个 seed 还没有结果)
    这样做的目的是避免重复计算已经存在结果的实验配置。
    通过检查 CSV 文件中的 seed 列，可以判断是否已经有对应 seed 的结果
    :param csv_dir:
    :param config:
    :return:
    """
    csv_path = csv_dir + '/mse.csv'
    if not os.path.exists(csv_path):
        return True
    df = pd.read_csv(csv_path)
    print(list(df['seed']))
    # exit()
    seed = config.exp.seed
    try:
        seeds = list(df['seed'])
        # if seed in seeds:
        #     print(f'found result for seed {seed}')
        #     exit()
        return seed not in seeds
    except:
        return True

def generate_perturbed_sequences(true_actions, k, tau, a_dim, device, treatment_mode='continue', perturb_ratio=0.5, flip_ratio=0.2):
    # Calculate number of challenging sequences, ensure at least 1
    if treatment_mode == 'multilabel':
        perturb_ratio = 0.5
    else:
        perturb_ratio = 0.2
        flip_ratio=0.5
        print(f"perturb_ratio:{perturb_ratio}")
    n_challenging = max(1, int((k-1) * perturb_ratio))
    n_random = k - 1 - n_challenging

    if treatment_mode == 'multilabel':
        # Generate random sequences
        random_sequences = torch.bernoulli(torch.ones(n_random, 1, tau, a_dim, device=device) * 0.5)
        
        # Generate perturbed sequences by randomly flipping bits
        challenging_sequences = []
        for _ in range(n_challenging):
            perturbed = true_actions.clone()
            # Use the provided flip_ratio parameter
            mask = torch.rand(true_actions.shape, device=device) < flip_ratio
            perturbed[mask] = 1 - perturbed[mask]  # Flip selected bits
            challenging_sequences.append(perturbed.unsqueeze(0))
        challenging_sequences = torch.cat(challenging_sequences, dim=0)

    else:  # Continuous case
        # Generate random sequences
        random_sequences = torch.rand(n_random, 1, tau, a_dim, device=device)
        # mask_ratio = 0.5
        # mask = torch.rand_like(random_sequences) < mask_ratio
        # random_sequences[mask] = 1e-5
        
        # Generate perturbed sequences by significantly changing selected positions
        challenging_sequences = []
        for _ in range(n_challenging):
            perturbed = true_actions.clone()
            # Use same flip_ratio to select positions to change
            mask = torch.rand(true_actions.shape, device=device) < flip_ratio
            
            # Generate significant shifts for selected positions
            # For values close to 0: shift up by 0.3-0.7
            # For values close to 1: shift down by 0.3-0.7
            # For middle values: randomly shift up or down by 0.3-0.7
            rand_shifts = torch.rand(true_actions.shape, device=device) * 0.4 + 0.3  # Random values between 0.3 and 0.7
            middle_mask = (perturbed >= 0.3) & (perturbed <= 0.7) & mask
            high_mask = (perturbed > 0.7) & mask
            low_mask = (perturbed < 0.3) & mask
            
            # Middle values: randomly shift up or down
            direction = torch.rand(true_actions.shape, device=device) < 0.5
            perturbed[middle_mask & direction] += rand_shifts[middle_mask & direction]
            perturbed[middle_mask & ~direction] -= rand_shifts[middle_mask & ~direction]
            
            # High values: shift down
            perturbed[high_mask] -= rand_shifts[high_mask]
            
            # Low values: shift up
            perturbed[low_mask] += rand_shifts[low_mask]
            
            # Ensure values stay in [0,1] range
            perturbed = torch.clamp(perturbed, 0, 1)
            challenging_sequences.append(perturbed.unsqueeze(0))
        challenging_sequences = torch.cat(challenging_sequences, dim=0)

    # Combine all sequences
    all_sequences = torch.cat([random_sequences, challenging_sequences, true_actions.unsqueeze(0)], dim=0)
    # all_sequences = remove_duplicates(all_sequences)
    
    return all_sequences

def enhanced_analyze_case(case_info, save_dir='./case_study_plots', plot=True):
    save_dir = save_dir + '/' + str(case_info['individual_id'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime('%Y%m%d')
    case_id = case_info['individual_id']
    
    model_losses = case_info['model_losses']
    true_losses = case_info['true_losses']
    pred_losses = case_info['pred_losses']
    
    if plot:
        plt.figure(figsize=(15, 15))
        
        plt.subplot(221)
        sns.scatterplot(x=model_losses, y=true_losses, label='Model-True')
        sns.scatterplot(x=pred_losses, y=true_losses, label='Pred-True', alpha=0.5)
        plt.title('Loss Relationships')
        
        # Box Plot
        plt.subplot(222)
        loss_data = pd.DataFrame({
            'Model Loss': model_losses,
            'True Loss': true_losses,
            'Pred Loss': pred_losses
        })
        sns.boxplot(data=loss_data)
        plt.title('Loss Distributions')
        
        plt.subplot(223)
        model_ranks = rankdata(model_losses)
        true_ranks = rankdata(true_losses)
        pred_ranks = rankdata(pred_losses)
        plt.scatter(model_ranks, true_ranks, alpha=0.5, label='Model-True Ranks')
        plt.scatter(model_ranks, pred_ranks, alpha=0.5, label='Model-Pred Ranks')
        plt.title('Rank Comparisons')
        plt.legend()
        
        plt.subplot(224)
        corr_matrix = np.corrcoef([model_losses, true_losses, pred_losses])
        sns.heatmap(corr_matrix, 
                   annot=True,
                   xticklabels=['Model', 'True', 'Pred'],
                   yticklabels=['Model', 'True', 'Pred'])
        plt.title('Correlation Matrix')
        
        plt.tight_layout()
        
        main_plot_path = os.path.join(save_dir, f'case_{case_id}_main_analysis_{timestamp}.png')
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(model_losses, alpha=0.5, label='Model Loss', bins=30)
        plt.hist(true_losses, alpha=0.5, label='True Loss', bins=30)
        plt.hist(pred_losses, alpha=0.5, label='Pred Loss', bins=30)
        plt.title('Loss Distributions Histogram')
        plt.legend()
        

        hist_plot_path = os.path.join(save_dir, f'case_{case_id}_loss_dist_{timestamp}.png')
        plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        residuals = model_losses - true_losses
        plt.scatter(model_losses, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Model Loss')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        
        
        residual_plot_path = os.path.join(save_dir, f'case_{case_id}_residuals_{timestamp}.png')
        plt.savefig(residual_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        
        plt.figure(figsize=(12, 6))
        seq_indices = np.arange(len(model_losses))
        plt.plot(seq_indices, model_losses, label='Model Loss', alpha=0.7)
        plt.plot(seq_indices, true_losses, label='True Loss', alpha=0.7)
        plt.plot(seq_indices, pred_losses, label='Pred Loss', alpha=0.7)
        plt.title('Loss Values per Sequence')
        plt.xlabel('Sequence Index')
        plt.ylabel('Loss Value')
        plt.legend()
        
        
        sequence_plot_path = os.path.join(save_dir, f'case_{case_id}_sequence_{timestamp}.png')
        plt.savefig(sequence_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
   
    stats = {
        'basic_stats': {
            'model': {'mean': np.mean(model_losses), 'std': np.std(model_losses)},
            'true': {'mean': np.mean(true_losses), 'std': np.std(true_losses)},
            'pred': {'mean': np.mean(pred_losses), 'std': np.std(pred_losses)}
        },
        'correlations': {
            'model_true': spearmanr(model_losses, true_losses)[0],
            'model_pred': spearmanr(model_losses, pred_losses)[0],
            'pred_true': spearmanr(pred_losses, true_losses)[0]
        },
        'rank_metrics': {
            'true_sequence_rank': case_info['true_sequence_rank'],
            'rank_percentile': case_info['true_sequence_rank'] / len(model_losses)
        }
    }
    
    stats_path = os.path.join(save_dir, f'case_{case_id}_stats_{timestamp}.txt')
    with open(stats_path, 'w') as f:
        f.write('Case Study Statistics\n')
        f.write('===================\n\n')
        f.write(f'Case ID: {case_id}\n')
        f.write(f'Timestamp: {timestamp}\n\n')
        
        f.write('Basic Statistics:\n')
        for loss_type, metrics in stats['basic_stats'].items():
            f.write(f'{loss_type.capitalize()}:\n')
            f.write(f'  Mean: {metrics["mean"]:.4f}\n')
            f.write(f'  Std:  {metrics["std"]:.4f}\n')
        
        f.write('\nCorrelations:\n')
        for pair, corr in stats['correlations'].items():
            f.write(f'{pair}: {corr:.4f}\n')
        
        f.write('\nRank Metrics:\n')
        f.write(f'True sequence rank: {stats["rank_metrics"]["true_sequence_rank"]}\n')
        f.write(f'Rank percentile: {stats["rank_metrics"]["rank_percentile"]:.4f}\n')
    
    return stats, {
        'main_plot': main_plot_path,
        'histogram': hist_plot_path,
        'residual_plot': residual_plot_path,
        'sequence_plot': sequence_plot_path,
        'stats_file': stats_path
    }
