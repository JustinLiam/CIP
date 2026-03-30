import pickle
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib import font_manager

models = ['VCIP', 'ACTIN', 'CT', 'CRN', 'RMSN']
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 
    'axes.unicode_minus': False, 
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

def plot_combined_distributions(merged_data, tau_values=[2, 4, 6, 8], figsize=(20, 7), vi=False, filter=True):
    """
    Plot both rank and correlation distributions in a 2x4 layout
    
    Args:
        merged_data: Dictionary containing all models' data
        tau_values: List of tau values to plot
        figsize: Figure size for the entire plot
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, len(tau_values))
    
    model_names = list(merged_data.keys())
    if filter:
        model_names = []
        for key in models:
            if key in merged_data:
                model_names.append(key)
    print(model_names)
    colors = sns.color_palette("Set2", len(model_names))
    fontsize = 20
    fontsize2 = 16
    
    for ax_idx, tau in enumerate(tau_values):
        ax = fig.add_subplot(gs[0, ax_idx])
        
        # Prepare data for plotting
        plot_data = []
        for model in model_names:
            if tau in merged_data[model]:
                ranks = [case['true_sequence_rank'] for case in merged_data[model][tau]]
                n = 100
                normalized = (n - np.array(ranks)) / (n - 1)
                if 'VCIP' in model or 'RMSN' in model:
                    if tau in [2, 4]:
                        print(f'tau = {tau}, rank, {model}: {np.mean(normalized):.3f} $\\pm$ {np.std(normalized):.3f}')
                plot_data.append(normalized)
        
        # Violin plots
        parts = ax.violinplot(plot_data, positions=range(len(model_names)), 
                            showmeans=True, showextrema=True)
        
        # Customize violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.3)
        
        # Box plots
        bp = ax.boxplot(plot_data, positions=range(len(model_names)), 
                       widths=0.2, showfliers=False,
                       patch_artist=True)
        
        # Customize box plots
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i])
            box.set_alpha(0.5)
        
        # Configure subplot
        ax.tick_params(axis='both', labelsize=fontsize2)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([])
        # ax.set_xticklabels(
        #     model_names,
        #     rotation=45,
        #     ha='right',
        #     va='top',
        #     fontdict={'size': fontsize2}
        # )
        
        if ax_idx == 0:
            # ax.set_ylabel('Normalized Rank', fontsize=fontsize)
            ax.set_ylabel('GRP', fontsize=fontsize)
        ax.set_title(r'$τ$' + f' = {tau}', fontsize=fontsize, pad=10)
        
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_ylim(0, 1.02)
    
    for ax_idx, tau in enumerate(tau_values):
        ax = fig.add_subplot(gs[1, ax_idx])
        
        # Prepare data for plotting
        plot_data = []
        for model in model_names:
            if tau in merged_data[model]:
                corrs = [case['correlations']['model_true'] for case in merged_data[model][tau]]
                corrs = np.array(corrs)
                corrs = corrs[~np.isnan(corrs)]
                if 'VCIP' in model or 'RMSN' in model:
                    if tau in [2, 4]:
                        print(f'tau = {tau}, corrs, {model}: {np.mean(corrs):.3f} $\\pm$ {np.std(corrs):.3f}')
                plot_data.append(corrs)

        if vi:
            # Violin plots
            parts = ax.violinplot(plot_data, positions=range(len(model_names)), 
                                showmeans=True, showextrema=True)
            
            # Customize violin plots
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.3)
        
        # Box plots
        bp = ax.boxplot(plot_data, positions=range(len(model_names)), 
                       widths=0.2, showfliers=False,
                       patch_artist=True)
        
        # Customize box plots
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i])
            box.set_alpha(0.5)
        
        # Configure subplot
        ax.tick_params(axis='both', labelsize=fontsize2)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(
            model_names,
            rotation=45,
            ha='right',
            va='top',
            fontdict={'size': fontsize2}
        )
        
        if ax_idx == 0:
            # ax.set_ylabel("Spearman's ρ", fontsize=fontsize)
            ax.set_ylabel("RCS", fontsize=fontsize)
        # ax.set_title(f'τ = {tau}', fontsize=fontsize, pad=10)
        
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_ylim(-1.02, 1.02)
    
    plt.tight_layout()
    return fig

def plot_rank_distributions_combined(merged_data, tau_values=[2, 4, 6, 8], figsize=(20, 4), mode='rank', filter=True):
    """
    Plot combined box plots and density plots for rank distributions
    
    Args:
        merged_data: Dictionary containing all models' data
        tau_values: List of tau values to plot
        figsize: Figure size for the entire plot
    """
    fig, axes = plt.subplots(1, len(tau_values), figsize=figsize)
    if len(tau_values) == 1:
        axes = [axes]
    
    model_names = list(merged_data.keys())
    if filter:
        model_names = []
        for key in merged_data.keys():
            if key in models:
                model_names.append(key)
    colors = sns.color_palette("Set2", len(model_names))
    fontsize = 20
    fontsize2 = 16
    
    for ax_idx, tau in enumerate(tau_values):
        ax = axes[ax_idx]
        
        # Prepare data for plotting
        plot_data = []
        labels = []
        
        for model_idx, model in enumerate(model_names):
            if tau in merged_data[model]:
                if mode == 'rank':
                    ranks = [case['true_sequence_rank'] for case in merged_data[model][tau]]
                    n = 100
                    normalized = (n - np.array(ranks)) / (n - 1)
                    # print(f'rank, {model}: {np.mean(normalized)+-np.std(normalized)}')
                    plot_data.append(normalized)
                else:
                    corrs = [case['correlations']['model_true'] for case in merged_data[model][tau]]
                    corrs = np.array(corrs)
                    corrs = corrs[~np.isnan(corrs)]
                    # print(f'corrs, {model}: {np.mean(corrs)+-np.std(corrs)}')
                    plot_data.append(corrs)

                # labels.extend([model] * len(normalized))
        
        # Create violin plot with overlaid box plot
        if mode == 'rank':
            parts = ax.violinplot(plot_data, positions=range(len(model_names)), 
                                showmeans=True, showextrema=True)
            
            # Customize violin plots
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.3)
        
        # Add box plot
        bp = ax.boxplot(plot_data, positions=range(len(model_names)), 
                       widths=0.2, showfliers=False,
                       patch_artist=True)
        
        # Customize box plots
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i])
            box.set_alpha(0.5)
        
        # Configure subplot
        ax.tick_params(axis='both', labelsize=fontsize2)
        
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(
            model_names,
            rotation=45,
            ha='right',  
            va='top',   
            fontdict={'size': fontsize2}
        )
        
        if ax_idx == 0:
            # ax.set_ylabel('Normalized Rank', fontsize=fontsize)
            ax.set_ylabel('GRP', fontsize=fontsize)
        ax.set_title(r'$τ$' + f' = {tau}', fontsize=fontsize, pad=10)
        
        ax.grid(True, alpha=0.2, linestyle='--')
        if mode == 'rank':
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(-1.02, 1.02)
    
    # plt.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.9, wspace=0.25)
    plt.tight_layout()
    return fig

def plot_mse_trends(data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]], 
                   test: bool,
                   coeff_indices: List[int],
                   models: List[str] = ['VCIP', 'ACTIN', 'CT', 'CRN', 'RMSN'],
                   max_tau: int = 6, 
                   fig_width: int = 20, 
                   fig_height: int = 5):
    """
    Plot MSE trends for all coefficients in a horizontal layout
    """
    plt.style.use('default')
    
    # 设置字体属性
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    times_font = font_manager.FontProperties(fname=font_path)
    font = {'family': 'Times New Roman'}
    plt.rc('font', **font)
    
    colors = sns.color_palette("Set2", len(models))
    markers = ['o', 's', '^', '<', '*', 'D', 'v', '>', 'p', 'h']
    linestyles = ['-', '--', '-.', ':', 
                 (0, (3, 1, 1, 1)),
                 (0, (5, 1)),
                 (0, (3, 1, 1, 1, 1, 1)),
                 (0, (1, 1)),
                 (0, (3, 5, 1, 5)),
                 (0, (5, 5))]
    
    # 只处理选定的系数
    selected_data = {f'coeff_{i}': data[f'coeff_{i}'] 
                    for i in coeff_indices 
                    if f'coeff_{i}' in data}
    
    n_plots = len(selected_data)
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(fig_width, fig_height * n_rows),
                            squeeze=False)
    axes_flat = axes.flatten()
    fontsize = 20
    fontsize2 = 16
    
    test_str = str(test)
    
    for idx, (coeff_name, coeff_data) in enumerate(selected_data.items()):
        ax = axes_flat[idx]
        ax.set_facecolor('white')
        
        for model_idx, model_name in enumerate(models):
            if model_name in coeff_data[test_str]:
                df = coeff_data[test_str][model_name]
                # 计算每列的平均值（按原代码逻辑处理）
                mse_values = df.mean()
                
                ax.plot(range(1, max_tau+1), mse_values[:max_tau], 
                       label=model_name, 
                       color=colors[model_idx % len(colors)],
                       marker=markers[model_idx % len(markers)],
                       linestyle=linestyles[model_idx % len(linestyles)],
                       linewidth=1.5,
                       markersize=5,
                       markerfacecolor='white',
                       markeredgewidth=2)
        
        # 设置标签和标题的字体
        ax.set_xlabel('τ', fontproperties=times_font, fontsize=fontsize)
        if idx % n_cols == 0:
            ax.set_ylabel('Mean MSE', fontproperties=times_font, fontsize=fontsize)
        # 从coeff_name中提取数字
        coeff_num = coeff_name.split('_')[1]
        ax.set_title(f'Coeff {coeff_num}', fontproperties=times_font, fontsize=fontsize)
        
        # 设置刻度标签字体
        ax.tick_params(axis='both', which='major', labelsize=fontsize2)
        
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, max_tau+1))
        
        # 设置图例字体
        ax.legend(loc='best',
                 fontsize=fontsize2,
                 frameon=True,
                 framealpha=0.8)
    
    # 移除未使用的子图
    for idx in range(len(selected_data), len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    plt.tight_layout()
    # 保存为PDF时使用高DPI以确保清晰度
    plt.savefig(f'./mse_{"test" if test else "train"}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def generate_latex_table(data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]], 
                        coeff_list: List[int],
                        tau_list: List[int], 
                        models: List[str],
                        output_file: str,
                        test: bool = False):
    """
    Generate LaTeX table for model comparisons across different gamma and tau values
    
    Args:
        data: Dictionary structure: {coeff_name: {test/train: {model: DataFrame}}}
        coeff_list: List of coefficient indices
        tau_list: List of tau values
        models: List of model names
        output_file: Output file path
        test: Boolean indicating if this is test data
    """
    
    latex_content = [
        "\\begin{table}[t]",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Model comparison results with different interference levels ($\\gamma$) and prediction steps ($\\tau$). " +
        "Shown: RMSE as mean $\\pm$ standard deviation over five runs.}",
        "\\label{tab:model_comparison}",
        "\\vskip 0.15in",
        "\\centering",
        "\\begin{tabular}{@{}l|" + "r" * len(tau_list) + "@{}}",
        "\\toprule"
    ]
    
    # Header row with tau values
    header = "              & " + " & ".join([f"$\\tau = {tau}$" for tau in tau_list]) + " \\\\"
    latex_content.append(header)
    latex_content.append("\\midrule")
    
    test_str = str(test)
    
    # For each gamma (coefficient)
    for coeff_idx in coeff_list:
        coeff_key = f'coeff_{coeff_idx}'
        # Add gamma header
        latex_content.append(f"\\(\\gamma={coeff_idx}\\)\\ \\ \\ \\ \\ \\ ")
        
        # Store all means for current gamma to find minimum
        current_gamma_means = {tau: {} for tau in tau_list}
        
        # First pass: collect all means
        for model in models:
            try:
                df = data[coeff_key][test_str][model]
                for tau_idx, tau in enumerate(tau_list):
                    tau_values = df[f'tau={tau}'] if f'tau={tau}' in df.columns else df.iloc[:, tau_idx]
                    mean = tau_values.mean()
                    current_gamma_means[tau][model] = mean
            except (KeyError, IndexError):
                continue
        
        # Find minimum mean for each tau
        min_means = {tau: min(model_means.values()) if model_means else float('inf') 
                    for tau, model_means in current_gamma_means.items()}
        
        # Second pass: generate rows with bold formatting for minimum values
        for model in models:
            if model == models[0]:
                model_prefix = ""
            else:
                model_prefix = "\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ "
                
            row_values = []
            try:
                df = data[coeff_key][test_str][model]
                for tau_idx, tau in enumerate(tau_list):
                    tau_values = df[f'tau={tau}'] if f'tau={tau}' in df.columns else df.iloc[:, tau_idx]
                    mean = tau_values.mean()
                    std = tau_values.std()
                    
                    value_str = f"{mean:.2f}$\\pm${std:.2f}"
                    # Add bold if this is the minimum mean for current tau
                    if abs(mean - min_means[tau]) < 1e-10:  # Using small epsilon for float comparison
                        value_str = f"\\textbf{{{value_str}}}"
                    row_values.append(value_str)
                    
            except (KeyError, IndexError):
                row_values = ["-"] * len(tau_list)
            
            # Construct the row
            row = f"{model_prefix}{model} & " + " & ".join(row_values) + " \\\\"
            latex_content.append(row)
            
        # Add midrule between gamma groups
        if coeff_idx != coeff_list[-1]:
            latex_content.append("\\midrule")
    
    # Close table
    latex_content.extend([
        "\\midrule",
        "\\end{tabular}",
        "\\vskip -0.15in",
        "\\end{table}"
    ])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(latex_content))
        
def generate_single_table(data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]], 
                         coeff: int,
                         tau_list: List[int],
                         models: List[str],
                         output_file: str,
                         test: bool = False):
    """
    Generate LaTeX table for model comparisons across different tau values for a single gamma
    
    Args:
        data: Dictionary structure: {coeff_name: {test/train: {model: DataFrame}}}
        coeff: Coefficient index (gamma value)
        tau_list: List of tau values
        models: List of model names
        output_file: Output file path for the LaTeX table
        test: Boolean indicating if this is test data
    """
    
    # Calculate column format string
    column_format = "l" + "r" * len(tau_list)
    
    latex_content = [
        "\\begin{table*}[t]",
        "\\caption{Multi-step-ahead prediction results. Shown: RMSE as mean $\\pm$ standard deviation over five runs.}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\label{tab:prediction_results}",
        "\\vskip 0.1in",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule"
    ]
    
    # Header row with tau values
    header = "              & " + " & ".join([f"$\\tau={tau}$" for tau in tau_list]) + " \\\\"
    latex_content.append(header)
    latex_content.append("\\midrule")
    
    test_str = str(test)
    coeff_key = f'coeff_{coeff}'
    
    # For each model
    for i, model in enumerate(models):
        row_values = []
        try:
            df = data[coeff_key][test_str][model]
            # Calculate mean and std for each tau
            for tau in tau_list:
                tau_values = df[f'tau={tau}'] if f'tau={tau}' in df.columns else df.iloc[:, tau]
                mean = tau_values.mean()
                std = tau_values.std()
                
                value_str = f"{mean:.2f} $\\pm$ {std:.2f}"  # 添加了空格
                if model == models[-1]:  # 如果是最后一个模型
                    value_str = f"\\textbf{{{value_str}}}"
                row_values.append(value_str)
                
        except (KeyError, IndexError):
            row_values = ["-"] * len(tau_list)
            
        # Construct the row
        row = f"{model} & " + " & ".join(row_values) + " \\\\"
        latex_content.append(row)
        
        # Add midrule before the last model
        if i == len(models) - 2:  # 在最后一个模型之前加入midrule
            latex_content.append("\\midrule")
    
    # Close table
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vskip -0.05in",
        "\\end{table*}"
    ])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(latex_content))

        