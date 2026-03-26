import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def main():
    # Load the latest results file
    reports_dir = Path("artifacts/reports")
    result_files = list(reports_dir.glob("scale_up_100_results_*.csv"))
    if not result_files:
        print("No results found.")
        return
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading '{latest_file.name}' for Figure 2 Generation...")
    
    df = pd.read_csv(latest_file)
    
    # Quantitative Analysis
    total_subjects = len(df)
    mean_true = df['true_rho'].mean()
    mean_pred = df['pred_rho'].mean()
    mean_error = df['error'].mean()
    
    pass_80 = df['is_above_80'].sum()
    pass_80_rate = (pass_80 / total_subjects) * 100
    
    pass_90 = (df['pred_rho'] >= 0.90).sum()
    pass_90_rate = (pass_90 / total_subjects) * 100
    
    failed = total_subjects - pass_80
    failed_rate = (failed / total_subjects) * 100
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(df['true_rho'], df['pred_rho'])
    
    # Linear Regression for Panel A
    slope, intercept, r_value, p_val_lin, std_err = stats.linregress(df['true_rho'], df['pred_rho'])
    r_squared = r_value**2
    
    print("\n--- QUANTITATIVE ANALYSIS ---")
    print(f"Total Subjects: {total_subjects}")
    print(f"Mean True FC: {mean_true:.4f}")
    print(f"Mean Predicted FC: {mean_pred:.4f}")
    print(f"Mean Absolute Error: {mean_error:.4f}")
    print(f"Pass Rate (>= 80%): {pass_80_rate:.1f}% ({pass_80}/{total_subjects})")
    print(f"Excellent Pass Rate (>= 90%): {pass_90_rate:.1f}% ({pass_90}/{total_subjects})")
    print(f"Failed Rate (< 80%): {failed_rate:.1f}% ({failed}/{total_subjects})")
    print(f"Paired T-test (True vs Pred): t={t_stat:.4f}, p={p_val:.4e}")
    # Fix the way boolean is formatted to handle older numpy versions
    try:
        r_squared_val = r_squared[0] if isinstance(r_squared, (list, np.ndarray)) else r_squared
    except:
        r_squared_val = r_squared
    print(f"Linear Regression R^2: {float(r_squared_val):.4f}")
    
    # Plotting Figure 2
    fig = plt.figure(figsize=(16, 12))
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Panel A: Scatter Plot
    ax1 = plt.subplot(2, 2, 1)
    sns.scatterplot(x='true_rho', y='pred_rho', data=df, alpha=0.7, ax=ax1, s=60, color='b')
    # Identity line
    ax1.plot([0.85, 1.00], [0.85, 1.00], 'k--', label='Identity (y=x)')
    # Regression line
    x_range = np.linspace(0.85, 1.00, 100)
    ax1.plot(x_range, intercept + slope * x_range, 'r-', label=f'Fit ($R^2={float(r_squared_val):.2f}$)')
    ax1.set_title("A. Prediction Accuracy (True vs. Predicted)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Ground Truth FC Correlation ($\\it{\\rho}_{true}$)", fontsize=12)
    ax1.set_ylabel("Predicted FC Correlation ($\\it{\\rho}_{pred}$)", fontsize=12)
    ax1.set_xlim(0.85, 1.00)
    ax1.set_ylim(0.6, 1.2)
    ax1.legend()
    
    # Panel B: Distribution Overlay
    ax2 = plt.subplot(2, 2, 2)
    sns.kdeplot(df['true_rho'], fill=True, label='Ground Truth (Raw)', ax=ax2, color='blue', alpha=0.5)
    sns.kdeplot(df['pred_rho'], fill=True, label='Predicted (Raw)', ax=ax2, color='orange', alpha=0.5)
    
    # Calculate Mean-Shift Bias
    bias = mean_true - mean_pred
    df['pred_rho_shifted'] = df['pred_rho'] + bias
    
    # Plot Mean-Shifted Predicted
    sns.kdeplot(df['pred_rho_shifted'], fill=False, label='Predicted (Mean-Shifted)', ax=ax2, color='darkorange', linestyle='--', linewidth=2.5)
    
    ax2.set_title("B. Density Distribution of FC Correlation", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Pearson Correlation ($\\it{\\rho}$)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    # Adding vertical line for the 80% threshold
    ax2.axvline(x=0.8, color='red', linestyle=':', label='Clinical Threshold (0.8)')
    ax2.legend(loc='upper left', fontsize=10)
    
    # Panel C: Error Histogram
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(df['error'], bins=20, kde=True, ax=ax3, color='purple')
    ax3.set_title("C. Prediction Error Distribution", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Absolute Error ($|\\it{\\rho}_{true} - \\it{\\rho}_{pred}|$)", fontsize=12)
    ax3.set_ylabel("Subject Count", fontsize=12)
    
    # Panel D: Categorical Pass Rate Bar Chart
    ax4 = plt.subplot(2, 2, 4)
    categories = ['Failed (< 0.8)', 'Good (0.8 - 0.9)', 'Excellent ($\\geq$ 0.9)']
    counts = [failed, pass_80 - pass_90, pass_90]
    colors = ['#e74c3c', '#f1c40f', '#2ecc71']
    bars = ax4.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax4.set_title("D. Clinical Reliability Pass Rate", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Number of Subjects (%)", fontsize=12)
    ax4.set_ylim(0, max(counts) + 15)
    
    plt.tight_layout(pad=3.0)
    
    # Save the figure
    out_path = reports_dir / 'Figure2_Validation.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved successfully to: {out_path}")
    
    # Save also to docs/figure
    docs_figure_dir = Path("docs/figure")
    docs_figure_dir.mkdir(parents=True, exist_ok=True)
    out_path_docs = docs_figure_dir / 'Figure2_Validation.png'
    plt.savefig(out_path_docs, dpi=300, bbox_inches='tight')
    print(f"Figure logically exported to documentation hub: {out_path_docs}")

if __name__ == '__main__':
    main()
