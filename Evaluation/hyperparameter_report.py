import pandas as pd
import numpy as np
import scipy

"""
Dicts for converting naming
"""
key_dict = {
    'f1':'f1_micro',
    'ce':'cross_entropy',
    'md':'average_MD',
    'loss':'joint_loss'
}

param_dict = {
    'f1_micro': 'F1',
    'cross_entropy': 'CE',
    'average_MD':'MD'
}

def read_data(input_file):
    """Read and return the CSV data as a DataFrame"""
    return pd.read_csv(input_file)

def get_top_results(df, top_n=5):
    """
    Return top n rows for f1_micro, average_MD, and cross_entropy
    Returns a dictionary of DataFrames with keys 'f1', 'md', 'ce'
    """
    return {
        # 'f1': df.nlargest(top_n, 'f1_micro'),
        # 'ce': df.nsmallest(top_n, 'cross_entropy'),  # Smaller cross-entropy is better
        # 'md': df.nsmallest(top_n, 'average_MD')  # Smaller MD is better
        'f1' : df.sort_values(by=['f1_micro', 'average_MD','cross_entropy','joint_loss'],
            ascending=[False, True, True, True]).head(top_n),
        'ce': df.sort_values(by=['cross_entropy', 'average_MD', 'f1_micro', 'joint_loss'],
            ascending=[True, True, False, True]).head(top_n),
        'md' : df.sort_values(by=['average_MD', 'cross_entropy','f1_micro' ,'joint_loss'],
            ascending=[True, True, False, True]).head(top_n),
        'loss': df.sort_values(by=['joint_loss', 'average_MD', 'cross_entropy', 'f1_micro'],
            ascending=[True, True, True, False]).head(top_n)
    }

def find_best_hyperparameters(df,best_target='f1'):
    """Find and return the best hyperparameters based on best_target"""
    best_row = df.loc[df[key_dict[best_target]].idxmax()] if best_target == "f1" else df.loc[df[key_dict[best_target]].idxmin()]
    return {
        'alpha': best_row['alpha'],
        'beta': best_row['beta'],
        'gamma': best_row['gamma'],
        'lambda': best_row['lambda'],
        'f1': best_row['f1_micro'],
        'ce': best_row['cross_entropy'],
        'md': best_row['average_MD'],
        'loss': best_row['joint_loss'],
        'n_ensembles': best_row['best_n_member']
    }

def create_best_params_summary(f1_dict, ce_dict, md_dict, loss_dict):
    """Organizes three metric dictionaries into a combined summary DataFrame.
    
    Args:
        f1_dict: Best params by F1 (higher better) with keys: alpha, beta, gamma, 
                 lambda, f1, ce, md
        ce_dict: Best params by CE (lower better) with same keys
        md_dict: Best params by MD (lower better) with same keys
    
    Returns:
        pd.DataFrame: Combined comparison of all best parameters
    """
    summary_data = [
        {
            'Metric': 'F1',
            **{k: v for k, v in f1_dict.items()},
        },
        {
            'Metric': 'CE',
            **{k: v for k, v in ce_dict.items()},
        },
        {
            'Metric': 'MD',
            **{k: v for k, v in md_dict.items()},
        },
        {
            'Metric': 'Loss',
            **{k: v for k, v in loss_dict.items()},
        }
    ]
    
    return pd.DataFrame(summary_data)

def perform_ablation_study(df, best_target='f1'):
    """Perform ablation study by testing different hyperparameter combinations"""
    ablation_results = []
    
    # Helper function to add results
    def add_result(case_name, subset_df, alpha=None, beta=None, gamma=None, lambda_val=None):
        if not subset_df.empty:
            best_row = df.loc[subset_df[key_dict[best_target]].idxmax()] if best_target == "f1" else df.loc[subset_df[key_dict[best_target]].idxmin()]
            ablation_results.append({
                'Case': case_name,
                'F1': best_row['f1_micro'],
                'CE': best_row['cross_entropy'],
                'MD': best_row['average_MD'],
                'loss': best_row['joint_loss'],
                'alpha': best_row['alpha'] if alpha is None else alpha,
                'beta': best_row['beta'] if beta is None else beta,
                'gamma': best_row['gamma'] if gamma is None else gamma,
                'lambda': best_row['lambda'] if lambda_val is None else lambda_val
            })
    
    # Individual parameters
    add_result('alpha only', 
               df[(df['alpha'] == 1) & (df['beta'] == 0) & (df['gamma'] == 0) & (df['lambda'] == 0)],
               beta=0, gamma=0, lambda_val=0)
    
    add_result('beta only', 
               df[(df['beta'] == 1) & (df['alpha'] == 0) & (df['gamma'] == 0) & (df['lambda'] == 0)],
               alpha=0, gamma=0, lambda_val=0)
    
    add_result('gamma only', 
               df[(df['gamma'] == 1) & (df['alpha'] == 0) & (df['beta'] == 0) & (df['lambda'] == 0)],
               alpha=0, beta=0, lambda_val=0)
    
    add_result('lambda only', 
               df[(df['lambda'] == 1) & (df['alpha'] == 0) & (df['beta'] == 0) & (df['gamma'] == 0)],
               alpha=0, beta=0, gamma=0)
    
    # Add regularisation term
    add_result('alpha + lambda', 
               df[(df['alpha'] == 1) & (df['beta'] == 0) & (df['gamma'] == 0) & (df['lambda'] == 1)],
               beta=0, gamma=0, lambda_val=1)
    
    add_result('beta + lambda', 
               df[(df['beta'] == 1) & (df['alpha'] == 0) & (df['gamma'] == 0) & (df['lambda'] == 1)],
               alpha=0, gamma=0, lambda_val=1)
    
    add_result('gamma + lambda', 
               df[(df['gamma'] == 1) & (df['alpha'] == 0) & (df['beta'] == 0) & (df['lambda'] == 1)],
               alpha=0, beta=0, lambda_val=1)
    
    
    # Parameter combinations
    add_result('alpha + beta', 
               df[(df['alpha'] == 1) & (df['beta'] == 1) & (df['gamma'] == 0) & (df['lambda'] == 0)],
               gamma=0, lambda_val=0)
    
    add_result('alpha + gamma', 
               df[(df['alpha'] == 1) & (df['gamma'] == 1) & (df['beta'] == 0) & (df['lambda'] == 0)],
               lambda_val=0)
    
    add_result('beta + gamma', 
               df[(df['beta'] == 1) & (df['gamma'] == 1) & (df['alpha'] == 0) & (df['lambda'] == 0)],
               lambda_val=0)

    add_result('alpha + beta + gamma', 
               df[(df['alpha'] == 1) & (df['beta'] == 1) & (df['gamma'] == 1) & (df['lambda'] == 0)],
               lambda_val=0)
    

    add_result('alpha + beta + lambda', 
               df[(df['alpha'] == 1) & (df['beta'] == 1) & (df['gamma'] == 0) & (df['lambda'] == 1)],
               gamma=0, lambda_val=1)
    
    add_result('alpha + gamma + lambda', 
               df[(df['alpha'] == 1) & (df['gamma'] == 1) & (df['beta'] == 0) & (df['lambda'] == 1)],
               lambda_val=1)
    
    add_result('beta + gamma + lambda', 
               df[(df['beta'] == 1) & (df['gamma'] == 1) & (df['alpha'] == 0) & (df['lambda'] == 1)],
               lambda_val=1)



    add_result('All parameters', 
               df[(df['alpha'] == 1) & (df['beta'] == 1) & (df['gamma'] == 1) & (df['lambda'] == 1)])
    
    return pd.DataFrame(ablation_results)

def save_results(output_file, top_results, best_params, ablation_results, impact_results):
    """Save all results to an Excel file with multiple sheets"""
    with pd.ExcelWriter(output_file) as writer:
        top_results['loss'].to_excel(writer, sheet_name='Minimum Loss', index=False)
        top_results['f1'].to_excel(writer, sheet_name='Top F1 Scores', index=False)
        top_results['ce'].to_excel(writer, sheet_name='Top CE Scores', index=False)
        top_results['md'].to_excel(writer, sheet_name='Top MD Scores', index=False)
        
        best_params.to_excel(writer, sheet_name='Best Parameters Summary', index=False)
        
        ablation_results.to_excel(writer, sheet_name='Ablation Study', index=False)
        # f1_ablation_results,ce_ablation_results, md_ablation_results = ablation_results
        # f1_ablation_results.to_excel(writer, sheet_name='F1 Ablation Study', index=False)
        # ce_ablation_results.to_excel(writer, sheet_name='CE Ablation Study', index=False)
        # md_ablation_results.to_excel(writer, sheet_name='MD Ablation Study', index=False)
        impact_df, impact_matrix = impact_results
        impact_df.to_excel(writer, sheet_name='Parameter Impact', index=False)
        impact_matrix.to_excel(writer, sheet_name='Parameter Matrix', index=False)
        pass

def analyze_parameter_impact(df):
    """Analyze impact of increasing each hyperparameter value on performance"""
    impact_results = []
    params = ['alpha', 'beta', 'gamma', 'lambda', 'joint_loss', 'best_n_member']
    
    for param in params:
        # Create a copy of the dataframe for this parameter's analysis
        param_df = df.copy()
        
        # Remove rows where THIS parameter is zero (but keep zeros for other parameters)
        param_df = param_df[param_df[param] > 0]
        
        if len(param_df) < 2:  # Need at least 2 values to calculate correlation
            continue


    for param in params:
        # Group by parameter value and calculate mean metrics
        grouped = df.groupby(param).agg({
            'f1_micro': 'mean',
            'cross_entropy': 'mean',
            'average_MD': 'mean',
            # 'n_ensembles': 'mean'
        }).reset_index()
        
        # Calculate correlations
        for metric in ['f1_micro', 'cross_entropy', 'average_MD']: #, 'n_ensembles'
            grouped_param = grouped[param][grouped[param]!=0]
            grouped_metric = grouped[metric][grouped[param]!=0]
            # corr = np.corrcoef(grouped_param, grouped_metric)[0, 1]
            # Use Spearman from Scipy
            corr_result = scipy.stats.spearmanr(grouped_param, grouped_metric)
            impact_results.append({
                'Parameter': param,
                'Metric': param_dict[metric],
                'Correlation': conv_corr(corr_result),
                # 'Correlation p-value': corr_pvalue,
                'Optimal Value': grouped.loc[grouped_metric.idxmax() if metric == 'f1_micro' else grouped_metric.idxmin(), param],
                # 'Max Impact Value': grouped.loc[grouped_metric.idxmax(), param] if metric == 'f1_micro' else grouped.loc[grouped_metric.idxmin(), param],
                'Max Impact': round((grouped_metric.max() if metric == 'f1_micro' else grouped_metric.min()),4)
            })

        impact_df = pd.DataFrame(impact_results)
        impact_matrix = create_enhanced_impact_matrix(impact_df)

    return impact_df,impact_matrix

def conv_corr(corr_result):
    corr = corr_result.statistic
    corr_pvalue = corr_result.pvalue

    new_corr = str(round(corr,2))+'*' if corr_pvalue < 0.05 else round(corr,2)
    new_corr = '+' + str(new_corr) if corr > 0 else str(new_corr)
    return new_corr




def create_enhanced_impact_matrix(input_df):
    # Create basic matrix
    matrix = input_df.pivot(index='Parameter', 
                          columns='Metric', 
                          values='Correlation')
    
    matrix = matrix.reset_index()
    metric_order = ['Parameter', 'F1', 'CE', 'MD']
    matrix = matrix.reindex(columns=metric_order)
    
    return matrix



def analyze_hyperparameters(input_file, output_file, top_n=5):
    """Main function to run the complete analysis"""
    # Read data
    df = read_data(input_file)
    
    # Perform analyses
    top_results = get_top_results(df, top_n)
    
    f1_best_params = find_best_hyperparameters(df, best_target='f1')
    ce_best_params = find_best_hyperparameters(df, best_target='ce')
    md_best_params = find_best_hyperparameters(df, best_target='md')
    loss_best_params = find_best_hyperparameters(df, best_target='loss')
    best_params = create_best_params_summary(f1_best_params, ce_best_params, md_best_params, loss_best_params)

    # ablation_results = (
    #     perform_ablation_study(df,best_target='f1'),
    #     perform_ablation_study(df,best_target='ce'),
    #     perform_ablation_study(df,best_target='md')
    # )

    ablation_results = perform_ablation_study(df,best_target='f1')

    impact_results = analyze_parameter_impact(df)
    
    # Save and display results
    save_results(output_file, top_results, best_params, ablation_results, impact_results)
    # print(ablation_results)
    print(f"\nResults saved to {output_file}")
    pass


if __name__ == "__main__":

    # method ='RF'
    # method = 'BERT-hard'
    # method = 'BERT-annotator'
    method = 'BERT-final2'

    # dataset= 'ArMIS'
    # dataset = 'MD-Agreement'
    # dataset= 'ConvAbuse'
    # dataset= 'HS-Brexit'
    # input_csv = f"reports/scipy_optimised_reports/{dataset}_{method}_hyperparameter_results.csv"
    # output_excel = f"reports/scipy_optimised_reports/summary/{dataset}_{method}_hyperparameter_analysis_results.xlsx"
    # analyze_hyperparameters(input_csv, output_excel, top_n=10)
    
    for dataset in ['ArMIS','ConvAbuse','MD-Agreement','HS-Brexit']:#
        input_csv = f"/mnt/hdd/zm/code/temp/Reports/{method}/Raytune-args/{dataset}.csv"
        output_excel = f"/mnt/hdd/zm/code/temp/Reports/{method}/Raytune-args/summary/{dataset}_{method}_hyperparameter_analysis_results.xlsx"
        analyze_hyperparameters(input_csv, output_excel, top_n=10)
    pass