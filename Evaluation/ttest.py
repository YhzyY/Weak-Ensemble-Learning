import numpy as np
from scipy import stats
import os

# BERT-final2 raytune best param
# data_json = {'ArMIS': {'best_member_indices': [5, 1, 3, 4, 0], 'best_weights': [0.19299221977356523, 0.24216708905603715, 0.3840470178572848, 0.15470633064836084,
#                           0.026087342664751968]},
#              'ConvAbuse': {'best_member_indices': [7, 6, 1, 0, 2, 3, 5], 'best_weights': [0.18951700890340714, 0.08017750772784699, 0.1821064891790944, 0.0526591375235044, 0.08445725678188729, 0.0953844756180629, 0.3156981242661969]},
#              'HS-Brexit': {'best_member_indices': [7, 0, 6, 5, 9, 8, 4, 3], 'best_weights': [0.045092779580723504, 0.12734745069209036, 0.1590982139521124, 0.15529352806511046, 0.10278987758848719, 0.22885121974346212, 0.11368163022500398, 0.06784530015300995]},
#              'MD-Agreement': {'best_member_indices': [9, 4, 8, 7, 2, 0], 'best_weights': [0.2933713632675308, 0.0982123108941626, 0.26285637323404487, 0.22376007772628637, 0.002639535749125367, 0.11916033912884974]}}

# annotator raytune best param
data_json = {'ArMIS': {'best_member_indices': [0, 2, 1], 'best_weights': [0.3612766364841487, 0.35875671438881096, 0.2799666491270403]},
             'ConvAbuse': {'best_member_indices': [1, 4, 2, 3], 'best_weights': [0.3262092127880609, 0.1747867211235639, 0.31189624172641994, 0.1871078243619554]},
             'HS-Brexit': {'best_member_indices': [4, 3, 0, 2, 1], 'best_weights': [0.30822691223879406, 0.27232466552496754, 0.09566265130407241, 0.29093056932407063, 0.0328552016080954]},
             'MD-Agreement': {'best_member_indices': [3, 1, 4, 2, 0], 'best_weights': [0.2527523433402538, 0.12600757152625255, 0.07036723903038633, 0.26355074532461775, 0.2873221007784895]}}


def ensemble_predictions(member_results, weights, eval_metrics='hard'):
    summed = np.tensordot(member_results, weights, axes=((0), (0)))
    return (summed > 0.5).astype(int) if eval_metrics == 'hard' else summed


def ttst(main_experiment_scores, baseline_experiment_scores):
    alpha = 0.05
    ttest_independent_result = stats.ttest_ind(main_experiment_scores, baseline_experiment_scores)
    print(f"t-statistic: {ttest_independent_result.statistic}")
    print(f"p-value: {ttest_independent_result.pvalue}")

    if ttest_independent_result.pvalue < alpha:
        print(f"pvalue < {alpha}, significant difference are found")
    else:
        print(f"pvalue >= {alpha}, no significant difference")


def extract_predictions(prediction_path, dataset):
    for filename in os.listdir(prediction_path):
        if filename.startswith(f"{dataset}_test") and filename.endswith(".npy"):
            filepath = os.path.join(prediction_path, filename)
            member_results = np.load(filepath)
            # print(f"Will fetch data in file : {filename}, data shape: {member_results.shape}")
            if 'hard' in filename:
                member_hard_results = member_results
            elif 'soft' in filename:
                member_soft_results = member_results
    if len(member_soft_results) == 0 or len(member_hard_results) == 0:
        print("Cannot find prediction results according to dataset name")
    return member_hard_results, member_soft_results


if __name__ == '__main__':
    for dataset in ['ArMIS', 'ConvAbuse', 'MD-Agreement', 'HS-Brexit']:
        baseline_model = "BERT-ce"
        # main_model = "BERT-final2"
        main_model = "BERT-annotator"

        baseline_prediction_path = f"/mnt/hdd/zm/code/temp/Results/predictions/{baseline_model}"
        main_prediction_path = f"/mnt/hdd/zm/code/temp/Results/predictions/{main_model}"

        print(f"T-Test for dataset: {dataset}, comparing between {baseline_model} and {main_model}")
        baseline_member_hard_results_raw, baseline_member_soft_results_raw = extract_predictions(
            baseline_prediction_path, dataset)
        baseline_member_hard_results, baseline_member_soft_results = baseline_member_hard_results_raw[0], baseline_member_soft_results_raw[0]
        main_member_hard_results_raw, main_member_soft_results_raw = extract_predictions(main_prediction_path, dataset)

        best_member_indices = data_json[dataset]['best_member_indices']
        best_weights = data_json[dataset]['best_weights']
        # calculate tuned prediction differences, parameters are from previous experiment
        main_member_hard_results_tuned = ensemble_predictions(main_member_hard_results_raw[best_member_indices],
                                                              best_weights, eval_metrics='hard')
        main_member_soft_results_tuned = ensemble_predictions(main_member_soft_results_raw[best_member_indices],
                                                              best_weights, eval_metrics='soft')

        print("Hard label differences: ")
        ttst(main_member_hard_results_tuned, baseline_member_hard_results)
        print("Soft label differences: ")
        ttst(main_member_soft_results_tuned[:, 0], baseline_member_soft_results[:, 0])

