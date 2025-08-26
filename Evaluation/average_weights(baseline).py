# average weight

import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

path = "/mnt/hdd/zm/code/temp/"
np.random.seed(42)

def l2_norm(weights):
    return np.linalg.norm(weights)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def cross_entropy(targets, predictions, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce

# average Manhattan Distance evaluation
def average_MD(targets, predictions):
    distances = []
    for target, prediction in zip(targets, predictions):
        # Compute the Manhattan Distance for a single pair
        distance = sum(abs(p - t) for p, t in zip(prediction, target))
        distances.append(round(distance, 5))
    # Compute and return the average Manhattan Distance
    average_distance = round(sum(distances) / len(distances), 5) if distances else 0
    return average_distance

def error_rate(targets, predictions):
    match_scores = []
    for target, prediction in zip(targets, predictions):
        # Compute the total absolute error for the pair
        errors = sum(abs(t - p) for t, p in zip(target, prediction))
        # Compute a normalized match score: higher is better, 1.0 means perfect match
        match_score = round(1- ((len(target) - errors) / len(target)), 5)
        match_scores.append(match_score)
    # Return the average match score across all pairs
    return float(np.mean(match_scores))

def evaluate_ensemble(prediction_hard, prediction_soft, y_hard_target, y_soft_target):
    # f1_micro_score
    f1_micro_score = f1_score(y_true=y_hard_target, y_pred=prediction_hard, average='micro')
    # cross_entropy_score
    cross_entropy_score = sigmoid(cross_entropy(targets=y_soft_target, predictions=prediction_soft))
    # average_MD_score
    average_MD_score = float(average_MD(targets=y_soft_target, predictions=prediction_soft))
    # error_rate_score
    error_rate_score  = float(error_rate(targets=y_soft_target, predictions=prediction_soft))
    return {"f1_micro": f1_micro_score, "cross_entropy": cross_entropy_score, "average_MD": average_MD_score, "error_rate": error_rate_score}

def get_data(dataset, split):
    df = pd.read_csv(path + f'Data/{dataset}_{split}.csv')
    X = df.text.tolist()
    y_hard = df.hard_label.tolist()
    y_soft = list(zip(df.soft_label_0.tolist(), df.soft_label_1.tolist()))
    return X, y_hard, y_soft

def ensemble_predictions(member_results, weights, eval_metrics='hard'):
    summed = np.tensordot(member_results, weights, axes=((0), (0)))
    return (summed > 0.5).astype(int) if eval_metrics == 'hard' else summed
    # return np.argmax(summed, axis=1) if eval_metrics == 'hard' else summed


def select_member_indices(all_member, select_member):
    selected_row_indices = np.random.choice(
        np.arange(all_member),  # the number of ensemble models in stock
        size=select_member,
        replace=False
    )
    return selected_row_indices

# 5. load all pre-trained member models and get their prediction for X
def get_results_from_all_member_models(selected_member_indices, split='dev'):
    prediction_path = path + "Results/predictions/BERT-hard"
    member_hard_results = []
    member_soft_results = []
    print("Selected member models are : ", selected_member_indices)
    for filename in os.listdir(prediction_path):
        if filename.startswith(f"{input_info['dataset']}_{split}") and filename.endswith(".npy"):
            filepath = os.path.join(prediction_path, filename)
            member_results = np.load(filepath)
            print(f"Will fetch data in file : {filename}, data shape: {member_results.shape}")
            # extract selected num_members models' results from the all models' results
            selected_rows = member_results[selected_member_indices]
            if 'hard' in filename:
                member_hard_results = selected_rows
            elif 'soft' in filename:
                member_soft_results = selected_rows

    if len(member_soft_results) == 0 or len(member_hard_results) == 0:
        print("Cannot find prediction results according to dataset name")
    return member_hard_results, member_soft_results



for dataset in ['ArMIS', 'ConvAbuse', 'MD-Agreement', 'HS-Brexit']:
    model_name = 'aubmindlab/bert-base-arabertv2' if dataset == 'ArMIS' else 'bert-base-uncased'
    results_list = []
    model_count = 10
    # if dataset == 'ArMIS':
    #     model_count = 3
    # elif dataset == 'ConvAbuse':
    #     model_count = 5
    # elif dataset == 'MD-Agreement':
    #     model_count = 5
    # elif dataset == 'HS-Brexit':
    #     model_count = 5

    for i in range(1, model_count+1):
        input_info = {
            'dataset': dataset,
            'n_member': i,
            'transformer_name': model_name,
            'split': 'dev',
            # the way to choose candidates to train the model: [:'sample', 'majority']
            'candidates_choose_method': 'sample',
            # available eval_metric: ["f1_micro", "cross_entropy", "average_MD"]
            'eval_metric': 'f1_micro',
            # for the initial test, alpha, beta, gamma should be all 1. the regularisation term mu is usually a value in [0.0001, 0.001,0.001, 0.1,1]
            'alpha': 1,
            'beta': 1,
            'gamma': 1,
            'mu': 0.1,
            'run': 0,
            'random_shuffle': 1
        }

        # have trained 10 member models in stock
        selected_member_indices = select_member_indices(model_count, input_info['n_member'])
        # tune the best weights using OptunaSearch
        average_weights = np.full(input_info['n_member'], 1.0/input_info['n_member'])
        print("Weights will be used in evaluation : ", average_weights)

        # evaluate the best weights and draw figure
        print("Start evaluating the best weights...")
        X_test, y_hard_test, y_soft_test = get_data(input_info["dataset"], 'test')
        member_hard_results, member_soft_results = get_results_from_all_member_models(selected_member_indices, split='test')
        prediction_hard = ensemble_predictions(member_hard_results, average_weights, eval_metrics='hard')
        prediction_soft = ensemble_predictions(member_soft_results, average_weights, eval_metrics='soft')
        evaluate_metrics = evaluate_ensemble(prediction_hard, prediction_soft, y_hard_test, y_soft_test)
        # combine all kind of evaluation score by weights
        joint_loss = input_info["alpha"] * (- evaluate_metrics['f1_micro']) + input_info["beta"] * evaluate_metrics[
            'cross_entropy'] + input_info["gamma"] * evaluate_metrics['average_MD'] + input_info["mu"] * l2_norm(
            average_weights)

        print("all loss based on the test set evaluation", {**evaluate_metrics, **{"joint_loss": joint_loss}})

        # record the current result into results_list
        results_list.append({
            'n_member': input_info["n_member"],
            'selected_member_indices': selected_member_indices.tolist(),
            'weights': average_weights.tolist(),
            'f1_micro': evaluate_metrics['f1_micro'],
            'cross_entropy': evaluate_metrics['cross_entropy'],
            'average_MD': evaluate_metrics['average_MD'],
            'joint_loss': joint_loss,
            'error_rate': evaluate_metrics['error_rate']
        })

    # save the results
    df_results = pd.DataFrame(results_list)
    save_dir = os.path.join(path, "Reports/BERT-hard/Average")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{dataset}.csv")
    df_results.to_csv(file_path, index=False)