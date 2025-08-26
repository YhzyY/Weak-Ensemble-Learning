import os
import math
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

os.environ["RAY_TEMP_DIR"] = "/mnt/hdd/zm/tmp/ray"
# os.environ["RAY_STORAGE"] = "/mnt/hdd/zm/code/temp/ray/storage"
import ray
from ray import tune
from sklearn.metrics import f1_score
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import itertools

path = "/mnt/hdd/zm/code/temp/"
np.random.seed(42)
ray.init(
    _temp_dir="/mnt/hdd/zm/tmp/ray",
)


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
        match_score = round(1 - ((len(target) - errors) / len(target)), 5)
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
    error_rate_score = float(error_rate(targets=y_soft_target, predictions=prediction_soft))
    return {"f1_micro": f1_micro_score, "cross_entropy": cross_entropy_score, "average_MD": average_MD_score,
            "error_rate": error_rate_score}


def get_data(dataset, split):
    df = pd.read_csv(path + f'Data/{dataset}_{split}.csv')
    X = df.text.tolist()
    y_hard = df.hard_label.tolist()
    y_soft = list(zip(df.soft_label_0.tolist(), df.soft_label_1.tolist()))
    return X, y_hard, y_soft


def ensemble_predictions(member_results, weights, eval_metrics='hard'):
    summed = np.tensordot(member_results, weights, axes=((0), (0)))
    return (summed > 0.5).astype(int) if eval_metrics == 'hard' else summed


def select_member_indices(all_member, select_member):
    selected_row_indices = np.random.choice(
        np.arange(all_member),  # the number of ensemble models in stock
        size=select_member,
        replace=False
    )
    return selected_row_indices.tolist()


# 5. load all pre-trained member models and get their prediction for X
def get_results_from_all_member_models(selected_members, split='dev'):
    prediction_path = path + "Results/predictions/BERT-final2"
    member_hard_results = []
    member_soft_results = []
    print("Selected member models are : ", selected_members)
    for filename in os.listdir(prediction_path):
        if filename.startswith(f"{input_info['dataset']}_{split}") and filename.endswith(".npy"):
            filepath = os.path.join(prediction_path, filename)
            member_results = np.load(filepath)
            print(f"Will fetch data in file : {filename}, data shape: {member_results.shape}")
            # extract selected num_members models' results from the all models' results
            selected_rows = member_results[selected_members]
            if 'hard' in filename:
                member_hard_results = selected_rows
            elif 'soft' in filename:
                member_soft_results = selected_rows

    if len(member_soft_results) == 0 or len(member_hard_results) == 0:
        print("Cannot find prediction results according to dataset name")
    return member_hard_results, member_soft_results


# 6. learn and optimize ensemble weights for a set of pre-trained member models
def tune_weights(config, y_hard_dev, y_soft_dev):
    print("******", input_info["dataset"], "********* num of members selected :", config['n_member'], "*********")
    member_hard_results, member_soft_results = get_results_from_all_member_models(config['selected_member_indices'],
                                                                                  split='dev')

    weights = config['weights']
    print("Testing config:", config)
    print("member_hard_results, member_soft_results", len(member_hard_results), len(member_soft_results))

    # compute ensemble output and evaluate tune weights using dev set
    prediction_hard = ensemble_predictions(member_hard_results, weights, eval_metrics='hard')
    prediction_soft = ensemble_predictions(member_soft_results, weights, eval_metrics='soft')
    evaluate_metrics = evaluate_ensemble(prediction_hard, prediction_soft, y_hard_dev, y_soft_dev)

    # combine all kind of evaluation score by weights
    joint_loss = input_info["alpha"] * (- evaluate_metrics['f1_micro']) + input_info["beta"] * evaluate_metrics[
        'cross_entropy'] + input_info["gamma"] * evaluate_metrics['average_MD'] + input_info["mu"] * l2_norm(weights)
    tune.report({**dict(itertools.islice(evaluate_metrics.items(), 3)), **{"joint_loss": joint_loss}})


# 6. define search spavce by Optuna define-by-run API
def define_optuna_search_space(trial: optuna.Trial):
    selected_n_member = trial.suggest_int("n_member", 1, input_info['n_member'])
    selected_member_indices = select_member_indices(input_info['n_member'], selected_n_member)
    raw_weights = [
        trial.suggest_float(f"raw_weight_{i}", 0, 10.0)
        for i in range(selected_n_member)
    ]
    sum_raw_weights = np.sum(raw_weights)
    if sum_raw_weights == 0:
        normalized_weights = np.ones(selected_n_member) / selected_n_member
    else:
        normalized_weights = np.array(raw_weights) / sum_raw_weights

    return {
        "n_member": selected_n_member,
        "selected_member_indices": selected_member_indices,
        "weights": normalized_weights.tolist()
    }


# 6. learn and optimize ensemble weights for a set of pre-trained member models
def tune_the_best_weights():
    X_dev, y_hard_dev, y_soft_dev = get_data(input_info["dataset"], 'dev')

    optuna_search = OptunaSearch(
        metric="joint_loss",
        mode="min",
        space=define_optuna_search_space
    )

    # scheduler = ASHAScheduler(
    #     time_attr='epoch',
    #     max_t=100,
    #     grace_period=10,
    #     reduction_factor=3,
    #     brackets=1,
    # )

    tuner = tune.Tuner(
        lambda config: tune_weights(config, y_hard_dev, y_soft_dev),
        tune_config=tune.TuneConfig(
            metric="joint_loss",
            mode="min",
            search_alg=optuna_search,
            # scheduler=scheduler,
            num_samples=20,
            # max_concurrent_trials=10
        )
    )

    result_grid = tuner.fit()
    # record the optimization search log
    result_df = result_grid.get_dataframe()
    print(result_df)
    best_result = result_grid.get_best_result()
    print("The best weights found by ray tune: ", best_result.config)
    return best_result.config['n_member'], best_result.config['selected_member_indices'], best_result.config['weights']


# 7. todo: drawer to visualize evaluation results
def n_members_drawer(n_member, single_scores, ensemble_scores,
                     weighted_ensemble_scores, dataset, eval_metrics):
    x_axis = [i for i in range(1, n_member + 1)]
    fig, ax = plt.subplots()
    ax.plot(x_axis, single_scores, marker='o', linestyle='None', label='single model')
    ax.plot(x_axis, ensemble_scores, marker='o', label='average weighted ensemble')
    ax.plot(x_axis, weighted_ensemble_scores, marker='o', label='optimized weighted ensemble')
    ax.legend()
    if eval_metrics == 'f1_micro':
        y_label = 'F1 (micro-average)'
    elif eval_metrics == 'cross_entropy':
        y_label = 'Cross Entropy'
    elif eval_metrics == 'average_MD':
        y_label = 'Average Manhattan Distance'
    else:
        print("invalid eval_metrics input, use f1_micro instead")
        y_label = 'F1 (micro-average)'
    plt.title(f'{dataset}')
    plt.xlabel("n")
    plt.ylabel(f"{y_label}")
    plt.savefig(path + f'Figs{dataset}({eval_metrics}).png')
    plt.close(fig)  # to avoid opening multiple plots by default


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

    for alpha in [0, 0.1, 0.01, 0.001, 0.0001, 1]:
        for beta in [0, 0.1, 0.01, 0.001, 0.0001, 1]:
            for gamma in [0, 0.1, 0.01, 0.001, 0.0001, 1]:
                for mu in [0, 0.1, 0.01, 0.001, 0.0001, 1]:
                    input_info = {
                        'dataset': dataset,
                        'n_member': model_count,
                        'transformer_name': model_name,
                        'split': 'dev',
                        # the way to choose candidates to train the model: [:'sample', 'majority']
                        'candidates_choose_method': 'sample',
                        # available eval_metric: ["f1_micro", "cross_entropy", "average_MD"]
                        'eval_metric': 'f1_micro',
                        # for the initial test, alpha, beta, gamma should be all 1. the regularisation term mu is usually a value in [0.0001, 0.001,0.01, 0.1,1]
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'mu': mu,
                        'run': 0,
                        'random_shuffle': 1
                    }

                    # tune the best weights using OptunaSearch
                    best_n_member, best_member_indices, best_weights = tune_the_best_weights()
                    print("Weights will be used in evaluation : ", best_weights)

                    # evaluation
                    print("Start evaluation")
                    X_test, y_hard_test, y_soft_test = get_data(input_info["dataset"], 'test')
                    member_hard_results, member_soft_results = get_results_from_all_member_models(best_member_indices, split='test')
                    prediction_hard = ensemble_predictions(member_hard_results, best_weights, eval_metrics='hard')
                    prediction_soft = ensemble_predictions(member_soft_results, best_weights, eval_metrics='soft')
                    evaluate_metrics = evaluate_ensemble(prediction_hard, prediction_soft, y_hard_test, y_soft_test)

                    # combine all kind of evaluation score by weights
                    joint_loss = input_info["alpha"] * (- evaluate_metrics['f1_micro']) + input_info["beta"] * evaluate_metrics[
                        'cross_entropy'] + input_info["gamma"] * evaluate_metrics['average_MD'] + input_info["mu"] * l2_norm(
                        best_weights)

                    print("all loss based on the test set evaluation", {**evaluate_metrics, **{"joint_loss": joint_loss}})

                    # record the current result into results_list
                    results_list.append({
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'mu': mu,
                        'n_member': input_info["n_member"],
                        'best_n_member': best_n_member,
                        'best_member_indices': best_member_indices,
                        'best_weights': best_weights,
                        'f1_micro': evaluate_metrics['f1_micro'],
                        'cross_entropy': evaluate_metrics['cross_entropy'],
                        'average_MD': evaluate_metrics['average_MD'],
                        'joint_loss': joint_loss,
                        'error_rate': evaluate_metrics['error_rate']
                    })

    # save the results
    df_results = pd.DataFrame(results_list)
    save_dir = os.path.join(path, "Reports/BERT-final2/Raytune-args")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{dataset}.csv")
    df_results.to_csv(file_path, index=False)
