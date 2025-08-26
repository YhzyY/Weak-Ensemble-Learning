from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
# import os
from utils import l2_norm,sigmoid,cross_entropy,average_MD,get_data,load_member_results
from sklearn.metrics import f1_score

from itertools import product
import optuna
import os

path = "/mnt/hdd/zm/code/temp/"

def ensemble_predictions(member_results, weights, eval_type='soft'):
    """
    Args:
        member_results:
            - Soft: (n_models, n_samples, 2)
            - Hard: (n_models, n_samples)
        weights: (n_models,) or list of floats
        eval_type: 'soft' for probabilities, 'hard' for class labels

    Returns:
        - Soft: (n_samples, 2)
        - Hard: (n_samples,)
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    member_results = np.asarray(member_results)

    if eval_type == 'soft':
        if member_results.ndim != 3:
            raise ValueError("Soft predictions must have shape (n_models, n_samples, 2)")
        # Weighted sum across models
        # print(weights.shape,member_results.shape)
        weighted_sum = np.einsum("m,mnk->nk", weights, member_results)  # (n_samples, 2)

        return weighted_sum

    elif eval_type == 'hard':
        if member_results.ndim != 2:
            raise ValueError("Hard predictions must have shape (n_models, n_samples)")
        weighted_votes = np.einsum("m,mn->n", weights, member_results)  # (n_samples,)
        return (weighted_votes >= 0.5).astype(int)

    else:
        raise ValueError("eval_type must be 'soft' or 'hard'")

def evaluate_ensemble(prediction_hard, prediction_soft, y_hard_target, y_soft_target):
    f1_micro_score = f1_score(y_true=y_hard_target, y_pred=prediction_hard, average='micro')
    cross_entropy_score = sigmoid(cross_entropy(targets=y_soft_target, predictions=prediction_soft))
    average_MD_score = average_MD(targets=y_soft_target, predictions=prediction_soft)

    return {
        "f1_micro": f1_micro_score,
        "cross_entropy": cross_entropy_score,
        "average_MD": average_MD_score
    }

def loss_function(params, member_hard_results, member_soft_results, 
        y_hard_target, y_soft_target,
        input_info, shuffle_models=False, return_indices=False,random_state=None):
    weights = params[:-1]
    n_ens_float = params[-1]

    total_models = len(member_soft_results)
    k = int(np.clip(round(n_ens_float), 1, total_models))

    weights = np.maximum(weights[:k], 0)
    if weights.sum() == 0:
        weights += 1e-6
    weights /= weights.sum()
    
    # Initialize random generator
    rng = np.random.RandomState(random_state)
    
    if shuffle_models:
        indices = rng.permutation(total_models)[:k]  # Shuffle all models and take first k
        member_hard_results = np.array(member_hard_results)[indices]
        member_soft_results = np.array(member_soft_results)[indices]
    else:
        indices = np.arange(k)

    soft_results_k = member_soft_results[:k]
    hard_results_k = member_hard_results[:k]
    # weights_k = weights[:k]

    # Use them in the ensemble prediction
    soft_preds = ensemble_predictions(soft_results_k, weights, eval_type='soft')
    hard_preds = ensemble_predictions(hard_results_k, weights, eval_type='hard')

    metrics = evaluate_ensemble(hard_preds, soft_preds, y_hard_target, y_soft_target)

    loss = (
        input_info["alpha"] * (-metrics["f1_micro"]) +
        input_info["beta"] * metrics["cross_entropy"] +
        input_info["gamma"] * metrics["average_MD"] +
        input_info["lambda"] * l2_norm(weights)
    )
    if return_indices:
        return loss, indices, k, (random_state if shuffle_models else None)
    return loss  # Default: return only loss for optimization

def optimization_objective(x, member_hard_results, member_soft_results, 
                         y_hard_target, y_soft_target, input_info):
    return loss_function(x, member_hard_results, member_soft_results,
                       y_hard_target, y_soft_target, input_info,
                       shuffle_models=True, return_indices=False,
                       random_state=None)

def learn_objective(input_info):
    # Load data
    member_hard_results_dev, member_soft_results_dev = load_member_results(
        input_info["dataset"], "dev", input_path=path+f'Results/predictions/{input_info["method"]}/')
    _, y_hard_dev, y_soft_dev = get_data(input_info["dataset"], "dev")

    n_models = len(member_soft_results_dev)
    bounds = [(0, 1)] * n_models + [(1, n_models)]
    
    # Store best solution info
    best_solution = {'params': None, 'random_state': None}
    
    # Modified callback to capture best parameters
    def callback(xk, convergence):
        nonlocal best_solution
        # Create a consistent random state based on parameters
        best_solution['params'] = xk
        best_solution['random_state'] = int(abs(hash(tuple(xk))) % (2**32))
    
    # Run optimization with the standalone objective function
    result = differential_evolution(
        optimization_objective,
        bounds=bounds,
        args=(member_hard_results_dev, member_soft_results_dev,
             y_hard_dev, y_soft_dev, input_info),
        strategy="best1bin",
        maxiter=100,
        tol=1e-6,
        seed=42,
        workers=-1,
        # disp=True, # Print loss on each step
        init="random",
        callback=callback
    )

    # Reconstruct the exact best solution with indices
    final_loss, best_indices, best_k, _ = loss_function(
        best_solution['params'],
        member_hard_results_dev,
        member_soft_results_dev,
        y_hard_dev,
        y_soft_dev,
        input_info,
        shuffle_models=True,
        return_indices=True,
        random_state=best_solution['random_state']
    )

    best_weights = best_solution['params'][:-1][:best_k]
    best_weights = np.maximum(best_weights, 0)
    if best_weights.sum() == 0:
        best_weights += 1e-6
    best_weights /= best_weights.sum()

    # Final evaluation on test set with the same model selection
    member_hard_results_test, member_soft_results_test = load_member_results(
        input_info["dataset"], "test", input_path=path+f'Results/predictions/{input_info["method"]}/')
    _, y_hard_test, y_soft_test = get_data(input_info["dataset"], "test")

    # Recreate the same shuffle for test data
    rng = np.random.RandomState(best_solution['random_state'])
    test_indices = best_indices
    
    selected_hard_test = np.array(member_hard_results_test)[test_indices]
    selected_soft_test = np.array(member_soft_results_test)[test_indices]

    final_hard_preds = ensemble_predictions(selected_hard_test, best_weights, eval_type='hard')
    final_soft_preds = ensemble_predictions(selected_soft_test, best_weights, eval_type='soft')

    final_metrics = evaluate_ensemble(final_hard_preds, final_soft_preds, y_hard_test, y_soft_test)

    return {
        "weights": best_weights,
        "n_ensembles": best_k,
        "loss": result.fun,
        "f1_micro": final_metrics["f1_micro"],
        "cross_entropy": final_metrics["cross_entropy"],
        "average_MD": final_metrics["average_MD"],
        "indices": best_indices  # The indices that were used on dev set
    }


def test():
    input_info = {
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "lambda": 1.0,
        "dataset": "ArMIS",
        "method": "RF",
    }

    result = learn_objective(input_info)
    print(result)
    print("Optimal ensemble size:", result["n_ensembles"])
    print("Optimal weights:", result["weights"])
    print("Loss value:", result["loss"])
    print("F1 micro:", result['f1_micro'])
    print("Cross Entropy:", result['cross_entropy'])
    print("Average MD:", result["average_MD"])
    print("Indices:", result["indices"])
    pass


def grid_search_hyperparameters(dataset="ArMIS", method="RF", 
                              hyperparam_values=None, 
                              output_path="Reports/Scipy_optimised_reports/"):
    """
    Perform grid search over hyperparameters and save results to CSV.
    
    Args:
        dataset: Name of the dataset to use (default: "ArMIS")
        method: Modeling method (default: "RF")
        hyperparam_values: List of values to test for each hyperparameter
                         (default: [0, 0.0001, 0.001, 0.01, 0.1, 1.0])
        output_path: Path to save results 
    
    Returns:
        DataFrame with all results
    """
    output_file=output_path+f"{dataset}_{method}_hyperparameter_results.csv"

    # Set default hyperparameter values if not provided
    if hyperparam_values is None:
        hyperparam_values = [0, 0.0001, 0.001, 0.01, 0.1, 1.0]
    
    # Create all combinations
    param_combinations = product(hyperparam_values, repeat=4)
    
    results = []
    total_combinations = len(hyperparam_values)**4
    
    print(f"Starting grid search for {dataset} with {method} method")
    print(f"Testing {total_combinations} combinations...")
    
    for i, (alpha, beta, gamma, mu) in enumerate(param_combinations, 1):
        print(f"\nCombination {i}/{total_combinations}: "
              f"α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}, λ={mu:.4f}")
        
        input_info = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "lambda": mu,
            "dataset": dataset,
            "method": method
        }
        
        try:
            # Run optimization
            result = learn_objective(input_info)
            
            # Store results
            results.append({
                "dataset": dataset,
                "method": method,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda": mu,
                "n_ensembles": result["n_ensembles"],
                "loss": result["loss"],
                "f1_micro": result["f1_micro"],
                "cross_entropy": result["cross_entropy"],
                "average_MD": result["average_MD"],
                "weights": str(result["weights"]),
                "indices": str(result.get("indices", [])),
            })
            
            # Save progress every 10 combinations
            if i % 10 == 0 or i == total_combinations:
                pd.DataFrame(results).to_csv(output_file, index=False)
                print(f"Saved progress to {output_file} ({len(results)} rows)")
                
        except Exception as e:
            print(f"Failed for combination {i}: {str(e)}")
            continue
    
    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"\nGrid search complete! Results saved to {output_file}")
    print(f"Successful runs: {len(results)}/{total_combinations}")
    
    return results_df


def optuna_objective(trial, dataset, method):
    # Suggest values for each hyperparameter
    alpha = trial.suggest_float("alpha", 1e-6, 1.0, log=True)
    beta  = trial.suggest_float("beta", 1e-6, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 1e-6, 1.0, log=True)
    mu    = trial.suggest_float("lambda", 1e-6, 1.0, log=True)

    print(f"\n***Trail{trial.number}: "
              f"α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}, λ={mu:.4f}")

    input_info = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "lambda": mu,
        "dataset": dataset,
        "method": method
    }

    try:
        result = learn_objective(input_info)
        # Store extra info in the trial for analysis
        trial.set_user_attr("n_ensembles", result['n_ensembles'])
        trial.set_user_attr("f1_micro", result['f1_micro'])
        trial.set_user_attr("cross_entropy", result['cross_entropy'])
        trial.set_user_attr("average_MD", result['average_MD'])
        trial.set_user_attr("weights", result['weights'])
        trial.set_user_attr("indices", result['indices'])
        return result['loss']  # We want to minimize the loss
    except Exception as e:
        trial.set_user_attr("error", str(e))
        return float("inf")  # Penalize failed runs

def optuna_hyperparameter_search(dataset="ArMIS", method="RF",
                                  n_trials=50,
                                  output_path="Reports/Scipy_optimised_reports/"):
    print(f"Starting Optuna optimization for {dataset} with {n_trials} trials...")

    study = optuna.create_study(direction="minimize")

    study.optimize(lambda trial: optuna_objective(trial, dataset, method), n_trials=n_trials)

    # Collect results from the study
    results = []
    for t in study.trials:
        if t.value == float("inf"):
            continue  # skip failed trials
        results.append({
            "dataset": dataset,
            "method": method,
            "n_ensembles": t.user_attrs.get("n_ensembles",None),
            "loss": t.value,
            "alpha": t.params["alpha"],
            "beta": t.params["beta"],
            "gamma": t.params["gamma"],
            "lambda": t.params["lambda"],
            "f1_micro": t.user_attrs.get("f1_micro", None),
            "cross_entropy": t.user_attrs.get("cross_entropy", None),
            "average_MD": t.user_attrs.get("average_MD", None),
            "weights": str(t.user_attrs.get("weights", None)),
            "indices": str(t.user_attrs.get("indices", [])),
        })

    df = pd.DataFrame(results)

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"{dataset}_{method}_optuna_results.csv")
    df.to_csv(out_file, index=False)
    print(f"\nOptuna search complete. Results saved to: {out_file}")
    print("Best trial:", study.best_trial.params)

    return df


def load_results(csv_path):
    """
    Load the results CSV with proper reconstruction of list-type columns
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with lists properly converted from strings
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    def parse_space_list(list_str):
        if pd.isna(list_str) or list_str == '[]':
            return []
        try:
            # Remove brackets and split by spaces
            return np.fromstring(list_str.strip('[]'), sep=' ').tolist()
        except:
            return []

    for col in ['weights', 'indices']:
        if col in df.columns:
            df[col] = df[col].apply(parse_space_list)
    
    return df


def predict_with_ensemble(result_dict, member_hard_results, member_soft_results):
    """
    Make predictions using a trained ensemble solution.
    
    Args:
        result_dict: Dictionary containing the ensemble solution in your format:
            {
                "weights": best_weights,
                "n_ensembles": best_k,
                "indices": test_indices,  # or dev_indices if preferred
                ... 
            }
        member_hard_results: List/array of hard predictions from all member models
        member_soft_results: List/array of soft predictions from all member models
            
    Returns:
        Tuple of (hard_predictions, soft_predictions)
    """
    # Extract solution parameters
    weights = np.array(result_dict["weights"])
    k = result_dict["n_ensembles"]
    indices = result_dict["indices"]  # Using test indices by default
    
    # Ensure we have the right number of weights and indices
    weights = weights[:k]
    indices = indices[:k]
    
    # Select the specified models
    hard_results_k = np.array(member_hard_results)[indices]
    soft_results_k = np.array(member_soft_results)[indices]
    
    # Normalize weights (just in case)
    weights = np.maximum(weights, 0)
    if weights.sum() == 0:
        weights += 1e-6
    weights /= weights.sum()
    
    # Make predictions
    hard_preds = ensemble_predictions(hard_results_k, weights, eval_type='hard')
    soft_preds = ensemble_predictions(soft_results_k, weights, eval_type='soft')
    
    return hard_preds, soft_preds



if __name__ == '__main__':
    # test()
    # method= "RF"
    # 'ArMIS',
    # for dataset in ['ConvAbuse','MD-Agreement','HS-Brexit']:
    method = 'BERT-final2'
    for dataset in ['ArMIS', 'ConvAbuse','MD-Agreement','HS-Brexit']:
        # results=grid_search_hyperparameters(dataset=dataset,method=method)
        results = optuna_hyperparameter_search(dataset=dataset,method=method,n_trials=10)
        print("\nTop 5 combinations by F1 score:")
        print(results.sort_values("f1_micro", ascending=False).head(5))

