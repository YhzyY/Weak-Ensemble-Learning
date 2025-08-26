import pandas as pd
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from pathlib import Path

path = '/mnt/hdd/zm/code/temp/'


def get_results_from_all_member_models(X, input_info):
    member_hard_results = []
    member_soft_results = []
    # split = input_info['split']
    dataset = input_info['dataset']
    run = input_info['run']
    n_member = input_info['n_member']
    transformer_name = input_info['transformer_name']

    model_name = transformer_name.split('/')[1] if '/' in transformer_name else transformer_name

    for i in range(n_member):
        model_path = f"{path}trained_models/BERT-ce/{dataset}/{run}_{dataset}_{model_name}_{i}"
        # model_path = f"trained_models/{dataset}/{run}_{dataset}_{model_name}_{i}"
        print("Try to retrieve model from : ", model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            transformer_name)  # tokenizer is not stored locally, will have to load from online version
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        test_encodings = tokenizer(X, truncation=True, padding=True, max_length=512, return_tensors="pt")

        with torch.no_grad():
            test_outputs = model(**test_encodings)

        # compute soft-aggregated label
        probs = torch.nn.functional.softmax(test_outputs.logits, dim=-1).numpy()
        member_soft_results.append(probs)
        # compute hard-aggregated label
        preds = np.argmax(probs, axis=1)
        member_hard_results.append(preds)

    return member_hard_results, member_soft_results


# helper method to store predictor results
def saveList(myList, filename):
    # the filename has the 'npy' extension
    np.save(filename + '.npy', myList)
    print("Saved successfully!")


def save_memeber_results(member_hard_results, member_soft_results, dataset_name, split):
    output_path = path + 'Results/predictions/BERT-ce/'
    os.makedirs(output_path, exist_ok=True)
    hard_fname = output_path + f'{dataset_name}_{split}_hard_results'
    soft_fname = output_path + f'{dataset_name}_{split}_soft_results'
    print(hard_fname)
    saveList(member_hard_results, hard_fname)
    print(soft_fname)
    saveList(member_soft_results, soft_fname)
    pass


def get_data(dataset, split):
    df = pd.read_csv(path + f'Data/{dataset}_{split}.csv')
    X = df.text.tolist()
    y_hard = df.hard_label.tolist()
    y_soft = list(zip(df.soft_label_0.tolist(), df.soft_label_1.tolist()))
    return X, y_hard, y_soft


for dataset in ['ArMIS', 'ConvAbuse', 'MD-Agreement', 'HS-Brexit']:
    model_name = 'aubmindlab/bert-base-arabertv2' if dataset == 'ArMIS' else 'bert-base-uncased'
    input_info = {
        # available dataset: ['ArMIS','ConvAbuse','MD-Agreement','HS-Brexit']
        'dataset': dataset,
        'n_member': 1,
        # ArMIS dataset contains Arabic, which is not supported by bert-base-uncased, so it should use bert-base-arabertv2 instead
        'transformer_name': model_name,
        'split': 'test',
        # 'split':'test',
        # available eval_metric: ["f1_micro", "cross_entropy", "average_MD"]
        'eval_metric': 'f1_micro',
        # for the initial test, alpha, beta, gamma should be all 1. the regularisation term mu is usually a value in [0.0001, 0.001,0.001, 0.1,1]
        'alpha': 1,
        'beta': 1,
        'gamma': 1,
        'mu': 1,
        'run': 0,
        'random_shuffle': 1
    }


    X_test, _, _ = get_data(input_info["dataset"], input_info['split'])
    member_hard_results, member_soft_results = get_results_from_all_member_models(X_test, input_info)
    save_memeber_results(member_hard_results, member_soft_results, input_info["dataset"], input_info['split'])
