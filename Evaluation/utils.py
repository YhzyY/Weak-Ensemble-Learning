import numpy as np
import pandas as pd
import math
# from utils import l2_norm,sigmoid,cross_entropy,average_MD

# helper method
def l2_norm(weights):
    return np.linalg.norm(weights)

def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))

# 0. the metrifcs used in model training and prediction evaluation

# cross entropy loss, soft evaluation
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

# helper method to store predictor results
def saveList(myList,filename):
    # the filename has the 'npy' extension 
    np.save(filename+'.npy',myList)
    print("Saved successfully!")

# helper method to load predictor results
def loadList(filename):
    # the filename has the 'npy' extension 
    tempNumpyArray=np.load(filename+'.npy')
    return tempNumpyArray.tolist()


def save_memeber_results(member_hard_results,member_soft_results,dataset_name,split,output_path='predictions/BERT-hard/'):
    hard_fname = output_path + f'{dataset_name}_{split}_hard_results'
    soft_fname = output_path + f'{dataset_name}_{split}_soft_results'
    print(hard_fname)
    saveList(member_hard_results,hard_fname)
    print(soft_fname)
    saveList(member_soft_results,soft_fname)
    pass

def load_member_results(dataset_name,split,input_path='predictions/BERT-hard/'):
    hard_fname = input_path + f'{dataset_name}_{split}_hard_results'
    soft_fname = input_path + f'{dataset_name}_{split}_soft_results'
    # print(hard_fname)
    member_hard_results = loadList(hard_fname)
    # print(soft_fname)
    member_soft_results = loadList(soft_fname)
    return member_hard_results,member_soft_results

def get_data(dataset, split):
    df = pd.read_csv(f'/mnt/hdd/zm/code/temp/Data/{dataset}_{split}.csv')
    X = df.text.tolist()
    y_hard = df.hard_label.tolist()
    y_soft = list(zip(df.soft_label_0.tolist(), df.soft_label_1.tolist()))
    return X,y_hard,y_soft