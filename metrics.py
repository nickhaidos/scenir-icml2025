import numpy as np
from sklearn.preprocessing import MinMaxScaler


def recall_at_k(actual, predicted, k):
    """
    Computes Recall for a single Query. In case of Dataset, take the MAR@k of all Queries.
    """
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / len(act_set)
    return result


def precision_at_k(actual, predicted, k):
    """
    Computes Precision for a single Query. In case of Dataset, take the MAP@k of all Queries.
    """
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / len(predicted[:k])
    return result


def mrr(actual, queries_results):
    """
    Computes MRR on all Queries.
    `actual` and `queries_results` must be of same dim-0 (number of queries)
    They can be either Numpy Arrays, or just List-of-Lists
    """
    mrr = 0
    for actual_row, queries_results_row in zip(actual, queries_results):
        if len(np.nonzero(np.isin(queries_results_row, actual_row))[0]) != 0:
            reciprocal = 1 / (np.nonzero(np.isin(queries_results_row, actual_row))[0][0] + 1)
        else:
            reciprocal = 0
        mrr += reciprocal
    try:
        Q = actual.shape[0]
    except:
        Q = len(actual)
    return mrr / Q


def average_precision_at_k(actual, predicted, k):
    """
    Computes Average Precision at k, on a single query-result pair.
    `actual` and `predicted` are 1-D arrays (or lists), representing a single query.
    """
    if len(predicted[:k])==0:
        return 0
    mask = np.isin(predicted[:k], actual)
    precisions = np.cumsum(mask) / np.arange(1, len(predicted[:k])+1)
    if np.sum(mask)==0:
        return 0
    average_precision = np.sum(precisions * mask) / np.sum(mask)
    return average_precision


def map_at_k(actual, predicted, k):
    """
    Computes Mean Average Precision.
    `actual` and `predicted` are 2-D arrays (or lists), representing all queries from the dataset.
    """
    map = 0
    for actual_row, predicted_row in zip(actual, predicted):
        map += average_precision_at_k(actual_row, predicted_row, k)
        
    try:
        Q = actual.shape[0]
    except:
        Q = len(actual)
    return map / Q


def query_ndcg(actual, predicted, relevance, k):
    """
    Computes NDCG of a single query.
    `actual` and `predicted` are 1-D arrays, representing the predicted answers and the correct answers.
    `relevance` is a 1-D array that must have length equal to actual.shape[0]
    """
    
    if len(predicted[:k])==0:
        return 0
    dcg = 0
    kc = 1
    for value in predicted[:k]:
        try:
            dcg += relevance[np.where(actual == value)[0][0]] / np.log2(1+kc)
            kc += 1
        except IndexError as e:
            kc += 1

    idcg = ( relevance[:actual[:k].shape[0]] / np.log2(1 + np.arange(1, actual[:k].shape[0]+1)) ).sum()
    return dcg / idcg


def mean_ndcg(actual_l, predicted_l, ground_truth_l, ground_truth_goal='similarity', k=5):
    """
    Computes Average NDCG among the given queries.
    `actual_l` and `predicted_l` are 2-D arrays/lists, representing all queries from the dataset. They have shape [Q x A], where Q is the number of queries, and A is the number of answers per query.
    
    `ground_truth_l` is a 2-D array/list, contains the similarity (or distance) values for the ground_truth answers (original results, NOT SORTED)
    
    `ground_truth_goal` is either 'similarity' or 'distance', and determines the sorting to be used, for computing relevance scores

    `relevance` is computed from the ground truth values, squeezed to [1,10]
    """

    actual = np.array(actual_l)
    predicted = np.array(predicted_l)
    ground_truth = np.array(ground_truth_l)

    scaler = MinMaxScaler(feature_range=(1,10))
    if ground_truth_goal=='distance':
        ground_truth_relevance = scaler.fit_transform(-np.sort(ground_truth)[:, :50].T).T
    elif ground_truth_goal=='similarity':
        ground_truth_relevance = scaler.fit_transform(-np.sort(-ground_truth)[:, :50].T).T
    else:
        print("ground_truth_goal can either be 'distance' or 'similarity'")

    ndcg = []
    
    for idx in range(actual.shape[0]):
        ndcg.append(query_ndcg(actual[idx, :], predicted[idx, :], ground_truth_relevance[idx, :], k))

    return np.mean(ndcg)


def compute_metrics(ground_truth_rankings, predicted_rankings, ground_truth, ground_truth_goal, k):
    """
    Returns Average MAP@k, MRR, NDCG@k for the given queries.
    `actual_l` and `predicted_l` are 2-D arrays/lists, representing all queries from the dataset. They have shape [Q x A], where Q is the number of queries, and A is the number of answers per query.
    
    `ground_truth_l` is a 2-D array/list, contains the similarity (or distance) values for the ground_truth answers (original results, NOT SORTED)
    
    `ground_truth_goal` is either 'similarity' or 'distance', and determines the sorting to be used, for computing relevance scores
    
    NDCG `relevance` is computed from the ground truth values, squeezed to [1,10]
    """

    return map_at_k(ground_truth_rankings, predicted_rankings, k), mrr(ground_truth_rankings, predicted_rankings), mean_ndcg(ground_truth_rankings, predicted_rankings, ground_truth, ground_truth_goal, k)