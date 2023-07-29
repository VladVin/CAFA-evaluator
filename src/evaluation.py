import numpy as np
import logging
import pandas as pd
import multiprocessing as mp


# Computes the root terms in the dag
def get_roots_idx(dag):
    return np.where(dag.sum(axis=1) == 0)[0]


# Computes the leaf terms in the dag
def get_leafs_idx(dag):
    return np.where(dag.sum(axis=0) == 0)[0]


# Return a mask for all the predictions (matrix) >= tau
def solidify_prediction(pred, tau):
    return pred >= tau


# computes the f metric for each precision and recall in the input arrays
def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def compute_s(ru, mi):
    return np.sqrt(ru**2 + mi**2)
    # return np.where(np.isnan(ru), mi, np.sqrt(ru + np.nan_to_num(mi)))


def compute_metrics_w_(tau_arr, g, pred, toi, n_gt, ic_arr):

    metrics = np.zeros((len(tau_arr), 3), dtype='float32')  # cov, wpr, wrc

    for i, tau in enumerate(tau_arr):

        p = solidify_prediction(pred.matrix[:, toi], tau)

        # Coverage, number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (p.sum(axis=1) > 0).sum()

        # Terms subsets
        intersection = np.logical_and(p, g)  # TP

        # Weighted precision, recall
        n_pred = (p * ic_arr[toi]).sum(axis=1)
        n_intersection = (intersection * ic_arr[toi]).sum(axis=1)

        metrics[i, 1] = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float32'),
                                  where=n_pred > 0).sum()
        metrics[i, 2] = np.divide(n_intersection, n_gt, out=np.zeros_like(n_intersection, dtype='float32'),
                                  where=n_gt > 0).sum()

    return metrics


def compute_metrics(pred, gt, tau_arr, toi, toi_ia, ic_arr, n_cpu=0):
    """
    Takes the prediction and the ground truth and for each threshold in tau_arr
    calculates the confusion matrix and returns the coverage,
    precision, recall, remaining uncertainty and misinformation.
    Toi is the list of terms (indexes) to be considered
    """
    # Parallelization
    if n_cpu == 0:
        n_cpu = mp.cpu_count()

    if ic_arr is not None:
        g = gt.matrix[:, toi_ia]
        n_gt = (g * ic_arr[toi_ia]).sum(axis=1)
        arg_lists = [[tau_arr, g, pred, toi_ia, n_gt, ic_arr] for tau_arr in np.array_split(tau_arr, n_cpu)]
        with mp.Pool(processes=n_cpu) as pool:
            metrics = np.concatenate(pool.starmap(compute_metrics_w_, arg_lists), axis=0)
        columns = ["wcov", "wpr", "wrc"]

    return pd.DataFrame(metrics, columns=columns)


def evaluate_prediction(prediction, gt, ontologies, tau_arr, normalization='cafa', n_cpu=0):

    dfs = []
    for p in prediction:
        ns = p.namespace
        ont = [o for o in ontologies if o.namespace == ns][0]

        # Number of predicted proteins
        ne = np.full(len(tau_arr), gt[ns].matrix[:, ont.toi].shape[0])

        # wcov, wpr, wrc
        metrics = compute_metrics(p, gt[ns], tau_arr, ont.toi, ont.toi_ia, ont.ia, n_cpu)

        for column in ["wpr", "wrc"]:
            if column in metrics.columns:
                if normalization == 'gt' or (column in ["rc", "wrc", "ru", "mi"] and normalization == 'cafa'):
                    # Normalize by gt
                    metrics[column] = np.divide(metrics[column], ne,
                                                out=np.zeros_like(metrics[column], dtype='float32'), where=ne > 0)
                else:
                    # Normalize by pred (cov)
                    if column in ["pr", "rc"]:
                        # Normalize by cov
                        metrics[column] = np.divide(metrics[column], metrics["cov"],
                                                    out=np.zeros_like(metrics[column], dtype='float32'), where=metrics["cov"] > 0)
                    else:
                        # Normalize by weighted cov
                        metrics[column] = np.divide(metrics[column], metrics["wcov"],
                                                    out=np.zeros_like(metrics[column], dtype='float32'), where=metrics["wcov"] > 0)

        metrics['ns'] = [ns] * len(tau_arr)
        metrics['tau'] = tau_arr

        if ont.ia is not None:
            ne = np.full(len(tau_arr), gt[ns].matrix[:, ont.toi_ia].shape[0])
            metrics['wcov'] = np.divide(metrics['wcov'], ne, out=np.zeros_like(metrics['wcov'], dtype='float32'), where=ne > 0)
            metrics['wf'] = compute_f(metrics['wpr'], metrics['wrc'])

        dfs.append(metrics)

    return pd.concat(dfs)
