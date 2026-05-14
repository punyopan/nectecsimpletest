"""
Evaluation metrics: EER and t-DCF.
Ported from ASVspoof 2019 evaluation toolkit.
"""
import sys
import os
import numpy as np


def calculate_tDCF_EER(cm_scores_file, asv_score_file, output_file,
                       printout=True):
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,
        'Ptar': (1 - Pspoof) * 0.99,
        'Pnon': (1 - Pspoof) * 0.01,
        'Cmiss': 1, 'Cfa': 10,
        'Cmiss_asv': 1, 'Cfa_asv': 10,
        'Cmiss_cm': 1, 'Cfa_cm': 10,
    }

    if not os.path.exists(asv_score_file):
        cm_data = np.genfromtxt(cm_scores_file, dtype=str)
        cm_keys = cm_data[:, 2]
        cm_scores = cm_data[:, 3].astype(float)
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_keys == 'spoof']
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
        
        if printout:
            with open(output_file, "w") as f_res:
                f_res.write('\nCM SYSTEM\n')
                f_res.write('\tEER\t\t= {:8.9f} % \n'.format(eer_cm * 100))
                f_res.write('\tmin-tDCF\t\t= N/A (No ASV scores)\n')
        return eer_cm * 100, 0.05  # Return dummy tDCF if none exists

    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)

    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(float)

    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    [Pfa_asv, Pmiss_asv,
     Pmiss_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                               asv_threshold)

    tDCF_curve, CM_thresholds = compute_tDCF(bona_cm, spoof_cm, Pfa_asv,
                                             Pmiss_asv, Pmiss_spoof_asv,
                                             cost_model, print_cost=False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(Equal error rate for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\nTANDEM\n')
            f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))
        os.system(f"cat {output_file}")

    return eer_cm * 100, min_tDCF


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost):
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: missing spoof ASV miss rate.')

    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: scores contain nan or inf.')

    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: provide soft CM scores, not binary decisions')

    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm)

    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] -
         cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    if C1 < 0 or C2 < 0:
        sys.exit('ERROR: negative weights in tDCF')

    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF / np.minimum(C1, C2)

    return tDCF_norm, CM_thresholds
