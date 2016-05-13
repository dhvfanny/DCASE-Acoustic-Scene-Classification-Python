from dnn2016med_mfcc.src.dataset import *

import numpy as np
from src.evaluation import DCASE2016_SceneClassification_Metrics


def get_result_filename(fold, path, extension='txt'):
    if fold == 0:
        return os.path.join(path, 'results.' + extension)
    else:
        return os.path.join(path, 'results_fold' + str(fold) + '.' + extension)


def get_individual_result(result_path):
    dataset =TUTAcousticScenes_2016_DevelopmentSet('../../../saved/data/')
    dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)

    y_true_all_folds = []
    y_pred_all_folds = []
    likelihoods_all_folds = []
    for fold in dataset.folds(mode='folds'):
        results = []
        result_filename = get_result_filename(fold=fold, path=os.path.join(result_path, 'evaluation_results/'))

        if os.path.isfile(result_filename):
            with open(result_filename, 'rt') as f:
                for row in csv.reader(f, delimiter='\t'):
                    results.append(row)

        y_true = []
        y_pred = []
        likelihoods = []
        for result in results:
            y_true.append(dataset.file_meta(result[0])[0]['scene_label'])
            y_pred.append(result[1])
            likelihoods.append(result[2:])

        dcase2016_scene_metric.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
        y_true_all_folds = y_true_all_folds + y_true
        y_pred_all_folds = y_pred_all_folds + y_pred
        likelihoods_all_folds = likelihoods_all_folds + likelihoods
    results = dcase2016_scene_metric.results()
    return {'acc': results['overall_accuracy'],
            'yp': y_pred_all_folds,
            'yt': y_true_all_folds,
            'll': likelihoods_all_folds}


def get_combined_by_weight_result(weight, res1, res2):
        comb_ll = weight*np.array(res1['ll']) + (1-weight)*np.array(res2['ll'])
        comb_pred = np.argmax(comb_ll)
        comb_true = res1['yt']
        dataset = TUTAcousticScenes_2016_DevelopmentSet('../../../saved/data/')
        dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
        dcase2016_scene_metric.evaluate(system_output=comb_pred, annotated_ground_truth=comb_true)
        results = dcase2016_scene_metric.results()
        return {'acc': results['overall_accuracy'],
                'yp': comb_pred,
                'yt': comb_true,
                'll': comb_ll}

acc = 0.0
max_acc = 0.0
result1 = get_individual_result('dnn2016med_mfcc/')
result2 = get_individual_result('dnn2016med_lfcc/')
for w in np.arange(0, 1, 0.1):
    acc = get_combined_by_weight_result(w, result1, result2)['acc']
    if acc > max_acc:
        max_acc = acc

print "Max accuracy :" + str(max_acc) + " at w : " + str(w)
