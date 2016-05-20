from __future__ import division
from src.dataset import *

import numpy as np
from src.evaluation import DCASE2016_SceneClassification_Metrics
from sklearn.metrics import confusion_matrix, roc_auc_score 

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
        result_filename = get_result_filename(fold=fold, path=os.path.join(result_path, 'evaluation_results/27ae017f3b0b02950c0c12c8c5769cea/ce5202bdb7340f963e9496e6effd06f5/'))
        if os.path.isfile(result_filename):
            with open(result_filename, 'rt') as f:
                for row in csv.reader(f, delimiter='\t'):
                    results.append(row)
	else:
	    print 'File not found'

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
            'll': likelihoods_all_folds,
	    'cm': confusion_matrix(y_true_all_folds, y_pred_all_folds)}



def get_combined_by_weight_result(weight, res1, res2):
	comb_ll = weight*np.array(res1['ll']).astype(float)+ (1-weight)*np.array(res2['ll']).astype(float)
       	
        comb_true = res1['yt']
        #print comb_true
	#print comb_pred
	dataset = TUTAcousticScenes_2016_DevelopmentSet('../../../saved/data/')
	comb_pred = []
	for ind in xrange(0,len(comb_ll)):
	    comb_pred.append(dataset.scene_labels[np.argmax(comb_ll[ind])])

	dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
        dcase2016_scene_metric.evaluate(system_output=comb_pred, annotated_ground_truth=comb_true)
        results = dcase2016_scene_metric.results()
        return {'acc': results['overall_accuracy'],
                'yp': comb_pred,
                'yt': comb_true,
                'll': comb_ll,
		'cm': confusion_matrix(comb_true, comb_pred)}


acc = 0.0
max_acc = 0.0
result1 = get_individual_result('../dnn2016med_mfcc/system/baseline_dcase2016_task1/')
result2 = get_individual_result('../dnn2016med_gd_2/system/baseline_dcase2016_task1/')
comb_cm = []
for w in np.arange(0, 1, 0.01):
    res =  get_combined_by_weight_result(w, result1, result2)
    acc = res['acc']
    acc = np.sum(np.diag(res['cm']))/np.sum(np.sum(res['cm']))
    if acc > max_acc:
	comb_cm = res['cm']
        max_acc = acc
	max_w = w
print "MFCC accuracy :" + str(result1['acc']) + ", " +  str(np.sum(np.diag(result1['cm']))/np.sum(np.sum(result1['cm']))) 
print "LPGD-2 accuracy :" + str(result2['acc']) + ", " +  str(np.sum(np.diag(result2['cm']))/np.sum(np.sum(result2['cm']))) 

print "Max accuracy :" + str(max_acc) + " at w : " + str(max_w)

#print result1['cm']
#print result2['cm']
#print comb_cm
acc = 0.0
max_acc = 0.0
result1 = get_individual_result('../dnn2016med_mfcc/system/baseline_dcase2016_task1/')
result2 = get_individual_result('../dnn2016med_gd_5/system/baseline_dcase2016_task1/')
comb_cm = []
for w in np.arange(0, 1, 0.01):
    res =  get_combined_by_weight_result(w, result1, result2)
    acc = res['acc']
    acc = np.sum(np.diag(res['cm']))/np.sum(np.sum(res['cm']))
    if acc > max_acc:
        comb_cm = res['cm']
        max_acc = acc
        max_w = w
print "MFCC accuracy :" + str(result1['acc']) + ", " +  str(np.sum(np.diag(result1['cm']))/np.sum(np.sum(result1['cm'])))
print "LPGD-5 accuracy :" + str(result2['acc']) + ", " +  str(np.sum(np.diag(result2['cm']))/np.sum(np.sum(result2['cm'])))

print "Max accuracy :" + str(max_acc) + " at w : " + str(max_w)

acc = 0.0
max_acc = 0.0
result1 = get_individual_result('../dnn2016med_mfcc/system/baseline_dcase2016_task1/')
result2 = get_individual_result('../dnn2016med_gd_20/system/baseline_dcase2016_task1/')
comb_cm = []
for w in np.arange(0, 1, 0.01):
    res =  get_combined_by_weight_result(w, result1, result2)
    acc = res['acc']
    acc = np.sum(np.diag(res['cm']))/np.sum(np.sum(res['cm']))
    if acc > max_acc:
        comb_cm = res['cm']
        max_acc = acc
        max_w = w
print "MFCC accuracy :" + str(result1['acc']) + ", " +  str(np.sum(np.diag(result1['cm']))/np.sum(np.sum(result1['cm'])))
print "LPGD-20 accuracy :" + str(result2['acc']) + ", " +  str(np.sum(np.diag(result2['cm']))/np.sum(np.sum(result2['cm'])))

print "Max accuracy :" + str(max_acc) + " at w : " + str(max_w)

acc = 0.0
max_acc = 0.0
result1 = get_individual_result('../dnn2016med_mfcc/system/baseline_dcase2016_task1/')
result2 = get_individual_result('../dnn2016med_gd_50/system/baseline_dcase2016_task1/')
comb_cm = []
for w in np.arange(0, 1, 0.01):
    res =  get_combined_by_weight_result(w, result1, result2)
    acc = res['acc']
    acc = np.sum(np.diag(res['cm']))/np.sum(np.sum(res['cm']))
    if acc > max_acc:
        comb_cm = res['cm']
        max_acc = acc
        max_w = w
print "MFCC accuracy :" + str(result1['acc']) + ", " +  str(np.sum(np.diag(result1['cm']))/np.sum(np.sum(result1['cm'])))
print "LPGD-50 accuracy :" + str(result2['acc']) + ", " +  str(np.sum(np.diag(result2['cm']))/np.sum(np.sum(result2['cm'])))

print "Max accuracy :" + str(max_acc) + " at w : " + str(max_w)

acc = 0.0
max_acc = 0.0
result1 = get_individual_result('../dnn2016med_mfcc/system/baseline_dcase2016_task1/')
result2 = get_individual_result('../dnn2016med_gd_100/system/baseline_dcase2016_task1/')
comb_cm = []
for w in np.arange(0, 1, 0.01):
    res =  get_combined_by_weight_result(w, result1, result2)
    acc = res['acc']
    acc = np.sum(np.diag(res['cm']))/np.sum(np.sum(res['cm']))
    if acc > max_acc:
        comb_cm = res['cm']
        max_acc = acc
        max_w = w
print "MFCC accuracy :" + str(result1['acc']) + ", " +  str(np.sum(np.diag(result1['cm']))/np.sum(np.sum(result1['cm'])))
print "LPGD-100 accuracy :" + str(result2['acc']) + ", " +  str(np.sum(np.diag(result2['cm']))/np.sum(np.sum(result2['cm'])))

print "Max accuracy :" + str(max_acc) + " at w : " + str(max_w)

