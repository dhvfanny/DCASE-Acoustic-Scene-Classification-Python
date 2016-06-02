from __future__ import division
from src.dataset import *

import numpy as np
from src.evaluation import DCASE2016_SceneClassification_Metrics
from sklearn.metrics import confusion_matrix, classification_report
	
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
        result_filename = get_result_filename(fold=fold, path=os.path.join(result_path, ''))
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
    
#    print y_true_all_folds
    results = dcase2016_scene_metric.results()
    return {'acc': results['overall_accuracy'],
            'yp': y_pred_all_folds,
            'yt': y_true_all_folds,
            'll': likelihoods_all_folds,
	    'cr': classification_report(y_true_all_folds, y_pred_all_folds, dataset.scene_labels,digits=3), 
	    'cm': confusion_matrix(y_true_all_folds, y_pred_all_folds)}



def get_combined_by_weight(weight, result_1, result_2):
	comb_ll = weight*np.array(result_1['ll']).astype(float)+ (1-weight)*np.array(result_2['ll']).astype(float)
       	
        comb_true = result_1['yt']
        #print comb_true
	#print comb_pred
	dataset = TUTAcousticScenes_2016_DevelopmentSet('../../../saved/data/')
	comb_pred = []
	for ind in xrange(0,len(comb_ll)):
	    comb_pred.append(dataset.scene_labels[np.argmax(comb_ll[ind])])

	dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
        dcase2016_scene_metric.evaluate(system_output=comb_pred, annotated_ground_truth=comb_true)
        result_mfcc_trapsults = dcase2016_scene_metric.results()
        return {'acc': result_mfcc_trapsults['overall_accuracy'],
                'yp': comb_pred,
                'yt': comb_true,
                'll': comb_ll,
  		'cr': classification_report(comb_true, comb_pred, dataset.scene_labels, digits=3),
		'cm': confusion_matrix(comb_true, comb_pred)}


def combine(result_1, result_2):
      acc = 0.0
      max_acc = 0.0
      comb_cm = []
	
      for w in np.arange(0, 1.01, 0.01):
		result_c =  get_combined_by_weight(w, result_1, result_2)
                acc = result_c['acc']
                if acc > max_acc:
                        comb_cm = result_c['cm']
                        max_acc = acc
                        max_w = w
                        comb_cr = result_c['cr']
      print "A accuracy :" + str(result_1['acc'])
      print "B accuracy :" + str(result_2['acc'])

      print "A + B accuracy :" + str(max_acc) + " at w : " + str(max_w)




result_mfcc = get_individual_result('../dnn2016med_mfcc/system/baseline_dcase2016_task1/evaluation_results/27ae017f3b0b02950c0c12c8c5769cea/ce5202bdb7340f963e9496e6effd06f5/')
result_traps = get_individual_result('../dnn2016med_traps/merger/system/baseline_dcase2016_task1/evaluation_results/673ca1889888954dfc7fa390bdf8bdef/ce5202bdb7340f963e9496e6effd06f5/')
result_lfcc = get_individual_result('../dnn2016med_lfcc/system/baseline_dcase2016_task1/evaluation_results/27ae017f3b0b02950c0c12c8c5769cea/ce5202bdb7340f963e9496e6effd06f5/')
result_antimfcc = get_individual_result('../dnn2016med_antimfcc/system/baseline_dcase2016_task1/evaluation_results/27ae017f3b0b02950c0c12c8c5769cea/ce5202bdb7340f963e9496e6effd06f5/')
result_gd = get_individual_result('../dnn2016med_gd/system/baseline_dcase2016_task1/evaluation_results/27ae017f3b0b02950c0c12c8c5769cea/ce5202bdb7340f963e9496e6effd06f5/')
result_chroma = get_individual_result('../dnn2016med_chroma/system/dnn_dcase2016/evaluation_results/1bd4f582e255e5adf031153485f7245a/ce5202bdb7340f963e9496e6effd06f5/')
#result_gmm = get_individual_result('../baseline2016_mfcc_21/system/baseline_dcase2016_task1/evaluation_results/56186ae17ac12b7d8ad63430548ebf63/db87907778f6d9eb1f8e5428542fa821/')

print 'MFCC + LFCC'
combine(result_mfcc, result_lfcc)

print '\nMFCC + Antimfcc'
combine(result_mfcc, result_antimfcc)

print '\nLFCC + Antimfcc'
combine(result_lfcc, result_antimfcc)

print '\nMFCC + GD'
combine(result_mfcc, result_gd)

print '\nMFCC + Chroma'
combine(result_mfcc, result_chroma)

print '\nMFCC + Traps'
combine(result_mfcc, result_traps)

#print '\nMFCC + Gmm'
#combine(result_mfcc, result_gmm)


#Large DNN
###########################################################################################
result_mfcc = get_individual_result('../dnn2016med_mfcc_21/system/baseline_dcase2016_task1/evaluation_results/56186ae17ac12b7d8ad63430548ebf63/1d964b8a1551d4edf988970ae415f145/')
result_traps = get_individual_result('../dnn2016med_traps/merger/system/baseline_dcase2016_task1/evaluation_results/d9b3ab21b7d46e1035e2a5f934367dbc/31bc5809969228168e51deae2f314c23/')
result_lfcc = get_individual_result('../dnn2016med_lfcc/system/baseline_dcase2016_task1/evaluation_results/27ae017f3b0b02950c0c12c8c5769cea/1d964b8a1551d4edf988970ae415f145/')
result_antimfcc = get_individual_result('../dnn2016med_antimfcc/system/baseline_dcase2016_task1/evaluation_results/27ae017f3b0b02950c0c12c8c5769cea/1d964b8a1551d4edf988970ae415f145/')
result_gd = get_individual_result('../dnn2016med_gd/system/baseline_dcase2016_task1/evaluation_results/d4f941614ef44580192fc6e95350839d/1d964b8a1551d4edf988970ae415f145/')
result_chroma = get_individual_result('../dnn2016med_chroma/system/dnn_dcase2016/evaluation_results/1bd4f582e255e5adf031153485f7245a/1d964b8a1551d4edf988970ae415f145/')

print 'MFCC + LFCC'
combine(result_mfcc, result_lfcc)

print '\nMFCC + Antimfcc'
combine(result_mfcc, result_antimfcc)

print '\nLFCC + Antimfcc'
combine(result_lfcc, result_antimfcc)

print '\nMFCC + GD'
combine(result_mfcc, result_gd)

print '\nMFCC + Chroma'
combine(result_mfcc, result_chroma)

print '\nMFCC + Traps'
combine(result_mfcc, result_traps)


#gmm is same
#print '\nMFCC + Gmm'
#combine(result_mfcc, result_gmm)




