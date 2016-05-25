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
    dataset =TUTAcousticScenes_2016_DevelopmentSet('../../../../saved/data/')
    dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)

    y_true_all_folds = []
    y_pred_all_folds = []
    likelihoods_all_folds = []
    filenames_all_folds = []
    for fold in dataset.folds(mode='folds'):
        results = []
        result_filename = get_result_filename(fold=fold, path=os.path.join(result_path, 'evaluation_results/673ca1889888954dfc7fa390bdf8bdef/ce5202bdb7340f963e9496e6effd06f5/'))
        if os.path.isfile(result_filename):
            with open(result_filename, 'rt') as f:
                for row in csv.reader(f, delimiter='\t'):
                    results.append(row)
	else:
	    print 'File not found'

        y_true = []
        y_pred = []
        likelihoods = []
	filenames = []
        for result in results:
            filenames.append(result[0])
  	    y_true.append(dataset.file_meta(result[0])[0]['scene_label'])
            y_pred.append(result[1])
	    lls = np.array(result[2:]).astype(float)
	    lls[~np.isfinite(lls)] = min(lls[np.isfinite(lls)])	
            likelihoods.append(lls)

        dcase2016_scene_metric.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
        filenames_all_folds = filenames_all_folds + filenames
        y_true_all_folds = y_true_all_folds + y_true
        y_pred_all_folds = y_pred_all_folds + y_pred
        likelihoods_all_folds = likelihoods_all_folds + likelihoods
    eval_results = dcase2016_scene_metric.results()
    return {'fn': filenames_all_folds, 
	    'acc': eval_results['overall_accuracy'],
            'yp': y_pred_all_folds,
            'yt': y_true_all_folds,
            'll': likelihoods_all_folds,
	    'cr': classification_report(y_true_all_folds, y_pred_all_folds, dataset.scene_labels,digits=3), 
	    'cm': confusion_matrix(y_true_all_folds, y_pred_all_folds)}



def merge_outputs_to_make_combined_feature(total_result):
    combined_feature = {}
    combined_feature['fn'] = total_result[0]['fn']; 
    combined_feature['yt'] = total_result[0]['yt'];
    combined_feature['ll'] = np.empty([len(total_result[0]['ll']),0]);	
    i = 0
    for result in total_result:	
	print 'Merging LL results from Resultfile: ' + str(i)  
	i=i+1
	combined_feature['ll'] = np.hstack((combined_feature['ll'], result['ll']))
    print combined_feature['ll'].shape
    for i in xrange(0,len(combined_feature['fn'])):
	audio_fn = os.path.split(combined_feature['fn'][i])[1]
	audio_fn = os.path.splitext(audio_fn)[0]
        com_ll = combined_feature['ll'][i]
	file_with_path = os.path.expanduser('~/ASem7/DCASE-Python/saved/features/2016/merger/features/abc/'+ audio_fn + '.txt')

	print file_with_path
	if np.all(np.isfinite(com_ll.astype(float))) == False:
	    if i == 73 :
	        print com_ll
		print str(i) + " is null"	
	pickle.dump(com_ll, open(file_with_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	#numpy.savetxt(file_with_path,com_ll,fmt='%4.8f', newline=" ")

acc = 0.0
max_acc = 0.0
total_result = []
for i in xrange(0,40):
    print 'Reading Resultfile: ' + str(i) 
    total_result.append(get_individual_result('../traps'+ str(i)+'/system/baseline_dcase2016_task1/'))

merge_outputs_to_make_combined_feature(total_result)




	


