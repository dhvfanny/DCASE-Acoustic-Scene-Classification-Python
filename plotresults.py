from sklearn.externals import joblib
import glob 
import re
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np

def plot_cm(cm, acc, targets, title="Confusion Matrix", cmap=plt.cm.Blues, norm=True):
    if(norm):
        cm = cm.astype(float)/cm.sum(axis=1)[:, np.newaxis]
    
    fig = plt.figure()
    ax = plt.gca()
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Clf: ' + title + ' Acc: ' + str(round(acc,4)*100))
    plt.colorbar()

    tick_marks = np.arange(len(targets))
    plt.xticks(tick_marks, targets, rotation=45)
    plt.yticks(tick_marks, targets)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # plt.show()
    
    fig.savefig(title + '.png',  bbox_inches='tight')
    
    plt.close()
    

for f in glob.glob('result*.pkl'):
   res = joblib.load(f)
   clf_name = re.search('(?<=result)\w+(?=.pkl)', f).group(0)
   print clf_name
   plot_cm(res['tot_cm'], res['result']['overall_accuracy'], res['result']['class_wise_accuracy'].keys(), title=clf_name)

