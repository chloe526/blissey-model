import numpy as np


def classification_metrics(y,pred):
    y=np.asarray(y);pred=np.asarray(pred)
    if not len(y):raise ValueError('No predictions')
    tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum())
    fn=int(((y==1)&(pred==0)).sum());tp=int(((y==1)&(pred==1)).sum())
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else None
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    return dict(accuracy=(tp+tn)/len(y),precision=precision,recall_fall=recall,f1=f1,
        confusion_matrix=[[tn,fp],[fn,tp]],class_distribution=[int((y==0).sum()),int((y==1).sum())],
        precision_defined=bool(tp+fp),f1_defined=bool(2*tp+fp+fn))
