import keras.backend as K
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, label_ranking_average_precision_score
from math import sqrt
import tensorflow as tf

def get_multilabel_metrics(cm):
    TN = cm[:, 0, 0].sum()
    FN = cm[:, 1, 0].sum()
    TP = cm[:, 1, 1].sum()
    FP = cm[:, 0, 1].sum()
    return TP, FP, TN, FN

# def label_ranking(y_true, y_pred):
#   return label_ranking_average_precision_score(y_true, y_pred), None

def mcc(y_true, y_pred, cm=None, multilabel=False):
  if multilabel:
    TP, FP, TN, FN = get_multilabel_metrics(cm)
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return MCC, None 
  else:
    return matthews_corrcoef(y_true, y_pred), None

def accuracy(y_true, y_pred, cm=None, multilabel=False):
  if multilabel:
    TP, FP, TN, FN = get_multilabel_metrics(cm)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    return ACC, None
  else:
    return accuracy_score(y_true, y_pred), None

def f1(y_true, y_pred):
  f1_weighted = f1_score(y_true, y_pred, average='weighted')
  f1_class = f1_score(y_true, y_pred, average=None)
  return f1_weighted, f1_class

def specificity(y_true, y_pred, cm, classes, multilabel=False):

  if multilabel:
    spc = []
    weights = []
    for c in cm:
      tp = np.diag(c)
      fn = c.sum(axis=1) - tp 
      fp = c.sum(axis=0) - tp
      tn = c.sum() - (tp + fn + fp)  
      spc.append(tn / (tn + fp))  
      weights.append(c.sum(axis=1))

    spc = np.array(spc)
    weights = np.array(weights)
    weighted = np.average(spc, weights=weights)
    return weighted, spc
  else:

    # Class level metrics
    TP = np.diag(cm)
    FN = cm.sum(axis=1) - TP 
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FN + FP)  

    spec_class = TN / (TN + FP)
    weights = cm.sum(axis=1)
    weighted = np.average(spec_class, weights=weights)
    return weighted, spec_class

def precision(y_true, y_pred):
  prec_weighted = precision_score(y_true, y_pred, average='weighted') 
  prec_class = precision_score(y_true, y_pred, average=None)
  return prec_weighted, prec_class

def recall(y_true, y_pred):
  rec_weighted = recall_score(y_true, y_pred, average='weighted')
  rec_class = recall_score(y_true, y_pred, average=None)
  return rec_weighted, rec_class