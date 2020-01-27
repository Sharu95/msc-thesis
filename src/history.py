import sys 
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
sys.path.append('./utils')
import metrics
import plotting
import pandas as pd 
import copy 

class TrainingHistory():
  def __init__(self, log_dir):
    self.train_history = {
      'loss': [],
      'val_loss': [],
      'acc': [],
      'val_acc': [],
    }
    self.total_training_time = []
    self.log_dir = log_dir

  def save_statistics(self):
    stats = ['mean', 'std', 'min', 'max']
    train_history = pd.read_csv(f'{self.log_dir}/_train.log')
    train_stats = (train_history.describe().T)[stats].round(3)
    train_stats.index = train_stats.index.values
    train_stats.to_csv(f'{self.log_dir}/stats/_train_history_stats.csv', index_label='metric')
    train_stats.to_latex(f'{self.log_dir}/stats/_train_history_stats.tex')

  def save_history(self):
    total_training_time = pd.DataFrame(data={'seconds': self.total_training_time})
    total_training_time.to_csv(f'{self.log_dir}/_train_history_time.csv', index=False)
    train_hist_df = pd.DataFrame(self.train_history)
    train_hist_df.to_csv(f'{self.log_dir}/_train_history.csv', index=False)

    plotting.plot_train_history(self.train_history, self.log_dir)

  def update_history(self, history, training_time):
    self.total_training_time.append(training_time)
    for train_metric, train_history in history.items():
      self.train_history[train_metric].append(train_history)
    
    # Plots for each fold to visualise training
    plotting.plot_train_history(self.train_history, self.log_dir, tmp=True)


class EvaluationHistory:
  def __init__(self, config):
    self.config = config
    self.evaluations = {
      'ACC': [],
      'PREC': [],
      'SPEC': [],
      'REC': [],
      'F1': [],
      'MCC': []
    }
    # if config.usecase == config.FOOTBALL_READINESS:
    #   # self.evaluations['LRAP'] = []
    #   del self.evaluations['SPEC']
    # else:
    #   self.evaluations['MCC'] = []

    self.class_evaluations = copy.deepcopy(self.evaluations)

  def save_statistics(self):
    stats = ['mean', 'std', 'min', 'max']
    eval_hist = pd.DataFrame(self.evaluations)
    eval_stats = (eval_hist.describe().T)[stats].round(3)
    eval_stats.index = eval_stats.index.values 
    eval_stats.to_csv(f'{self.config.log_dir}/stats/_test_history_stats.csv', index_label='metric')
    eval_stats.to_latex(f'{self.config.log_dir}/stats/_test_history_stats.tex')
    
  def save_history(self):
    num_classes = self.config.data['classes']
    eval_hist = pd.DataFrame(self.evaluations)
    eval_hist.to_csv(f'{self.config.log_dir}/_test_history.csv', index=False)
    plotting.plot_test_history(eval_hist, self.config.log_dir)
  
  def custom_evaluate(self, y_pred, y_test, fold_nr, columns):
    num_classes = self.config.data['classes']
    true = y_test.copy()
    pred = y_pred.copy()
    threshold = self.config.network['sigmoid_threshold']
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    cm = multilabel_confusion_matrix(true, pred)
    metric_functions = {
      'ACC': metrics.accuracy,
      'PREC': metrics.precision,
      'SPEC': metrics.specificity,
      'REC': metrics.recall,
      'F1': metrics.f1,
      'MCC': metrics.mcc
    }

    weighted = class_metric = None
    for metric, compute_metric in metric_functions.items():

      if metric == 'MCC' or metric == 'ACC':
        weighted, class_metric = compute_metric(true, pred, cm=cm, multilabel=True)
        class_metric = [weighted] * num_classes
        print(f'| {metric}: {weighted:.2f}')
      elif metric == 'SPEC':
        weighted, class_metric = compute_metric(true, pred, cm, num_classes, multilabel=True)
      else:
        weighted, class_metric = compute_metric(true, pred)

      self.evaluations[metric].append(weighted)
      self.class_evaluations[metric].append(class_metric)
      
      # Override function with computed metric and return this object
      metric_functions[metric] = weighted

    ACC = metric_functions['ACC']
    MCC = metric_functions['MCC'] 
    cm_title = f'K: {fold_nr} | ACC: {ACC:.2f} | MCC: {MCC:.2f}'
    np.save(f'{self.config.log_dir}/cm/cm{fold_nr}.npy', cm)
    plotting.plot_confusion_matrix_multilabel(cm, cm_title, self.config, k=fold_nr, labels=columns)

    return metric_functions

  def evaluate(self, y_pred, y_test, fold_nr):
    num_classes = self.config.data['classes']
    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y_test, axis=1)

    metric_functions = {
      'ACC': metrics.accuracy,
      'PREC': metrics.precision,
      'SPEC': metrics.specificity,
      'REC': metrics.recall,
      'F1': metrics.f1,
      'MCC': metrics.mcc
    }


    # TODO: TEST IF MCC WORKS AFTER ALL (AND SPECIFICITY)
    cm = confusion_matrix(true, pred, labels=list(range(num_classes)))
    weighted = class_metric = None
    for metric, compute_metric in metric_functions.items():
      if metric == 'SPEC':
        weighted, class_metric = compute_metric(true, pred, cm, num_classes)
      else:
        weighted, class_metric = compute_metric(true, pred)

      if metric == 'MCC' or metric == 'ACC':
        class_metric = [weighted] * num_classes
        print(f'| {metric}: {weighted:.2f}')

      self.evaluations[metric].append(weighted)
      self.class_evaluations[metric].append(class_metric)
      
      # Override function with computed metric and return this object
      metric_functions[metric] = weighted

    ACC = metric_functions['ACC']
    MCC = metric_functions['MCC']
    cm_title = f'K: {fold_nr} | ACC: {ACC:.2f} | MCC: {MCC:.2f}'
    np.save(f'{self.config.log_dir}/cm/cm{fold_nr}.npy', cm)
    plotting.plot_confusion_matrix(cm, cm_title, self.config, k=fold_nr)

    return metric_functions