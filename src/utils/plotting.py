import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np 
import itertools 
import pandas as pd 
from matplotlib.colors import ListedColormap 
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.style.use('ggplot')
# mpl.rcParams['font.size'] = 8

RED = '#E2513C' # material: '#ef5350'
BLUE = '#4F97C3' # material: '#42a5f5'
FOLD_COLORS = [BLUE, RED, '#988ed5', '#777777', '#8eba42', '#fbc15e', '#ffb5b8', '#cddc39', '#26c6da', '#26a69a'] 
CMAP_COLD = plt.get_cmap('YlGnBu', 256)
CMAP_COLD = ListedColormap(CMAP_COLD(np.linspace(0.3, 0.9, 256)))
CMAP_HOT = plt.get_cmap('YlOrRd', 256)
CMAP_HOT = ListedColormap(CMAP_HOT(np.linspace(0.3, 0.9, 256)))

# Chosen EV colors
CMAP_COLD = plt.get_cmap('YlGnBu', 64)
CMAP_COLD = ListedColormap(CMAP_COLD(np.linspace(0.3, 0.6, 64)))
CMAP_HOT = plt.get_cmap('YlOrRd', 64)
CMAP_HOT = ListedColormap(CMAP_HOT(np.linspace(0.3, 0.6, 64)))
CMAP_HOT_2 = plt.get_cmap('YlOrRd', 8)
CMAP_HOT_2 = ListedColormap(CMAP_HOT_2(np.linspace(0.3, 0.7, 8)))
CMAP_COLD_2 = plt.get_cmap('YlGnBu', 8)
CMAP_COLD_2 = ListedColormap(CMAP_COLD_2(np.linspace(0.3, 0.7, 8)))

CMAP_GRADIENT = plt.get_cmap('YlGnBu', 128)
CMAP_GRADIENT = ListedColormap(CMAP_GRADIENT(np.linspace(0.3, 0.4, 128)))

CMAP_C1 = plt.get_cmap('YlOrRd', 256)
CMAP_C1 = ListedColormap(CMAP_HOT(np.linspace(0.3, 0.9, 256)))

def set_styles():
  mpl.rcParams['ytick.minor.left'] = True
  mpl.rcParams['xtick.minor.bottom'] = True
  mpl.rcParams['axes.facecolor'] = 'white'
  mpl.rcParams['grid.color'] = '747474'
  mpl.rcParams['grid.linewidth'] = 1
  mpl.rcParams['grid.linestyle'] = '--'
  mpl.rcParams['legend.facecolor'] = 'white'
  mpl.rcParams['legend.edgecolor'] = '747474'
  mpl.rcParams['axes.edgecolor'] = 'bababa'

def plot_confusion_matrix_multilabel(cm, title, config, k='', labels=[]):
  fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
  fig.suptitle(title, va='center', x=0.48, y=1.1, size='x-large')
  fig.tight_layout(h_pad=3)
  axes = axes.flatten()


  for cls_cm, ax, label in zip(cm, axes, labels):
    ax.grid(False)
    cmap = ax.imshow(cls_cm, interpolation='nearest', cmap='Blues', vmin=np.min(cm), vmax=np.max(cm))
    #cax = make_axes_locatable(ax).append_axes('right', size='2%', pad=0.10)
    #cb = fig.colorbar(cmap, cax=cax)
    #cb.ax.minorticks_on()
    ax.set_title(label, size='large')
    num_ticks = range(2)
    labels = ['Not Present', 'Present'] 
    ax.set_xticks(num_ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(num_ticks)
    ax.set_yticklabels(labels)

    thresh = cls_cm.max() / 2
    for i, j in itertools.product(range(cls_cm.shape[0]), range(cls_cm.shape[1])):
      if not cls_cm[i, j] == 0:
          ax.text(j, i, format(cls_cm[i, j], 'd'),
                  horizontalalignment="center",
                  color="white" if cls_cm[i, j] > thresh else "black")
      else:
          ax.text(j, i, '-', horizontalalignment="center")

  fig.text(-0.01, 0.50, 'True class', va='center', rotation='vertical', size='large')
  fig.text(0.37, -0.02, 'Predicted class', va='center', rotation='horizontal', size='large')
  cb = fig.colorbar(cmap, ax=list(axes), aspect=25, pad=0.05, location='right', shrink=1)
  cb.ax.minorticks_on()
  fig.savefig('{}/{}_confusion_matrix.pdf'.format(config.log_dir, k), bbox_inches='tight', pad_inches=0)
  # plt.close()



  
def plot_confusion_matrix(cm, title, config, k=''):
  num_classes = config.data['classes']
  plt.grid(False)
  plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=np.min(cm), vmax=np.max(cm))
  plt.title(title)
  cb = plt.colorbar()
  cb.ax.minorticks_on()
  num_ticks = np.arange(num_classes)
  labels = None
  if config.usecase == config.EV:
    classes = np.linspace(0.1, 1, num_classes)
    labels = [f'{classes[i]*100:.2f} %' for i in range(len(classes))]
  elif config.usecase == config.DEPRESSION:
    labels = ['nondep', 'dep']
  else:
    labels = ['Not ready', 'Uncertain', 'Ready'] #np.arange(1, num_classes + 1, dtype=int)

  plt.xticks(num_ticks, labels, rotation=45)
  plt.yticks(num_ticks, labels)

  thresh = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if not cm[i, j] == 0:
      plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    else:
      plt.text(j, i, '-', horizontalalignment="center")

  plt.xlabel('Predicted class')
  plt.ylabel('True class')

  figure = plt.gcf()
  figure.savefig('{}/{}_confusion_matrix.pdf'.format(config.log_dir, k), bbox_inches='tight', pad_inches=0)
  # plt.show()
  plt.close()

def plot_train_history(train_history, log_dir, tmp=False):
  set_styles()
  labels = {
    'loss': 'Training loss',
    'val_loss': 'Validation loss',
    'acc': 'Training accuracy',
    'val_acc': 'Validation accuracy'
  }

  def history_plot(data, of):
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.tight_layout()
    folds = len(train_history['loss'])

    for hist, ax in zip(data.keys(), axes):
      max_ran_epoch = max(map(len, data[hist]))
      # ax.set_xticks(range(0, max_ran_epoch))
      # ax.set_xticklabels(range(1, max_ran_epoch + 1))
      for k, fold_history in enumerate(data[hist]):
        ax.plot(fold_history, marker='.' if max_ran_epoch < 100 else None, color=FOLD_COLORS[k])
        ax.set_ylabel(labels[hist])
        ax.minorticks_on()

    fig.legend([f'Fold {n}' for n in range(1, folds + 1)], loc='center right', borderaxespad=0.1)
    plt.subplots_adjust(right=0.82)
    plt.xlabel('Epoch')
    fig.savefig(f'{log_dir}/{of}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
  
  # Losses 
  validation = {
    'val_loss': train_history['val_loss'],
    'val_acc': train_history['val_acc']
  }

  training = {
    'loss': train_history['loss'],
    'acc': train_history['acc']
  }

  # Plot temporary fold history
  if tmp:
    # Multiple in one
    fig, axes = plt.subplots(2, 2, sharex=True)
    axes = axes.flatten()
    folds = len(train_history['loss'])
    fig.tight_layout(h_pad=4) 

    for hist, ax in zip(train_history.keys(), axes):
      max_ran_epoch = max(map(len, train_history[hist]))
      ax.set_xticks([])

      for k, fold_history in enumerate(train_history[hist]):
        ax.plot(fold_history, marker='.' if max_ran_epoch < 100 else None, color=FOLD_COLORS[k])
        ax.set_title(labels[hist])
        ax.minorticks_on()

    # fig.legend('Current fold', loc='center right', borderaxespad=0.1)
    # plt.subplots_adjust(right=0.80)
    fig.savefig(f'{log_dir}/train_hist_tmp.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
  else:
    history_plot(validation, 'train_hist_val')
    history_plot(training, 'train_hist_train')






def plot_test_history(metric_history, log_dir):
  set_styles()

  # print(metric_history)
  # TODO: Bar plot of all? 
  styles = {
    'base': {
      'edgecolor': 'black',
      'linewidth': 1,
      'width': 0.5,
      'color': BLUE,
      'error_kw': {
        'capsize': 3,
        'capthick': 1,
        'elinewidth': 1,
      }
    },
    'annotations': {
      'color': 'black',
      'size': 10, 
      'horizontalalignment': 'center', 
      'verticalalignment': 'bottom'
    }
  }
  # Average plots
  fig = plt.figure()
  ax = fig.gca()
  xs, ys, sd = metric_history.columns.values, metric_history.mean(), metric_history.std()
  ax.bar(xs, ys, yerr=sd, **styles['base'])
  for x, y in zip(xs, ys):
    ax.text(x, 0.02, f'{y:.2f}', **styles['annotations'])
  ticks = None
  if 'MCC' in ys and (ys['MCC'] < 0 or sd['MCC'] < 0):
    ticks = np.linspace(-1, 1, 11)
  else:
    ticks = np.linspace(0, 1, 6)
  ax.axhline(y=0, color='black')
  ax.set_yticks(ticks)
  ax.set_yticklabels(np.round(ticks, 2))
  ax.tick_params(axis='y', which='minor')
  fig.savefig(f'{log_dir}/test_history_avg.png', bbox_inches='tight', pad_inches=0)
  plt.close()


