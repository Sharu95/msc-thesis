import matplotlib.pyplot as plt
import pandas as pd
import sys 
sys.path.append('../utils/')
import plotting
import numpy as np 
import os 
from random import shuffle

DATA = '../../data'
INTRODUCTION = '../../../thesis/1_introduction/figures'
BACKGROUND = '../../../thesis/2_background/figures'
BASE_FOLDER = f'{DATA}/analysis'

def activity_levels():
  data_control = os.listdir(f'{DATA}/depression/control/')
  data_condition = os.listdir(f'{DATA}/depression/condition/')
  shuffle(data_control)
  shuffle(data_condition)
  patients = 2
  data_condition = data_condition[:patients]
  data_control = data_control[:patients]
  data = data_condition + data_control

  # fig, axes = plt.subplots(2, 2, sharey=True)
  # fig.tight_layout(h_pad=2.5)
  # axes = axes.flatten()
  
  for i in range(len(data)):
    tag = data[i].split('_')[0]
    activity = pd.read_csv(f'{DATA}/depression/{tag}/{data[i]}', parse_dates=True, index_col='timestamp')
    activity = activity.resample('H').mean()
    
    # Plot normalised / scaled
    # activity = (activity - activity.min()) / (activity.max() - activity.min()) 
    color = plotting.RED if tag == 'condition' else plotting.BLUE
    ax = activity.plot(legend=False, color=color, figsize=(3, 2))

    # Tick/x label settings. Datetime formats
    days = activity.index.strftime('%d')
    months = activity.index.strftime('%b')
    years = activity.index.strftime('%Y')
    months, year = pd.unique(months), pd.unique(years).item()
    m_string = f'{months[0]} to {months[-1]}' if len(months) > 1 else months[0]
    ax.set_title(f'{m_string} ({year})', size='large')
    
    # Set tick and labels
    num_ticks = 10
    ticks = ax.get_xticks()
    ticks = np.linspace(min(ticks), max(ticks), num_ticks)
    labels = np.linspace(0, len(days) - 1, num_ticks, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels(days[labels])
    ax.minorticks_on()
    ax.set_xlabel('Day of month', size='large')
    ax.set_ylabel('Activity count', size='large')
    ax.set_xlim(left=min(ticks) - 10, right=max(ticks) + 10)

    fig = plt.gcf()
    fig.savefig(f'{INTRODUCTION}/{tag}_{i}.pdf', bbox_inches='tight', pad_inches=0)
    # plt.show()
  # plt.close()
  





def imagenet():
  with open(f'{BASE_FOLDER}/imagenet.txt') as f:
    winners = [line.split(' ') for line in f.read().split('\n')]
    winners = [(int(winner[0]), float(winner[1])) for winner in winners]
    years, errors = zip(*winners)
    errors = [round(err * 100, 2) for err in errors]

    styles = {
      'base': {
        'edgecolor': 'black',
        'linewidth': 1.5,
        'width': 0.55,
        'color': plotting.BLUE
      },
      'annotations': {
        'color': 'black',
        'size': 10, 
        'horizontalalignment': 'center', 
        'verticalalignment': 'top'
      }
    }
    plt.bar(years, errors, **styles['base'])
    for x, y in zip(years, errors):
      plt.text(x, y-0.5, f'{y:.1f}', **styles['annotations'])
    plt.xlabel('Years')
    plt.ylabel('Error %')
    plt.savefig(f'{BACKGROUND}/ImageNet.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    return years, errors




def activation():
  
  # Activation functions
  sigmoid = lambda xs: 1 / (1 + np.exp(-xs))
  tanh = lambda xs: (np.exp(xs) - np.exp(-xs)) / (np.exp(xs) + np.exp(-xs))
  relu = lambda xs: np.where(xs > 0, xs, 0)
  threshold = lambda xs: np.where(xs > 0, 1, 0)
  softmax = lambda xs: np.exp(xs) / (np.sum(np.exp(xs)))

  fns = {
    'Sigmoid': sigmoid, 
    'Tanh': tanh,
    'ReLU': relu,
    'Threshold': threshold,
    'Softmax': softmax
  }

  def settings(ax, x='x', y='y'):
    ax.minorticks_on()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.axvline(x=0, color='gray', linewidth=1.2, label=None, zorder=1)
    ax.axhline(y=0, color='gray', linewidth=1.2, label=None, zorder=1) 

  # Plot all activations
  for i, (fn, activation) in enumerate(fns.items()):
    xs = np.linspace(-10, 10, 100)
    is_softmax = fn == 'Softmax'
    if is_softmax:
      xs = np.linspace(0, 10, 100)
    df = pd.DataFrame({'xs': xs, 'ys': activation(xs)})
    ax = df.plot(x='xs', y='ys', figsize=(3, 2.5), legend=False, color='black', linewidth=2, zorder=5)
    settings(ax, x='Class' if is_softmax else None, y='Probability' if is_softmax else None)
    fig = plt.gcf()
    fig.savefig(f'{BACKGROUND}/activation_{fn.lower()}.pdf', bbox_inches='tight', pad_inches=0)
  plt.close()


def read_nordpool(read_file, dec='.'):
  series = pd.read_csv(f'{BASE_FOLDER}/{read_file}.csv', decimal=dec)
  fmt = series['date'] + '-' + series['hours'].apply(lambda x: x[:2])
  datetime = pd.to_datetime(fmt, format='%d-%m-%Y-%H')
  series['timestamp'] = datetime
  series.index = datetime
  return series.dropna().resample('16H').mean()


def plot_nordpool():

  consumption = read_nordpool('consumption_2018')
  prices = read_nordpool('prices_2018', dec=',')
  series = {
    'prices': {
      'df': prices,
      'col': 'SYS',
      'clr': plotting.BLUE,
    },
    'consumption': {
      'df': consumption,
      'col': 'NO',
      'clr': plotting.RED,
    }
  }
  for tag, f in series.items():
    (df, col, clr) = f.values()
    ax = df.plot(y=col, legend=None, color=clr, figsize=(6, 3))
    # ax.set_title(f'Energy {tag}', size='medium')
    
    if tag == 'prices':
      ax.set_ylabel('NOK pr. MWh', size='medium')
    else:
      ax.set_ylabel('MWh', size='medium')
    
    ax.set_xlabel('Time', size='medium')
    ax.minorticks_on()
    ax.minorticks_on()

    fig = plt.gcf()
    fig.savefig(f'{INTRODUCTION}/{tag}.pdf', bbox_inches='tight', pad_inches=0)



# def plot_consumption():
#   df = pd.read_csv(f'{DATA}/energy/dataset.csv', index_col='timestamp', parse_dates=True)
#   car1, car2 = df[df['id'] == 1], df[df['id'] == 3]
#   print(car1)
#   # car1 = car1.resample('D').mean()
#   ax = car2.plot(y='bat_used')
#   car2.plot(y='bat_charged', ax=ax)
#   # car2.plot(y='driven')
#   plt.show()

if __name__ == "__main__":
  plotting.set_styles()
  # activity_levels()
  activation()
  # imagenet()
  # plot_nordpool()
  # plot_consumption()