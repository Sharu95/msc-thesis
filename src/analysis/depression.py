import matplotlib.colors as clrs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import os 
import sys 
sys.path.append('../utils/')
import plotting
import numpy as np 
import datetime

DATA = '../../data/depression'
EXPERIMENTS = '../../../thesis/4_experiments/usecases/depression/figures'
EXP_TABLES = '../../../thesis/4_experiments/usecases/depression/tables'


def read_data(rs='1T'):
  conditions = os.listdir(f'{DATA}/condition')
  controls = os.listdir(f'{DATA}/control')
  files = conditions + controls
  
  init_index = pd.date_range(start='2000-01-01', end='2003-12-31', freq='H')
  conditions, controls = pd.DataFrame(index=init_index), pd.DataFrame(index=init_index)
  
  for filename in files:
    data_class, patient = filename.split('_')
    patient, _ = patient.split('.')
    df = pd.read_csv(f'{DATA}/{data_class}/{filename}', index_col='timestamp', parse_dates=True)
    df = df.drop(columns=['date'])
    df = df.rename(columns={'activity': f'{data_class}_{patient}'})
    if rs:
      df = df.resample(rs).mean()

    if data_class == 'control':
      controls = pd.concat([controls, df], axis=1)
      # controls = controls.merge(df, left_on=controls.index.hour, right_on=df.index.hour, how='left')
    else:
      conditions = pd.concat([conditions, df], axis=1)
      # conditions = conditions.merge(df, left_on=conditions.index.hour, right_on=df.index.hour, how='left')
      pass

  conditions = conditions[conditions.first_valid_index():conditions.last_valid_index()]
  controls = controls[controls.first_valid_index():controls.last_valid_index()]

  return conditions, controls




def hourly_average(cond, contr):
  cond['hour'] = cond.index.hour
  contr['hour'] = contr.index.hour
  cond['day'] = cond.index.weekday
  contr['day'] = contr.index.weekday

  hourly_avg_condition = cond.groupby(by='hour').sum().mean(axis=1)
  hourly_avg_control = contr.groupby(by='hour').sum().mean(axis=1)
  # x_norm = (x - x.min()) / (x.max() - x.min())
  ax = hourly_avg_condition.plot(label='Condition', color=plotting.RED, marker='.')
  ax.plot(hourly_avg_control, label='Control', color=plotting.BLUE, marker='.')
  # plt.title('Hourly average activity levels')
  plt.ylabel('Total activity count')
  plt.legend()
  return plt.gcf()
  # print(controls.head(50))
  # print(conditions.head(50))


def weekly_activity(cond, contr):
  # Scale to 0 - 1
  cond = (cond - cond.min()) / (cond.max() - cond.min())
  contr = (contr - contr.min()) / (contr.max() - contr.min())

  cond['hour'] = cond.index.hour
  contr['hour'] = contr.index.hour
  cond['day'] = cond.index.weekday
  contr['day'] = contr.index.weekday

  fig, axes = plt.subplots(2, 1, sharex=True)
  fig.tight_layout()

  week_act_cond = cond.groupby(by=['day', 'hour']).sum().mean(axis=1).unstack()
  week_act_contr = contr.groupby(by=['day', 'hour']).sum().mean(axis=1).unstack()
  days = ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
  data = {
    'condition': week_act_cond,
    'control': week_act_contr
  }

  for i, key in enumerate(data.keys()):
    ax = axes[i]
    # clr = 'YlOrRd' if key == 'condition' else 'YlGnBu'
    # cmap = ax.imshow(data[key], interpolation='nearest', cmap=plt.get_cmap('YlGnBu', 15), vmin=0, vmax=1)
    cmap = ax.imshow(data[key], interpolation='nearest', cmap=plt.get_cmap('YlOrRd', 20), vmin=0, vmax=1)
    # ax.set_title(key)
    ax.grid(False)
    ax.set_yticks(range(7))
    ax.set_yticklabels(days)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{x:02d}:00' for x in range(0, 24, 2)], rotation=45)
    
    # cax = make_axes_locatable(ax).append_axes('right', size='2%', pad=0.10)
    # cb = fig.colorbar(cmap, cax=cax)
    # cb.ax.minorticks_on()
  cb = fig.colorbar(cmap, ax=list(axes), aspect=45, pad=0.05, location='top', shrink=0.95)
  cb.ax.minorticks_on()
  return fig 


def acf():
  # fig, axes = plt.subplots(2, 2)
  # axes = axes.flatten()
  # fig.tight_layout(h_pad=2.5)

  lags = {
    'H': 24 * 3,
    '8H': 3 * 7 * 2,
    '12H': 2 * 7 * 2,
    '16H': 3 * 7,
  }

  for rs in lags.keys():
    print(f'ACF: {rs}')
    fig, ax = plt.subplots(1, figsize=(3.5, 3))
    fig.tight_layout()
    cond, contr = read_data(rs=rs)
    cond = cond.interpolate(method='time').mean(axis=1)
    contr = contr.interpolate(method='time').mean(axis=1)
    cond_acf = [cond.autocorr(lag) for lag in range(lags[rs] + 1)]
    contr_acf = [contr.autocorr(lag) for lag in range(lags[rs] + 1)]
    ax.plot(range(len(cond_acf)), cond_acf, label='Condition', color=plotting.RED, marker='o', markersize=3.5)
    ax.plot(range(len(contr_acf)), contr_acf, label='Control', color=plotting.BLUE, marker='o', markersize=3.5)
    ax.minorticks_on()
    ax.set_title(f'{rs} resample')
    plt.legend()
    fig.savefig(f'{EXPERIMENTS}/acf_{rs}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close() 
  return fig


def dataset_statistics():
  # Number of conditioned patients
  # Number of control patients
  # Average/min/max/std activity levels for condition group
  # =="== activity levels for control group
  
  scores = pd.read_csv(f'{DATA}/scores.csv')
  condition = scores['number'].apply(lambda x: True if 'condition' in x else False)
  control = scores['number'].apply(lambda x: True if 'control' in x else False)
  
  cond_days = scores[condition]['days']
  contr_days = scores[control]['days']

  stats = pd.DataFrame(data={'Condition': cond_days.describe(), 'Control': contr_days.describe()}).T
  stats['total'] = [cond_days.sum(), contr_days.sum()]
  stats = stats[['count', 'mean', 'std', 'min', 'max', 'total']]
  stats = stats.round(2)
  stats['count'] = stats['count'].astype(int)
  stats['min'] = stats['min'].astype(int)
  stats['max'] = stats['max'].astype(int) 
  stats.columns = ['\\textbf{Patients}', '\\textbf{Average}', '\\textbf{SD}', '\\textbf{Min}', '\\textbf{Max}', '\\textbf{Total}']
  stats.to_latex(f'{EXP_TABLES}/overview.tex', bold_rows=True, escape=False)









def sleep_pattern(cond, contr):
  fig, axes = plt.subplots(3, 1, figsize=(5, 5))
  ax1, ax2, ax3 = axes.flatten()
  fig.tight_layout()

  # Total activity count  
  hourly_avg_condition = cond.resample('H').mean()
  hourly_avg_control = contr.resample('H').mean()
  hourly_avg_condition = hourly_avg_condition.groupby(by=hourly_avg_condition.index.hour).sum().mean(axis=1)
  hourly_avg_control = hourly_avg_control.groupby(by=hourly_avg_control.index.hour).sum().mean(axis=1)
  ax3.plot(hourly_avg_condition, label='Condition', color=plotting.RED, marker='.', markersize=9)
  ax3.plot(hourly_avg_control, label='Control', color=plotting.BLUE, marker='.', markersize=9)
  ax3.legend(loc='upper left')
  ax3.set_xlabel('Hour', size='medium')
  ax3.set_ylabel('Activity', size='medium')
  ax3.minorticks_on()
  ax3.set_xticks(range(0, 23, 2))
  cax = make_axes_locatable(ax3).append_axes('right', size='2%', pad=0.10)
  cax.axis('off')

  # Minutely over night hours
  cond = (cond - cond.min()) / (cond.max() - cond.min())
  contr = (contr - contr.min()) / (contr.max() - contr.min())
  from_hour = 20
  to_hour = 10
  num_hours = 14
  cond_hour = cond.between_time(f'{from_hour:02d}:00', f'{to_hour:02d}:59')
  contr_hour = contr.between_time(f'{from_hour:02d}:00', f'{to_hour:02d}:59')
  cond_night = cond_hour.groupby(by=[cond_hour.index.hour, cond_hour.index.minute], sort=False) 
  contr_night = contr_hour.groupby(by=[contr_hour.index.hour, contr_hour.index.minute], sort=False) 
  cond_night = cond_night.sum().mean(axis=1).unstack()
  contr_night = contr_night.sum().mean(axis=1).unstack()
  
  data = {
    'condition': cond_night,
    'control': contr_night
  }
  
  x_ticks = range(0, 60, 5)
  x_labels = [f':{x:02d}' for x in x_ticks]
  y_ticks = range(0, num_hours + 1, 2)

  for i, key in enumerate(data.keys()):
    ax = axes[i]
    cmap = ax.imshow(data[key], interpolation='nearest', cmap=plt.get_cmap('YlOrRd', 30), vmin=0, vmax=1)
    ax.grid(False)
    ax.set_yticks(y_ticks)
    hours = data[key].index.values[y_ticks]
    y_labels = [f'{x:02d}' for x in hours]
    ax.set_yticklabels(y_labels)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45)
    cax = make_axes_locatable(ax).append_axes('right', size='2%', pad=0.10)
    cb = fig.colorbar(cmap, cax=cax)
    cb.set_ticks(np.linspace(0, 1, 6))
    cb.ax.minorticks_on()

  # cb = fig.colorbar(cmap, ax=list(axes), aspect=45, pad=0.05, location='top')
  # cb.ax.minorticks_on()
  return fig

def prob_distribution():

  # Data
  cond, contr = read_data(rs='D')
  contr_mean = contr.mean(axis=1)
  cond_mean = cond.mean(axis=1)

  # Figure settings
  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
  fig.tight_layout()
  bar_style = { 'edgecolor': 'black', 'linewidth': 1 }

  # Distribution plot
  def make_plot(df, color, ax, name):
    df.plot.hist(color=color, ax=ax, **bar_style, bins=15)
    ax.legend([name])
    new_ax = ax.twinx()
    df.plot.density(color='black', ax=new_ax, style='-')
    new_ax.grid(False)
    ax.minorticks_on()
    new_ax.minorticks_on()
  
  make_plot(contr_mean, plotting.BLUE, ax1, name='Control')
  make_plot(cond_mean, plotting.RED, ax2, name='Condition')
  fig.savefig(f'{EXPERIMENTS}/histogram_kde.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
  plotting.set_styles()
  freq = sys.argv[1]
  # cond, contr = read_data(rs=freq)
  # OLD
  # hourly_avg = hourly_average(cond, contr) 
  # hourly_avg.savefig(f'{EXPERIMENTS}/data_total_hourly.pdf', bbox_inches='tight', pad_inches=0)

  # weekly_avg = weekly_activity(cond, contr)
  # weekly_avg.savefig(f'{EXPERIMENTS}/data_total_weekly.pdf', bbox_inches='tight', pad_inches=0)
  acf_fig = acf()
  # acf_fig.savefig(f'{EXPERIMENTS}/acf.pdf', bbox_inches='tight', pad_inches=0)
  
  # dataset_statistics()
  # prob_distribution()

  # cond, contr = read_data(rs='T')
  # sleep_fig = sleep_pattern(cond, contr)
  # sleep_fig.savefig(f'{EXPERIMENTS}/data_total_sleep.pdf', bbox_inches='tight', pad_inches=0)
