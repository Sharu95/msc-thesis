import matplotlib.pyplot as plt
import pandas as pd
import sys 
sys.path.append('../utils/')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime

# sys.path.append('../models/usecases/')
# from usecases import energy
import plotting
import numpy as np 

ENERGY_DATA = f'../../data/energy'
EXPERIMENTS = '../../../thesis/4_experiments/usecases/energy/figures'
EXP_TABLES = '../../../thesis/4_experiments/usecases/energy/tables'

def acf():
  data = pd.read_csv(f'{ENERGY_DATA}/dataset_raw.csv', parse_dates=True, index_col='timestamp')
  
  lags = {
    'H': 24,
    '4H': 6 * 7,
    '8H': 3 * 14,
    'D': 28,
  }
  lag_labels = {
    'H': r'$n$ hour lag',
    'D': r'$n$ day lag',
  }

  for i, freq in enumerate(lags.keys()):
    fig, ax = plt.subplots(1, figsize=(3.5, 3))
    fig.tight_layout()
    battery = data[['vehicle__id', 'battery_level']]
    battery = battery.groupby(by=['vehicle__id']).resample(freq).mean().unstack(level=0)
    battery = battery['battery_level'].mean(axis=1)
    acf_battery = [battery.autocorr(lag) for lag in range(lags[freq] + 1)]
    ax.plot(range(len(acf_battery)), acf_battery, color=plotting.BLUE, marker='o', markersize=5)
    ax.minorticks_on()
    ax.set_title(f'{freq} resample')
    fig.savefig(f'{EXPERIMENTS}/acf_battery_{freq}.pdf', bbox_inches='tight', pad_inches=0)
  return fig




def charging_pattern():
  """
    OLD: General level average charging pattern/battery levels.
  """
  data = pd.read_csv(f'{ENERGY_DATA}/dataset_raw.csv', parse_dates=True, index_col='timestamp')
  battery = data[['vehicle__id', 'battery_level']]
  fig, axes = plt.subplots(2, 1, sharex=True)
  fig.tight_layout()
  axes = axes.flatten()

  weekly = battery.groupby(by=[battery.index.weekday, battery.index.hour, 'vehicle__id']).mean().unstack()
  weekly = weekly.mean(axis=1).unstack()
  weekly = (weekly - weekly.min()) / (weekly.max() - weekly.min())
  x_ticks = range(0, 24, 2)
  x_labels = [f'{x:02d}:00' for x in x_ticks]
  days = ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

  # Weekly driving pattern
  ax = axes[0]
  cmap = ax.imshow(weekly, interpolation='nearest', cmap=plt.get_cmap('YlGnBu', 20), vmin=0, vmax=1)
  # cmap = ax.imshow(weekly, interpolation='nearest', cmap=plt.get_cmap('YlOrRd', 20))
  ax.grid(False)
  ax.set_yticks(range(7))
  ax.set_yticklabels(days)
  ax.set_xticks(x_ticks)
  ax.set_xticklabels(x_labels, rotation=45)
  cax = make_axes_locatable(ax).append_axes('right', size='2%', pad=0.10)
  cb = fig.colorbar(cmap, cax=cax)
  cb.ax.minorticks_on()

  # Hourly driving pattern
  ax1 = axes[1]
  hourly = battery.groupby(by=[battery.index.hour, 'vehicle__id']).mean().unstack().mean(axis=1)
  hourly.plot(ax=ax1, color=plotting.BLUE, marker='.', markersize=9)
  ax1.set_xticks(x_ticks)
  ax1.set_xticklabels(x_labels, rotation=45)
  ax1.set_xlim(left=-0.5, right=len(hourly) - 0.5)
  cax = make_axes_locatable(ax1).append_axes('right', size='2%', pad=0.10)
  cax.axis('off')
  ax1.set_xlabel(None)
  return fig  







def dataset_statistics():
  data = pd.read_csv(f'{ENERGY_DATA}/dataset.csv', parse_dates=True, index_col='timestamp')
  raw_data = pd.read_csv(f'{ENERGY_DATA}/dataset_raw.csv', parse_dates=True, index_col='timestamp')
  
  cols = ['', 'Car ID', 'Observations', 'Days', 'Period']
  stats = pd.DataFrame()
  cars = pd.unique(raw_data['vehicle__id'])
  first_obs = last_obs = None 
  fmt = '%d.%m.%y'

  for cid in cars:
    cdf = raw_data[raw_data['vehicle__id'] == cid]
    car = {}
    
    # Collected period
    first, last = cdf.first_valid_index(), cdf.last_valid_index()
    start = datetime.strftime(first, format=fmt)
    end = datetime.strftime(last, format=fmt)
    first_obs = first if not first_obs or first < first_obs else first_obs
    last_obs = last if not last_obs or last > last_obs else last_obs
    
    # Number of days 
    num_days = (first - last).days * -1   # Make timedelta positive

    # Total observations 
    num_obs = len(cdf)

    # Car information
    car['Car ID'] = cid
    car['Period'] = f'{start} - {end}'
    car['Days'] = num_days
    car['Observations'] = num_obs
    car = pd.DataFrame([car], columns=cols)
    stats = stats.append(car)
  
  # Total summary
  started, ended = datetime.strftime(first_obs, format=fmt), datetime.strftime(last_obs, format=fmt)
  summary = {
    '': 'Total',
    'Car ID': len(cars),
    'Observations': stats['Observations'].sum(),
    'Days': stats['Days'].sum(),
    'Period': f'{started} - {ended}',
  }
  summary = pd.DataFrame([summary], columns=cols)
  stats = stats.append(summary)

  # Overall main statistics
  # days = data.resample('D').mean()
  # start = datetime.strftime(days.first_valid_index(), format='%d.%m.%y')
  # end = datetime.strftime(days.last_valid_index(), format='%d.%m.%y')
  # num_days = len(days)
  # num_vehicles = len(pd.unique(data['id']))
  # stats.loc[0, cols] = [f'{start} - {end}', num_vehicles, num_days, f'{len(raw_data)} ({len(data)})']
  stats.columns = [f'\\textbf{{{x}}}' for x in cols]
  stats.to_latex(f'{EXP_TABLES}/main_overview.tex', bold_rows=True, escape=False, index=False, na_rep='')



def driving_data():
  dataset = pd.read_csv(f'{ENERGY_DATA}/dataset_raw.csv', parse_dates=True, index_col='timestamp')
  one_hot = pd.get_dummies(dataset['vehicle__brand'])
  dataset = pd.concat((dataset, one_hot), axis=1)
  dataset = dataset.ffill()
  
  odometer = pd.pivot_table(dataset, index='timestamp', columns='vehicle__id', values='odometer')
  driven = odometer.resample('D').last() - odometer.resample('D').first()
  driven = driven.rolling(4).median()
  axes = driven.plot(sharey=True, subplots=True, layout=(3, 3), figsize=(8, 6), colormap=plotting.CMAP_COLD, legend=False)
  labels = dataset.reset_index().groupby(by=['vehicle__id', 'BMW', 'TESLA']).groups.keys()
  labels = list(map(lambda tpl: f'{tpl[0]}: ' + ('BMW' if tpl[1] else 'TESLA'), labels))
  
  # Axes settings
  for ax, label in zip(axes.flatten(), labels):
    ax.set_yticks(np.linspace(0, 60, 5).astype(int))
    ax.set_ylim(bottom=0, top=60)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.minorticks_on()
    ax.set_title(label, size='medium')
    ax.tick_params(which='minor', axis='x', bottom=False)

  # Figure settings
  fig = plt.gcf()
  fig.tight_layout()
  fig.text(0, 0.55, 'Driven (km)', va='center', rotation='vertical')
  return fig


def plot_distribution_battery(): 
  dataset = pd.read_csv(f'{ENERGY_DATA}/dataset_raw.csv', parse_dates=True, index_col='timestamp')
  one_hot = pd.get_dummies(dataset['vehicle__brand'])
  dataset = pd.concat((dataset, one_hot), axis=1)

  styles = { 'edgecolor': 'black', 'linewidth': 0.5 }
  battery = pd.pivot_table(dataset, index='timestamp', columns='vehicle__id', values='battery_level')
  axes = battery.plot(kind='hist', subplots=True, layout=(3, 3), colormap=plotting.CMAP_COLD, legend=False, **styles)
  labels = dataset.reset_index().groupby(by=['vehicle__id', 'BMW', 'TESLA']).groups.keys()
  labels = list(map(lambda tpl: f'{tpl[0]}: ' + ('BMW' if tpl[1] else 'TESLA'), labels))
  
  # Axes settings
  for ax, label in zip(axes.flatten(), labels):
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticks(np.linspace(0, 100, 6).astype(int))
    ax.set_yticks(np.linspace(0, ax.get_yticks().max(), 5).astype(int))
    ax.set_title(label, size='medium')
  
  # Figure settings  
  fig = plt.gcf()
  fig.tight_layout()
  fig.text(0, 0.55, 'Frequency', va='center', rotation='vertical')
  return fig 

def hourly_distribution():
    dataset = pd.read_csv(f'{ENERGY_DATA}/dataset_raw.csv', parse_dates=True, index_col='timestamp')
    one_hot = pd.get_dummies(dataset['vehicle__brand'])
    dataset = pd.concat((dataset, one_hot), axis=1)

    battery = dataset[['vehicle__id', 'battery_level']]
    labels = dataset.reset_index().groupby(by=['vehicle__id', 'BMW', 'TESLA']).groups.keys()
    labels = list(map(lambda tpl: f'{tpl[0]}: ' + ('BMW' if tpl[1] else 'TESLA'), labels))
    hourly = battery.groupby(by=['vehicle__id', battery.index.hour]).mean().unstack(0)

    axes = hourly.plot(subplots=True, layout=(3, 3), legend=False, marker='.', markersize=8, colormap=plotting.CMAP_COLD)
    x_ticks = range(0, 24, 4)
    x_labels = [f'{x:02d}:00' for x in x_ticks]

    for ax, label in zip(axes.flatten(), labels):
      ax.set_xlabel(None)
      ax.set_ylabel(None)
      ticks = ax.get_yticks()
      ax.set_title(label, size='medium')
      ax.set_yticks(np.linspace(ticks.min(), ticks.max(), 4, dtype=int))
      ax.set_xlim(left=-1, right=max(x_ticks) + 4)
      ax.set_xticks(x_ticks)
      ax.set_xticklabels(x_labels, rotation=45, size='small')
      ax.minorticks_on()

    fig = plt.gcf()
    fig.tight_layout()
    return fig

def weekly_distribution():
  dataset = pd.read_csv(f'{ENERGY_DATA}/dataset_raw.csv', parse_dates=True, index_col='timestamp')
  one_hot = pd.get_dummies(dataset['vehicle__brand'])
  dataset = pd.concat((dataset, one_hot), axis=1)

  fig, axes = plt.subplots(3, 3, sharex=True)
  fig.tight_layout(w_pad=4)
  axes = axes.flatten()
  cars = pd.unique(dataset['vehicle__id'])
  cars = sorted(cars)
  x_ticks = range(0, 24, 4)
  x_labels = [f'{x:02d}:00' for x in x_ticks]
  days = ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
  battery = dataset[['vehicle__id', 'battery_level']]
  weekly = battery.groupby(by=['vehicle__id', battery.index.weekday, battery.index.hour]).mean().unstack(2).reset_index(0)

  for cid, ax in zip(cars, axes):
    week = weekly[weekly['vehicle__id'] == cid].drop('vehicle__id', axis=1)
    cmap = ax.imshow(week, aspect='auto', interpolation='nearest', cmap=plt.get_cmap('YlGnBu', 20))
    
    is_bmw = pd.unique(dataset[dataset['vehicle__id'] == cid]['BMW']).item()
    brand = 'BMW' if is_bmw else 'TESLA'
    ax.set_title(f'{cid}: {brand}', size='small')
    ax.grid(False)
    ax.set_yticks(range(7))
    ax.set_yticklabels(days, size='small')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, size='small')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.10)
    cb = fig.colorbar(cmap, cax=cax)
    cb.set_ticks(np.linspace(week.min().min(), week.max().max(), 5, dtype=int))
    cb.ax.minorticks_on()
  return fig

if __name__ == "__main__":
  plotting.set_styles()
  # acf_fig = acf()
  # acf_fig.savefig(f'{EXPERIMENTS}/acf_battery.pdf', bbox_inches='tight', pad_inches=0)
  # driving_pattern = charging_pattern()  # Weekly and hourly
  # driving_pattern.savefig(f'{EXPERIMENTS}/driving_pattern.pdf', bbox_inches='tight', pad_inches=0)
  
  dataset_statistics()
  
  # fig = driving_data()
  # fig.savefig(f'{EXPERIMENTS}/driven_distances.pdf', bbox_inches='tight', pad_inches=0)
  # fig = plot_distribution_battery()
  # fig.savefig(f'{EXPERIMENTS}/histogram.pdf', bbox_inches='tight', pad_inches=0)
  
  # fig = weekly_distribution()
  # fig.savefig(f'{EXPERIMENTS}/weekly_distribution.pdf', bbox_inches='tight', pad_inches=0)
  # fig = hourly_distribution()
  # fig.savefig(f'{EXPERIMENTS}/hourly_distribution.pdf', bbox_inches='tight', pad_inches=0)
