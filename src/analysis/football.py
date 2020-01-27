import matplotlib.pyplot as plt
import pandas as pd
import sys 
sys.path.append('../utils/')
import plotting
import numpy as np 
import utils 
from mpl_toolkits.axes_grid1 import make_axes_locatable


DATA = '../../data/football'
EXPERIMENTS = '../../../thesis/4_experiments/football/figures'
EXP_TABLES = '../../../thesis/4_experiments/football/tables'

# Dataset statistics
## Number of players
## Days with valid observations
def dataset_statistics(df, raw_df):
  
  df_stats = {}
  for d, tag in zip([df, raw_df], ['df', 'raw']):
    players = d.groupby(by=['pid'])
    stats = players.describe()

    # Dataset information
    features = stats.agg('sum')[:, 'count']
    features['Players'] = len(stats.index)
    # features['Players (raw)'] = len(raw_df['pid'].unique())
    features = features.astype(int).reset_index()
    df_stats[tag] = features
  
  cleaned = df_stats['df'].values 
  raw = df_stats['raw'].values
  features = dict((ft, f'{cl} ({rw})') for (ft, cl), (_, rw) in zip(cleaned, raw))
  features = pd.DataFrame([features]).T.reset_index()
  features.columns = ['Description', 'Count (raw)']
  features = features.reindex(features['Count (raw)'].str.len().sort_values().index)
  features.columns = [f'\\textbf{{{col}}}' for col in features.columns]
  features.to_latex(f'{EXP_TABLES}/overview.tex', bold_rows=True, index=False, escape=False)


# Correlation between variables
def avg_corr_matrix(df, annotate=False):
  corr = df.mean().corr()
  fig, axes = plt.subplots(1, 1)
  ax1 = axes
  cm = plt.get_cmap('YlGnBu', 20)
  cmap = ax1.imshow(corr, interpolation='nearest', cmap=cm)
  ax1.grid(False)
  cols = corr.columns.values
  ticks = np.linspace(0, len(cols) - 1, len(cols))
  ax1.set_xticks(ticks)
  ax1.set_xticklabels(cols, rotation=90)
  ax1.set_yticks(ticks)
  ax1.set_yticklabels(cols)
  #cb = ax.colorbar()
  #cb.ax.minorticks_on()
  cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.10)
  cb = fig.colorbar(cmap, cax=cax)
  cb.ax.minorticks_on()
  
  if annotate:
    for i in range(len(cols)):
      for j in range(len(cols)):
        c = round(corr.iloc[i, j], 2)
        text = ax1.text(i, j, c, size=9, ha="center", 
                        va="center", color='black' if c < 0.5 else 'white')
  return fig


# Distribution for players (readiness)
def distribution(df, feature):
  pvt = pd.pivot_table(df, index='timestamp', columns='pid', values=feature)
  other_scale = feature == 'Readiness' or feature == 'SleepDuration'
  styles = { 'edgecolor': 'black', 'linewidth': 0.5 }
  axes = pvt.plot(kind='hist', subplots=True, figsize= (9.5, 6), #(13, 9), 
          layout=(5, 7), colormap=plotting.CMAP_HOT, legend=False, 
          bins=10 if other_scale else 5, **styles)
  for ax in axes.flatten():
    ax.minorticks_on()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(np.linspace(0, ax.get_yticks().max(), 4, dtype=int))
    ax.set_xticks(range(0, 11, 2))  #if other_scale else range(0, 6, 3))

  fig = plt.gcf()
  fig.tight_layout(pad=0.5)
  fig.text(-0.025, 0.55, 'Frequency', va='center', rotation='vertical', size='x-large')
  fig.text(0.48, 0.15, 'Readiness', va='center', rotation='horizontal', size='x-large')

  return fig 

# Cross-correlation?
# NOTE: Maybe remove
def avg_cross_corr(players, lags, pred):
  players_time = players.groupby(['pid', 'timestamp'])
  avg_df = players_time.mean().unstack(0).stack(0).mean(axis=1).unstack(1)
  predictor = utils.normalise_series(avg_df[pred], (0, 1))
  ccf = pd.DataFrame() 

  # Compute auto-correlation
  for feature in avg_df.columns:
    if feature != pred:
      feat = utils.normalise_series(avg_df[feature], (0, 1))
      ccf[feature] = [predictor.corr(feat.shift(lag)) for lag in range(lags + 1)]

  # axes = ccf.plot(subplots=True, marker='.', ms=9, layout=(3, 4), 
  #                 colormap=plotting.CMAP_HOT, figsize=(14, 6))
  axes = ccf.plot(kind='bar', subplots=True, layout=(3, 4), 
                  colormap=plotting.CMAP_HOT, figsize=(12, 7))
  
  for ax in axes.flatten():
    ax.minorticks_on()
    ax.set_title('')
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1.5)
    ax.set_ylim(-0.2, 0.2)
    yticks = np.linspace(-0.2, 0.2, 5)
    ax.set_yticks(np.round(yticks, 2))

    xargs = np.linspace(0, lags, 4, dtype=int)
    ax.set_xticks(xargs)
    xlabels = np.arange(0, lags + 1, dtype=int)[xargs]
    ax.set_xticklabels(xlabels, rotation=0)

  plt.tight_layout()
  return plt.gcf() 

def avg_acf(df, days):
  corr = pd.DataFrame()
  labels = df.columns.values
  for feature in df.columns:
    corr[feature] = [df[feature].autocorr(lag) for lag in range(1, days)]
  
  axes = corr.plot(kind='line', colormap=plotting.CMAP_HOT, figsize=(13, 6), 
                  title=False, subplots=True, layout=(2, 4),  
                  marker='.', ms=9, legend=False)
  #axes = corr.plot(kind='bar', figsize=(13, 6), title='ACF', subplots=True, layout=(2, 4),  legend=False)

  for ax, label in zip(axes.flatten(), labels):
    ax.minorticks_on()
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1.5)
    yticks = np.linspace(ax.get_yticks().min(), ax.get_yticks().max(), 5)
    yticks = np.round(yticks, 2)
    ax.set_yticks(yticks)
    ax.set_title(label, size='x-large')
    ax.set_xticks(np.linspace(0, days, 6))
    #ax.set_yticks(np.linspace(-0.10, 0.60, 4))
    ax.set_yticks(np.linspace(ax.get_yticks().min(), ax.get_yticks().max(), 6))   
  
  fig = plt.gcf()
  fig.tight_layout()
  fig.text(-0.025, 0.55, 'Correlation coefficient', va='center', rotation='vertical', size='x-large')
  fig.text(0.48, -0.05, r'Lag $n$', va='center', rotation='horizontal', size='x-large')
  
  return fig



if __name__ == "__main__":
  plotting.set_styles()

  # Make sure dataset is generated
  dataset = pd.read_csv(f'{DATA}/dataset.csv', parse_dates=True, index_col='timestamp')
  dataset = dataset.drop(['Session-RPE'], axis=1)
  players = dataset.groupby(by=['pid'])
  observations = players.count().mean(axis=1)
  drop_players = observations[observations <= 30].index
  players = dataset[~dataset['pid'].isin(drop_players)]
  players_gr = players.groupby(by=['pid'])

  # NOTE: Statistics  
  # dataset_statistics(players, dataset)
  
  # NOTE: Corr-matrix
  # fig = avg_corr_matrix(players_gr, annotate=True)
  # fig.savefig(f'{EXPERIMENTS}/avg_corr_matrix.pdf', bbox_inches='tight', pad_inches=0)

  # NOTE: Distribution
  # ft = 'Readiness'
  # fig = distribution(players, ft)
  # fig.savefig(f'{EXPERIMENTS}/histogram_{str(ft.lower())}.pdf', bbox_inches='tight', pad_inches=0)

  # NOTE: CCF
  # for ft in ['Readiness', 'Stress']:
  #   fig = avg_cross_corr(players, lags=60, pred=ft)
  #   fig.savefig(f'{EXPERIMENTS}/ccf_{ft.lower()}.pdf', bbox_inches='tight', pad_inches=0)

  # NOTE: ACF
  # players_time = players.groupby(['pid', 'timestamp'])
  # avg_df = players_time.mean().unstack(0).stack(0).mean(axis=1).unstack(1)
  # fig = avg_acf(avg_df, days=40)
  # fig.savefig(f'{EXPERIMENTS}/acf_variables.pdf', bbox_inches='tight', pad_inches=0)
