import pandas as pd
import numpy as np
import utils 
import os 

class FootballPrediction:
  def __init__(self, data_path):
    self.data_path = data_path

  
  def _create_dataset(self):
    dataset = pd.read_excel(f'{self.data_path}/football.xlsx', sheet_name=None, index_col='player/date')
    series = pd.DataFrame()
    players = {} 
    if not os.path.exists(f'{self.data_path}/players'):
      os.makedirs(f'{self.data_path}/players')
      for feature, df in dataset.items():
        ids = df.index.values
        for i in ids:
          if not i in players:
            players[i] = pd.DataFrame()
            
          pl_df = df.loc[i, :]
          if not isinstance(pl_df.index, pd.DatetimeIndex):
            empty_col = pl_df.index[-1].lower().startswith('unnamed')
            if empty_col:
              pl_df = pl_df[:-1]
          
          pl_df.index = pd.to_datetime(pl_df.index)
          pl_df.index.name = 'timestamp'
          pl_df.name = feature
          players[i] = pd.concat((players[i], pl_df), axis=1) 
          
      dataset = pd.DataFrame()
      for player, data in players.items():
        data.to_csv(f'{self.data_path}/players/player_{player}.csv')
        data['pid'] = player
        dataset = dataset.append(data)
      
      dataset = dataset[dataset.notna().all(axis=1)]
      dataset.to_csv(f'{self.data_path}/dataset.csv')
      print('| Created football dataset ...')
    else:
      dataset = pd.read_csv(f'{self.data_path}/dataset.csv', parse_dates=True, index_col='timestamp')

    return dataset, dataset['pid'].unique()


  def read_data(self, window):
    print('| Reading football data ...')
    dataset, pids = self._create_dataset()
    print(f'| Number of players: {len(pids)} ...')
    # Clean dataset
    players = dataset.groupby(by=['pid'])
    observations = players.count().mean(axis=1)
    drop_players = observations[observations <= 40].index
    players = dataset[~dataset['pid'].isin(drop_players)].groupby(by=['pid'])
    
    features = dataset.columns.drop('pid')
    features = ['Readiness', 'Stress', 'Mood', 'Soreness', 'Fatigue']
    features_out = ['Mood', 'Stress', 'Soreness', 'Fatigue']

    print(f'| Using features {str(features)} for {predictor}...')
    xs, ys = [], []

    i = 0
    for pid, player_df in players:
      # TODO: Write data generation
      # 1. Collect features
      feat_vec = player_df[features]

      # 2. Normalise features
      feat_vec_norm = utils.normalise_series(feat_vec, (0, 1))

      # 3. Create sequences
      y_dist = player_df[features_out].copy()

      input_seq = utils.create_sequences(y_dist.values, window + 1, 0)
      input_seq_norm = utils.create_sequences(feat_vec_norm.values, window + 1, 0)
      
      # NOTE: Test output
      # ys = input_seq[:, -1, 0]    # First column is Readiness
      # ys_s = input_seq[:, -1, 2]  # Third columns is Stress

      if i == 0:
        print(y_dist)

      seq_x = input_seq_norm[:, :-1, :]
      seq_y = input_seq[:, -1, :]
      i += 1
      # print(pid)
      # sh = 0
      # print(feat_vec[:][sh:40+sh+1])
      # print(input_seq[sh, :-1])
      # print(seq_y[sh])
      # # print(input_seq)
      # # print(ys)
      # exit()

      # Multi-label classification
      # seq_y = input_seq[:, -1, list(range(1, 5))]

      # print(seq_x[sh])
      # print(seq_y[sh])
      xs.extend(seq_x)
      ys.extend(seq_y)
      # exit()

    # 4. Return Xs and ys to be formatted in the classification formatter
    xs, ys = np.array(xs), np.array(ys)

    return {
      'X': xs,
      'y': ys,
      'columns': features_out,
      'classes': len(features_out)
    } 