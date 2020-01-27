import pandas as pd 
import numpy as np 
import os 
import utils 
import random 

class PredictDepression:
  def __init__(self, data_path):
    self.data_path = data_path

  def read_data(self, window, resample):
    print('| Reading depression data ...')
    conditions = os.listdir(f'{self.data_path}/condition')
    controls = os.listdir(f'{self.data_path}/control')
    # random.shuffle(conditions)
    # random.shuffle(controls)
    files = conditions + controls
    random.shuffle(files)
    xs, ys = [], []

    for i, filename in enumerate(files):
      data_class = 'condition' if 'condition' in filename else 'control'
      raw_df = pd.read_csv(f'{self.data_path}/{data_class}/{filename}', index_col='timestamp', parse_dates=True)
      raw_df = raw_df.drop(columns=['date'])
      raw_df = raw_df.rename(columns={'activity': f'{data_class}_{i + 1}'}) 

      df = raw_df
      total = df.resample(resample).sum()
      if resample:
        df = df.resample(resample).mean()

      # Extract features and normalise
      # df = (df - df.min()) / (df.max() - df.min())
      mean, deviation = df.values.mean(), df.values.std()
      mn, mx = df.min(), df.max()
      # df['rate'] = df / mean
      # df['maxrate'] = df.iloc[:, 0] / mx
      # df['max'] = mx
      # df['min'] = mn
      df['mean'] = mean
      df['deviation'] = deviation
      # df['total'] = (total - total.min()) / (total.max() - total.min())
      # print(df)
      # exit()

      seq = utils.create_sequences(df.values, window, 0)
      xs.extend(seq)
      data_class = 1 if data_class == 'condition' else 0
      ys.extend(np.full(seq.shape[0], data_class))

    xs, ys = np.array(xs), np.array(ys)
    
    return {
      'X': xs,
      'y': ys
    }