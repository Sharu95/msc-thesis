import json
from datetime import datetime 
import os 


class Config():
  def __init__(self, comments):
    self.data = {}
    self.network = {}
    self.DEPRESSION = 'depression'
    self.FOOTBALL = 'football'
    self.EV = 'ev'
    self.usecases = [self.DEPRESSION, self.EV, self.FOOTBALL]
    self.usecase = None
    self.log_dir = None
    self.comments = comments
    self.experiment_folder = ''
    self.logs = {
      'lstm': '../logs/LSTM',
      'cnn': '../logs/CNN'
    }

  def read_config(self):
    with open('./config.json') as f:
      config = json.load(f)
      self.data = config['data']
      self.network = config['network']
      self.usecase = config['_usecase']
      self.experiment_folder = config['_experiment_folder']
    return config

  def validate(self):
    if not self.usecase in self.usecases:
      print('| Invalid usecase')
      exit()
    else:
      if not self.network['architecture'] in self.logs.keys():
        print('| Invalid architecture')
        exit()

      # Validate settings for each usecase 
      if self.usecase == self.DEPRESSION:
        self.data['classes'] = 2
      elif self.usecase == self.EV:
        self.data['classes'] = 10
      elif self.usecase == self.FOOTBALL:
        self.data['classes'] = 4
      else:
        print('| Invalid usecase ...')

  def create_log_configurations(self):
    arch = self.network['architecture']
    base_path = f'{self.logs[arch]}/{self.experiment_folder}' 

    # Folder configuration and paths
    timestamp = datetime.now().strftime('%y%m%d_%H:%M:%S')
    log_dir = f'{base_path}/{self.usecase}/{timestamp}'

    # Create folders
    folders = [log_dir]
    sub_folders = [f'{log_dir}/{sub_folder}' for sub_folder in ['models', 'cm', 'stats']]
    folders.extend(sub_folders)

    for folder in folders:
      if not os.path.exists(folder):
        os.makedirs(folder)

    self.log_dir = log_dir

  def save_config(self):
    with open(f'{self.log_dir}/config.json', 'w+') as f:
      cfg = {
        'data': self.data,
        'network': self.network,
        '_usecase': self.usecase,
        '_comments': self.comments,
        '_experiment_folder': self.experiment_folder
      }
      json.dump(cfg, f, sort_keys=True, indent='\t', separators=(',', ': '))

if __name__ == "__main__":
  print('| Run run.py')
