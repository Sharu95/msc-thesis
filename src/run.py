import numpy as np 
np.warnings.filterwarnings('ignore')
import sys 
sys.path.append('./utils')
import os 
import time 
import utils 
import model 
import metrics
import plotting 
import history as experiment_history
import keras.backend as K
import usecases.energy as energy
import usecases.depression as depression 
import usecases.football as football
from config import Config
from sklearn.model_selection import KFold

def main(config, data):
  X, y = utils.format_data(config, data)

  print('| Creating log configurations ...')  
  config.create_log_configurations()

  print('| Saving log configurations ...')
  config.comments = f'{config.comments}, {str(X.shape)}'
  config.save_config()

  folds = config.data['folds']
  current_best_acc, current_best_mcc = 0, -2
  best_fold = None

  print(f'| Training model with {folds} folds ...')
  folder = KFold(n_splits=folds)
  exp_evaluation = experiment_history.EvaluationHistory(config=config) 
  exp_training = experiment_history.TrainingHistory(log_dir=config.log_dir)

  for k, (train_index, test_index) in enumerate(folder.split(X)):
    print(f'| X: {X.shape}\n| y: {y.shape}')
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Build model
    model_handler = model.Model(config=config)
    md = model_handler.build_model(x_train.shape)
    start, used = 0, 0

    try:
      # Train model
      start = time.time()
      history = model_handler.train_model(x_train, y_train)
      used = time.time() - start
      print(f'| K = {k + 1} | Used {used:.2f} seconds') 
    except KeyboardInterrupt as e:
      break

    # Evaluate model
    print('| Evaluating model on test set ...')
    predictions = md.predict(x_test)
    is_football = config.usecase == config.FOOTBALL
    evaluations = None 
    if is_football:
      evaluations = exp_evaluation.custom_evaluate(predictions, y_test, k + 1, data['columns'])
    else:
      evaluations = exp_evaluation.evaluate(predictions, y_test, k + 1)
    exp_training.update_history(history.history, used)
    
    # Save model
    print('| Saving model ...')
    md.save(f'{config.log_dir}/models/k{k + 1}.h5')
    
    
    # Replace current best model if ACC is better (or ACC is similar but MCC is better)
    same_acc = abs(evaluations['ACC'] - current_best_acc) < 0.05
    better_mcc = evaluations['MCC'] > current_best_mcc
    better_acc = evaluations['ACC'] > current_best_acc

    if better_acc or (same_acc and better_mcc):
      print('| Replacing best model ...')
      md.save(f'{config.log_dir}/models/best.h5')
      current_best_mcc = evaluations['MCC']
      current_best_acc = evaluations['ACC']
      best_fold = k + 1 
    
    # Clear model/session
    del md
    del model_handler
    K.clear_session()
    print('=======================')
  
  # Rename best model for best fold and save evaluations
  if os.path.exists(f'{config.log_dir}/models/best.h5'):
    os.rename(f'{config.log_dir}/models/best.h5', f'{config.log_dir}/models/best_({best_fold}).h5')
  
  if best_fold or current_best_mcc != -2:
    exp_evaluation.save_history()
    exp_evaluation.save_statistics()
    exp_training.save_history()
    exp_training.save_statistics()
    print(f'| Saved all in: {config.log_dir}')
  else:
    print('| Did not save anything')

if __name__ == '__main__':
  comments = input('=== Comments ===\n> ')
  plotting.set_styles()
  config = Config(comments)
  config.read_config()
  config.validate()
  data = None 

  if config.usecase == config.DEPRESSION:
    dp = depression.PredictDepression(data_path='../data/depression')
    data = dp.read_data(config.data['window'], config.data['resample'])
  elif config.usecase == config.FOOTBALL:
    fp = football.FootballPrediction(data_path='../data/football')
    data = fp.read_data(config.data['window'])
  else:
    ep = energy.EnergyPrediction(data_path='../data/energy')
    data = ep.read_data(config.data['window'])
  
  print('| Running {} ...'.format(config.network['architecture']))
  main(config, data)
  print('|', 40 * '-')