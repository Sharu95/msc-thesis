import utils 
import numpy as np
from keras.utils import to_categorical

def create_sequences(delta_values, window, future_window):
  window_frame = window + future_window
  data_points = delta_values.shape[0]
  driving_window = lambda i, seq: seq[i:i + window_frame]
  data = [driving_window(i, delta_values) for i in range(data_points - (window_frame + 1))]

  return np.array(data)


def format_data(config, data):
  num_classes = config.data['classes']
  xs, ys = data['X'], data['y']

  if config.usecase == config.DEPRESSION:
    xs = xs.reshape((xs.shape[0], xs.shape[1], xs.shape[2]))
    ys = to_categorical(ys, num_classes=2)
  elif config.usecase == config.EV:
    classes = np.linspace(0.1, 1.0, num_classes)

    # For each y_value, get the 'closest' class.
    class_map = [min(classes, key=lambda x: abs(x - y)) for y in ys]

    # Get index of the class
    args = [np.argwhere(classes == class_i).flatten() for class_i in class_map]

    # Encode using the indices
    ys = to_categorical(args, num_classes=num_classes)

  else:

    # Multi-label classification
    # ys.shape = [N, OUTPUT_LABELS (same scale of 5)]
    # ys[ys < 3] = 0
    # ys[ys > 3] = 1
    # print(ys)
    # exit()


    columns = data['columns']
    for i, _ in enumerate(columns):
      ys[:, i] = np.where(ys[:, i] < config.data['score_threshold'], 0, 1)
    # exit()
    # ys = ys 


    # exit()
    print('TODO: FIX STRESS CASE FORMATTING')
    # Encode using class indices (y - 1)
    # classes for readiness: low=1-3, mid=4-7, high=8-10
    # not_ready = ys < 4
    # maybe_ready = np.logical_and(ys >= 4, ys <= 7)
    # is_ready = ys > 7
    # ys[not_ready] = 0
    # ys[maybe_ready] = 1
    # ys[is_ready] = 2
    # ys = to_categorical(ys, num_classes=num_classes)

  return xs, ys


def normalise_series(series, feature_range):

  normalised = (series - series.min()) / (series.max() - series.min())
  scaled = (normalised * (max(feature_range) - min(feature_range))) + min(feature_range)
  return scaled