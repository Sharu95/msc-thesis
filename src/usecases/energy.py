import utils 
import sys 
import pandas as pd
import numpy as np 
import random

# sys.path.append('../../utils/')

class EnergyPrediction:
  def __init__(self, data_path):
    self.data_path = data_path
    self.data = None
    self.test_data = None
    self.seqs = None 
    self.test_seqs = None
    self.car_brands = None

  def _distances_from_point(self, lat_1, lon_1, lats, lons):
  
    # Equatorial radius in km
    earth_radius = 6371 

    # Radians 
    lat_1_rad = np.radians(lat_1)
    lat_2_rad = np.radians(lats)

    # From home 
    from_home_lat = np.radians(lats - lat_1)
    from_home_lon = np.radians(lons - lon_1)

    # Haversine formula
    a = np.sin(from_home_lat/2) * np.sin(from_home_lat/2) + \
        np.cos(lat_1_rad) * np.cos(lat_2_rad) * \
        np.sin(from_home_lon/2) * np.sin(from_home_lon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = earth_radius * c
    return distance
    


  def _charge_probability(self, cdf):

    # Find most frequent location for car
    locations = cdf.groupby(['lon', 'lat'])
    location_freq = locations.size().to_frame().reset_index()
    most_freq = np.argmax(location_freq[0].values)
    lon_1, lat_1 = location_freq[['lon', 'lat']].iloc[most_freq, :].values

    # Distances from most frequent point in km
    lons, lats = cdf['lon'].values, cdf['lat'].values
    center_distances = self._distances_from_point(lat_1, lon_1, lats, lons)
    return center_distances


  def drop_invalid_observations(self, df, not_driven_days=5):
      rolling_window = df.rolling(window=not_driven_days)
      
      # Moving window detects "still"-standing cars. 
      # Driven and battery used to 0 indicates not used car within period
      bat_used = rolling_window['bat_used'].sum() != 0
      odo_driven = rolling_window['driven'].sum() != 0
      
      # Only include observations where battery used and car driven (jointly)
      battery_used = df['bat_used'] < 0
      odometer_used = df['odometer'] > 0
      car_is_used = battery_used & odometer_used
      
      return df[car_is_used]
      
  def clean_data(self, df):
    print(f'| Cleaning EV dataset ... (observations: {len(df)})')
    # Drop observations for car 12 (which has outliers in odometer)
    df = df[~(df['vehicle__id'].isin([12]))]
    
    # Drop observations where odometer is 0 (invalid) 
    df = df[~(df['odometer'] == 0)]
    
    # Drop charging_state and est_battery_range
    df = df.drop(['charging_state', 'est_battery_range'], axis=1)
    
    # Numerical encoding if charging_state is used.
    # Valid states: ['Disconnected', 'Charging', 'Complete', 'Stopped', 'NoPower', 'Error', 'Starting']
    encode_state = lambda row: 1 if (row == 'Charging') or (row == 'Starting') else 0
    
    # Car IDS and dataset initialisation
    cars  = pd.unique(df['vehicle__id'])
    dataset = pd.DataFrame()

    # Dataframes for cross-checking 
    sorted_df = pd.DataFrame()
    hourly_df = pd.DataFrame()
    interpolated_df = pd.DataFrame() 

    # First day with approximately regular collection
    start_period = '2018-11-12'
    
    for cid in cars:
      cdf = df[df['vehicle__id'] == cid]
      cdf = cdf.loc[start_period:, :]
      sorted_df = sorted_df.append(cdf)

      itp = cdf.resample('H').mean().interpolate()
      itp['id'] = cid
      interpolated_df = interpolated_df.append(itp)
      
      # Compute used battery on hourly frequency
      hourly = cdf.resample('H').mean() 
      hourly_used = hourly['battery_level'].diff()
      hourly['used'] = hourly_used
      hourly_df = hourly_df.append(hourly)
      
      # Create series for daily resampling
      period = 'D'
      cleaned_df = pd.DataFrame() 
      
      # Battery features
      ####################################### TEST APPROACH 
      # bat = itp['battery_level'].diff() 
      # sample = itp['battery_level'].resample(period)

      # charged = bat.where(bat > 0, 0) #.to_frame(name='charged')
      # used = bat.where(bat < 0, 0) #.to_frame(name='used')
      # driven = itp['odometer'].diff().ffill().bfill() #.to_frame(name='driven')
      # # cleaned_df = pd.concat((used, charged, driven), axis=1)
      # # cleaned_df = cleaned_df.resample('D').sum()

      # cleaned_df['bat_used'] = used.resample(period).sum() #hourly_used.where(hourly_used < 0, 0).resample(period).sum()
      # cleaned_df['bat_charged'] = charged.resample(period).sum() # hourly_used.where(hourly_used > 0, 0).resample(period).sum()
      # cleaned_df['bat_avg'] = sample.mean()
      # cleaned_df['bat_max'] = sample.max()
      # cleaned_df['bat_min'] = sample.min()
      # cleaned_df['bat_med'] = sample.median()
      # cleaned_df['bat_std'] = sample.std()
      ####################################### END TEST APPROACH


      bat = cdf['battery_level']
      sample = bat.resample(period)
      cleaned_df['bat_used'] = hourly_used.where(hourly_used < 0, 0).resample(period).sum()
      cleaned_df['bat_charged'] = hourly_used.where(hourly_used > 0, 0).resample(period).sum()
      cleaned_df['bat_avg'] = sample.mean()
      cleaned_df['bat_max'] = sample.max()
      cleaned_df['bat_min'] = sample.min()
      cleaned_df['bat_med'] = sample.median()
      cleaned_df['bat_std'] = sample.std()
      
      # Location features
      # cleaned_df['lat'] = cdf['latitude'].round(4).resample(period).median()
      # cleaned_df['lon'] = cdf['longitude'].round(4).resample(period).median()
      
      # Odometer features
      first, last = cdf['odometer'].resample(period).first(), cdf['odometer'].resample(period).last()
      cleaned_df['odometer'] = first
      cleaned_df['driven'] = last - first
      
      # Temperature features
      # cleaned_df['tmp_out'] = cdf['outside_temp'].resample(period).mean()
      
      # Drop observations where:
      # ** Car is not driven or standing still for a given period
      # ** Battery is not used, but distance is driven
      # is_used = cleaned_df['bat_used'] < 0
      # is_driven = cleaned_df['driven'] > 0
      # cleaned_df = cleaned_df[is_used & is_driven]
      # cleaned_df = self.drop_invalid_observations(cleaned_df, not_driven_days=5) #.ffill()    
      cleaned_df = cleaned_df.resample(period).mean().ffill()

      # Meta information
      cleaned_df['id'] = cid
      # cleaned_df['brand'] = pd.unique(cdf['vehicle__brand']).item()
      dataset = dataset.append(cleaned_df)

    # Rename, round decimals and save
    dataset = dataset.round(4)
    # one_hot = pd.get_dummies(dataset['brand'])
    # self.car_brands = one_hot.columns.values
    # dataset = pd.concat((dataset, one_hot), axis=1)
    dataset.to_csv(f'{self.data_path}/dataset.csv')
    sorted_df.to_csv(f'{self.data_path}/dataset_sorted.csv')
    hourly_df.to_csv(f'{self.data_path}/dataset_hourly.csv')
    interpolated_df.to_csv(f'{self.data_path}/dataset_interpolated.csv')
    print(f'| Cleaned ... (observations: {len(dataset)})')


    return dataset



  def read_data(self, window):
    print('| Reading EV data ...')
    df = pd.read_csv(f'{self.data_path}/dataset_raw.csv', index_col='timestamp', parse_dates=True)
    df = self.clean_data(df)

    seqs = []
    feature_df = pd.DataFrame()
    r_window = 3
    
    # Shuffle dataset, generate features for each car and aggregate
    cars  = pd.unique(df['id'])
    random.shuffle(cars)
    for cid in cars:
      cdf = df[df['id'] == cid]
      cdf_features = pd.DataFrame()

      # Battery features, normalised to 0-1
      battery_features = ['bat_used', 'bat_avg', 'bat_std'] #['bat_used', 'bat_charged', 'bat_avg', 'bat_std']
      # battery_features = ['bat_used']
      cdf_features[battery_features] = (cdf[battery_features] / 100)
      # cdf_features[battery_features] = (cdf[battery_features] / 100).rolling(window=r_window, min_periods=1).mean()

      # cdf_features['driven'] = cdf['driven']
      # cdf_features['driven'] = cdf['driven'].rolling(window=r_window, min_periods=1).median() 
      cdf_features['driven'] = utils.normalise_series(cdf['driven'], (0, 1))
      
      # cdf_features['temperature'] = utils.normalise_series(cdf['tmp_out'], (0, 1))
      
      # Features: distances from most frequent location. Scaled relative to 1km range
      # distances = self._charge_probability(cdf)
      # cdf_features['distances'] = 1 / (distances + 1) #(distances - distances.min()) / (distances.max() - distances.min())
      
      # Brand vector 
      # cdf_features = pd.concat((cdf_features, cdf[self.car_brands]), axis=1) 
      # feature_df = feature_df.append(cdf_features)

      dummy_df = cdf_features.copy()
      dummy_df['id'] = cid
      feature_df = feature_df.append(dummy_df)

      # Create sequences
      input_seq = utils.create_sequences(cdf_features.values, window, 1)
      seqs.extend(input_seq)

    # Save feature df, for the sake of later analysis/cross-checking
    feature_df = feature_df.round(4)
    feature_df.to_csv(f'{self.data_path}/dataset_features.csv')

    # Drop used and charged features, first two columns
    seqs = np.array(seqs)
    
    # With rolling and used first column is "used", and is dropped as feature
    # Otherwise, rolling is first, where Y=mean of rolling
    xs = seqs[:, :window, :]

    # Used feature is first column. Get used t + n
    ys = np.abs(seqs[:, window:, 0]) 
    # exit()
    return {
      'X': xs,
      'y': ys
    }