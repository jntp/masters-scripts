import pandas as pd
import numpy as np
import datetime as dt
import math


def convert_str_to_dt(datetime_str, time_str, code = 0):
  # Code 0 works with datetimes in format YYYY-MM-DD or YYYY/MM/DD (default)
  # Code 1 works with datetimes in format MM/DD/YYYY 
  # Create slice objects for parsing datetime string
  if code == 0: # YYYY-MM-DD
    year_slc = slice(0, 4)
    month_slc = slice(5, 7)
    day_slc = slice(8, 10)
  elif code == 1: # MM/DD/YYYY 
    month_slc = slice(0, 2)
    day_slc = slice(3, 5)
    year_slc = slice(6, 10)

  # Create slice objects for parsing time str
  hour_slc = slice(0, 2)
  minute_slc = slice(3, 5)

  # Parse string accordingly
  year = datetime_str[year_slc]
  month = datetime_str[month_slc]
  day = datetime_str[day_slc]
  hour = time_str[hour_slc]
  minute = time_str[minute_slc]

  # Create datetime object
  datetime = dt.datetime(int(year), int(month), int(day), int(hour), int(minute))

  return datetime

def load_data(file_path, date_str, time_str, parameter_str, code = 0):
  entries = pd.read_csv(file_path)
  datetimes_str = entries.loc[:, date_str]
  times_str = entries.loc[:, time_str]
  parameters_str = entries.loc[:, parameter_str]
  
  # Create new lists that will hold converted datetime and parameter data
  datetimes = []
  times = []
  parameters = []

  # Loop through entries and convert datetime and precipitation data
  for i, datetime in enumerate(datetimes_str):  
    datetimes.append(convert_str_to_dt(datetimes_str[i], times_str[i], code))
  
    parameters.append(float(parameters_str[i]))
                                                                  
  return datetimes, parameters

def get_mean_data(stream_datetimes, stream_data, ncfr_dt1, ncfr_dt2):
  # Convert datetimes and data to numpy array for array operations                      
  stream_datetimes = np.array(stream_datetimes)
  stream_data = np.array(stream_data)
    
  # Look for indices where the ncfr time matches the streamflow time and its respective data
  time_ind1 = np.where(stream_datetimes == ncfr_dt1)[0][0]
  time_ind2 = np.where(stream_datetimes == ncfr_dt2)[0][0]
  time_datum1 = stream_data[time_ind1]
  time_datum2 = stream_data[time_ind2]
    
  # Find the mean of the two mean data points
  mean_data = (time_datum1 + time_datum2) / 2

  return mean_data


def main():
  ## Load the hourly RAWS data
  # Get the file paths
  cheeseboro_SP_fp = "./data/Cheeseboro_sample_Nov_2023.csv"

  datetime = dt.datetime(2023, 11, 26, 18, 30)
  print(datetime) # delete later

  # Load hourly precipitation data
  cheeseboro_SP_dts, cheeseboro_SP_prcp = load_data(cheeseboro_SP_fp, "Date", "Time", "Precip_hr_mm")
  print(cheeseboro_SP_dts, cheeseboro_SP_prcp)

  ## Load data from NCFR_Stats2
  ncfr_fp = "./data/NCFR_Stats.csv" # perhaps rename to NCFR_Stats2; see entries below to see the default
  ncfr_entries = pd.read_csv(ncfr_fp)
  years = ncfr_entries.loc[:, "Year"]
  months = ncfr_entries.loc[:, "Month"]
  days = ncfr_entries.loc[:, "Day"]
  start_hours = ncfr_entries.loc[:, "Start_Hour"]
  end_hours = ncfr_entries.loc[:, "End_Hour"]
  max_refs = ncfr_entries.loc[:, "Max_Ref"]
  peak_watersheds = ncfr_entries.loc[:, "peak_watershed"]

  ## Format the NCFR_Stats2 data and check for mean discharge and daily precipitation
  # Create list of drainage areas of watersheds [Sepulveda, Whittier, Santa Ana, San Diego]
  drainage_areas = [455060911, 327892495, 6363859786, 1116284876]

  # Create empty lists to store new data to be saved in dataframe
  mean_Qs_SP = []
  mean_Qs_WN = []
  mean_Qs_SA = []
  mean_Qs_SD = []
  mean_prcps_SP = []
  mean_prcps_WN = []
  mean_prcps_SA = []
  mean_prcps_SD = []
  runoffs_SP = []
  runoffs_WN = []
  runoffs_SA = []
  runoffs_SD = []
  run_ratios_SP = []
  run_ratios_WN = []
  run_ratios_SA = []
  run_ratios_SD = []

  # Iterate through every NCFR entry
  for i, year in enumerate(years):
    # Check for mean discharge and daily precip, only if max_refs has an entry
    if not math.isnan(max_refs[i]):
      ncfr_dt = dt.datetime(int(years[i]), int(months[i]), int(days[i]), int(start_hours[i]))
      ncfr_dt2 = dt.datetime(int(years[i]), int(months[i]), int(days[i]), int(end_hours[i]))
      # Left off here




 


  

if __name__ == '__main__':
  main()
