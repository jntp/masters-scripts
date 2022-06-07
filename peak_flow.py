import math
import pandas as pd
import numpy as np
import datetime as dt

## Auxiliary Functions

def isOvernight(start_hour, end_hour):
  is_overnight = False

  if start_hour > end_hour:
    is_overnight = True

  return is_overnight

def isMonthEnd(year, month, day):
  # Initialize variables; variables check whether month and day could mean end of month
  right_day = False
  right_month = False

  # Check if day could entail the "end of month"
  if day == 28 or day == 30 or day == 31:
    right_day = True

  # Check if the day is the end of the month for that specific month
  if day:
    # Check if the month is February
    if month == 2:
      right_month = True

    # Check order months, whether a month has 30 or 31 days gets swapped in August
    if month <= 7:
      # Jan/Mar/May/July has 31 days
      if month % 2 == 1 and day == 31:
        right_month = True 
      # April/June has 30 days
      elif month % 2 == 0 and day == 30:
        right_month = True
    elif month >= 8:
      # Aug/Oct/Dec has 31 days
      if month % 2 == 0 and day == 31:
        right_month = True
      # Sept/Nov has 30 days
      elif month % 2 == 1 and day == 30:
        right_month = True

  # If month and day correspond to end of month, they return True
  if right_day and right_month:
    # print(right_day, right_month) 
    return True
  else:
    # print(right_day, right_month) 
    return False

def check_excess_time(hour):
  # Check if the hour meets or exceeds 24, reformat the time to 24-hr format
  if hour >= 24:
    hour -= 24
  
  return hour 

## Main Functions

def convert_str_to_dt(datetime_str):
  # Create slice objects for parsing string
  year_slc = slice(0, 4)
  month_slc = slice(5, 7)
  day_slc = slice(8, 10)
  hour_slc = slice(11, 13)
  min_slc = slice(14, 16)

  # Parse string accordingly
  year = datetime_str[year_slc]
  month = datetime_str[month_slc]
  day = datetime_str[day_slc]
  hour = datetime_str[hour_slc]
  minute = datetime_str[min_slc]

  # Create datetime object
  datetime = dt.datetime(int(year), int(month), int(day), int(hour), int(minute))

  return datetime

def load_streamflow_data(file_path):
  entries = pd.read_csv(file_path)
  datetimes_str = entries.loc[:, "datetime"]
  discharges_cfs_str = entries.loc[:, "discharge_cfs"]

  # Create new lists that will hold converted datetime and discharge data
  datetimes = []
  discharges_cfs = []

  # Loop through entries and convert datetime and discharge data
  for i, datetime in enumerate(datetimes_str): 
    datetimes.append(convert_str_to_dt(datetimes_str[i]))
    discharges_cfs.append(int(discharges_cfs_str[i]))

  return datetimes, discharges_cfs

def reformat_time(start_hour, end_hour, year, month, day):
  # Check if the NCFR goes overnight; modify the time parameters if that's the case
  if isOvernight(start_hour, end_hour):
    # Check if the first day of the NCFR event is at month's end
    if isMonthEnd(year, month, day):
      # Check if the month is December
      # If that's the case, add a new year and set month/day to Jan 1
      if month == 12:
        year += 1
        month = 1
        day = 1
      else:
        # If not December, add 1 to month and set day equal 1
        month += 1
        day = 1
    else:
      # Simply add a new day
      day += 1

  return year, month, day

def create_datetimes(year_str, month_str, day_str, start_hour_str, end_hour_str):
  # Convert string to ints
  year = int(year_str)
  month = int(month_str)
  day = int(day_str) 
  start_hour = int(start_hour_str)
  end_hour = int(end_hour_str)

  # Create the first datetime object
  start_dt = dt.datetime(year, month, day, start_hour)

  # Create the end datetime object for Sepulveda and Whittier Narrows Dam (plus 2 hours)
  end_hour_SP_WN = check_excess_time(end_hour)
  reformat_time(start_hour, end_hour_SP_WN, year, month, day)

  # Create the end datetime object for Santa Ana River (plus 3 hours)
  
  # Create the end datetime object for San Diego River (plus 6 hours)

  # Will for loop work better for this???
  
  # Create the end datetime object
  end_dt = dt.datetime(year, month, day, end_hour)

  return start_dt, end_dt 

def get_peak_flow(stream_datetimes, stream_Qs, start_datetime, end_datetime):
  # Convert datetimes and streamflows to numpy array for array operations
  stream_datetimes = np.array(stream_datetimes)
  stream_Qs = np.array(stream_Qs)

  # Look for indices where the time falls within the time of interest and its respective streamflow data
  time_inds = np.where((stream_datetimes >= start_datetime) & (stream_datetimes <= end_datetime))[0]
  time_Qs = stream_Qs[time_inds]

  # Get and return the peak streamflow
  peak_flow = max(time_Qs) 

  return peak_flow 


def main():
  # Load data and look at streamflows for ze timeframe (and perhaps a few hrs more?)
  # Might want to verify how many hours... perhaps longer for SD and SA????
  ## Load streamflow data
  # Get the file paths
  sepulveda_fp = "./data/sepulveda_15min_discharge_2002_2020.csv"
  whittier_fp = "./data/whittier_15min_discharge_1995_2020.csv"
  santa_ana_fp = "./data/santa_ana_15min_discharge_1995_2020.csv"
  san_diego_fp = "./data/san_diego_15min_discharge_1995_2020.csv"

  # Load streamflow data
  sepulveda_dts, sepulveda_Qs = load_streamflow_data(sepulveda_fp)
  whittier_dts, whittier_Qs = load_streamflow_data(whittier_fp)
  santa_ana_dts, santa_ana_Qs = load_streamflow_data(santa_ana_fp)
  san_diego_dts, san_diego_Qs = load_streamflow_data(san_diego_fp)

  ## Load data from NCFR_Stats
  ncfr_fp = "./data/NCFR_Stats.csv"
  ncfr_entries = pd.read_csv(ncfr_fp)
  years = ncfr_entries.loc[:, "Year"]
  months = ncfr_entries.loc[:, "Month"]
  days = ncfr_entries.loc[:, "Day"]
  start_hours = ncfr_entries.loc[:, "Start_Hour"]
  end_hours = ncfr_entries.loc[:, "End_Hour"]
  max_refs = ncfr_entries.loc[:, "Max_Ref"]

  ## Format the NCFR_Stats data and check for peak streamflow 
  # Create empty lists to store datetimes, as well as peak streamflows
  start_dts = []
  end_dts = [] 
  peakQs = []

  # Iterate through every NCFR entry
  for i, year in enumerate(years):
    # Convert time parameters to datetimes and append to lists
    start_dt, end_dt = create_datetimes(years[i], months[i], days[i], start_hours[i], end_hours[i]) 

    start_dts.append(start_dt)
    end_dts.append(end_dt) 

    # Check for peak streamflow for all watersheds, only if max_ref has an entry
    if not math.isnan(max_refs[i]):
      # Create an empty list to store peak streamflows
      local_peakQs = []

      # Get the peak streamflows for each watershed
      sepulveda_peakQ = get_peak_flow(sepulveda_dts, sepulveda_Qs, start_dt, end_dt)
      whittier_peakQ = get_peak_flow(whittier_dts, whittier_Qs, start_dt, end_dt)
      santa_ana_peakQ = get_peak_flow(santa_ana_dts, santa_ana_Qs, start_dt, end_dt)
      san_diego_peakQ = get_peak_flow(san_diego_dts, san_diego_Qs, start_dt, end_dt)

      # Append streamflows to list
      local_peakQs.append([sepulveda_peakQ, whittier_peakQ, santa_ana_peakQ, san_diego_peakQ])

      # Get the max streamflow for all watersheds and append to the main peakQs list
      peakQs.append(max(local_peakQs))

  print(peakQs) 

if __name__ == '__main__':
  main()
