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

def append_streamflows(local_peakQs, streamflow_list):
  for streamflow in streamflow_list:
    local_peakQs.append(streamflow)

  return local_peakQs

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
  end_hour += 2
  end_hour_SP_WN = check_excess_time(end_hour)
  year, month, day = reformat_time(start_hour, end_hour_SP_WN, year, month, day)
  end_dt_SP_WN = dt.datetime(year, month, day, end_hour_SP_WN)

  # Create the end datetime object for Santa Ana River (plus 3 hours)
  end_hour += 1
  end_hour_SA = check_excess_time(end_hour)
  year, month, day = reformat_time(start_hour, end_hour_SA, year, month, day)
  end_dt_SA = dt.datetime(year, month, day, end_hour_SA)

  # Create the end datetime object for San Diego River (plus 6 hours)
  end_hour += 3
  end_hour_SD = check_excess_time(end_hour)
  year, month, day = reformat_time(start_hour, end_hour_SD, year, month, day)
  end_dt_SD = dt.datetime(year, month, day, end_hour_SD)

  return start_dt, end_dt_SP_WN, end_dt_SA, end_dt_SD

def get_peak_flow(stream_datetimes, stream_Qs, start_datetime, end_datetime):
  # Convert datetimes and streamflows to numpy array for array operations
  stream_datetimes = np.array(stream_datetimes)
  stream_Qs = np.array(stream_Qs)

  # Look for indices where the time falls within the time of interest and its respective streamflow data
  time_inds = np.where((stream_datetimes >= start_datetime) & (stream_datetimes <= end_datetime))[0]
  time_Qs = stream_Qs[time_inds]

  # Get and return the peak streamflow
  try:
    peak_flow = max(time_Qs) 
  except:
    peak_flow = 0

  return peak_flow

def get_peak_watershed(local_peakQs):
  # Initialize variable that will return which watershed reported peak flow 
  peak_watershed = "" 

  # Find the maximum streamflow for all watersheds 
  max_local_peakQs = max(local_peakQs)
  
  # Get the index of max local streamflow
  max_ind = np.where(local_peakQs == max_local_peakQs)[0]
  
  # Get the watershed that has the peak streamflow
  if len(local_peakQs) == 4: # test for all 4 watersheds
    if max_ind == 0:
      peak_watershed = 'SP' # Sepulveda Dam
    elif max_ind == 1:
      peak_watershed = 'WN' # Whittier Narrows Dam
    elif max_ind == 2:
      peak_watershed = 'SA' # Santa Ana River
    elif max_ind == 3:
      peak_watershed = 'SD' # San Diego River
  elif len(local_peakQs) == 3: # test for all but Sepulveda Dam
    if max_ind == 0:
      peak_watershed = 'WN' # Whittier Narrows Dam
    elif max_ind == 1:
      peak_watershed = 'SA' # Santa Ana River
    elif max_ind == 2:
      peak_watershed = 'SD' # San Diego River

  return max_local_peakQs, peak_watershed

def append_regional_streamflows(list1, list2, list3, list4, Q1, Q2, Q3, Q4):
  list1.append(Q1)
  list2.append(Q2)
  list3.append(Q3)
  list4.append(Q4)

  return list1, list2, list3, list4

def main():
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
  end_dts_SP_WN = []
  end_dts_SA = []
  end_dts_SD = [] 
  peakQs_SP = []
  peakQs_WN = []
  peakQs_SA = []
  peakQs_SD = []
  peak_watersheds = [] 

  # Iterate through every NCFR entry
  for i, year in enumerate(years):
    # Convert time parameters to datetimes and append to lists
    start_dt, end_dt_SP_WN, end_dt_SA, end_dt_SD = create_datetimes(years[i], months[i], days[i], \
        start_hours[i], end_hours[i])

    start_dts.append(start_dt)
    end_dts_SP_WN.append(end_dt_SP_WN)
    end_dts_SA.append(end_dt_SA)
    end_dts_SD.append(end_dt_SD)

    # Check for peak streamflow for all watersheds, only if max_ref has an entry
    if not math.isnan(max_refs[i]):
      print(years[i], months[i], days[i], start_hours[i], end_hours[i])
      # Create an empty list to store peak streamflows
      local_peakQs = []

      # Get the peak streamflows for each watershed and append streamflows to list
      # Sepulveda streamflow data only available 2002 and later, so don't run for years before that
      if year < 2002:
        whittier_peakQ = get_peak_flow(whittier_dts, whittier_Qs, start_dt, end_dt_SP_WN)
        santa_ana_peakQ = get_peak_flow(santa_ana_dts, santa_ana_Qs, start_dt, end_dt_SA)
        san_diego_peakQ = get_peak_flow(san_diego_dts, san_diego_Qs, start_dt, end_dt_SD)

        # Append streamflows to list
        local_peakQs = append_streamflows(local_peakQs, [whittier_peakQ, santa_ana_peakQ, san_diego_peakQ])

        # Append streamflows to site-specific lists
        peakQs_SP, peakQs_WN, peakQs_SA, peakQs_SD = append_regional_streamflows(peakQs_SP, peakQs_WN, \
            peakQs_SA, peakQs_SD, np.nan, whittier_peakQ, santa_ana_peakQ, san_diego_peakQ)
      elif year >= 2002:
        sepulveda_peakQ = get_peak_flow(sepulveda_dts, sepulveda_Qs, start_dt, end_dt_SP_WN)
        whittier_peakQ = get_peak_flow(whittier_dts, whittier_Qs, start_dt, end_dt_SP_WN)
        santa_ana_peakQ = get_peak_flow(santa_ana_dts, santa_ana_Qs, start_dt, end_dt_SA)
        san_diego_peakQ = get_peak_flow(san_diego_dts, san_diego_Qs, start_dt, end_dt_SD)

        # Append streamflows to list
        local_peakQs = append_streamflows(local_peakQs, [sepulveda_peakQ, whittier_peakQ, santa_ana_peakQ, \
            san_diego_peakQ]) 

        # Append streamflows to site-specific lists
        peakQs_SP, peakQs_WN, peakQs_SA, peakQs_SD = append_regional_streamflows(peakQs_SP, peakQs_WN, \
            peakQs_SA, peakQs_SD, sepulveda_peakQ, whittier_peakQ, santa_ana_peakQ, san_diego_peakQ)

      # Get the max streamflow for all watersheds and append to the main peakQs list
      max_local_peakQs, peak_watershed = get_peak_watershed(local_peakQs) 
      peak_watersheds.append(peak_watershed)
    else:
      # Append NaN and "" to an entry with no max_ref
      peakQs_SP, peakQs_WN, peakQs_SA, peakQs_SD = append_regional_streamflows(peakQs_SP, peakQs_WN, \
          peakQs_SA, peakQs_SD, np.nan, np.nan, np.nan, np.nan)

      peak_watersheds.append("")

  print(peakQs_SP)
  print(peakQs_WN)
  print(peakQs_SA)
  print(peakQs_SD)

  ## Export new data to NCFR_Stats.csv file
  # Append peakQs column to dataframe
  ncfr_entries['peak_Q_SP'] = peakQs_SP
  ncfr_entries['peak_Q_WN'] = peakQs_WN
  ncfr_entries['peak_Q_SA'] = peakQs_SA
  ncfr_entries['peak_Q_SD'] = peakQs_SD
  ncfr_entries['peak_watershed'] = peak_watersheds

  # Export the updated dataframe
  ncfr_entries.to_csv(ncfr_fp)


if __name__ == '__main__':
  main()
