import pandas as pd
import numpy as np
import datetime as dt
import math
import statistics as stat

## Auxiliary Functions

def isOvernight(start_hour, end_hour):
  is_overnight = False

  if start_hour > end_hour:
    is_overnight = True

  return is_overnight

## Main Functions

def format_date_mdy(mdy_datetime_str):
  # Figure out what the format of the date string is (MM/DD/YYYY, MM/D/YYYY, M/DD/YYYY, or M/D/YYYY?)
  # Create the slice objects for the month, day, and year depending on the format
  
  if len(mdy_datetime_str) == 10: # MM/DD/YYYY
    month_slc = slice(0, 2)
    day_slc = slice(3, 5)
    year_slc = slice(6, 10)
  elif len(mdy_datetime_str) == 9: # M/DD/YYYY or MM/D/YYYY
    ## Find if the month or the day is double-digit and create slice object accordingly
    # Parse the month based on a "test slice" 
    test_month = mdy_datetime_str[slice(0, 2)]

    # Test if the month is valid by converting to an int, will reveal the date format
    # A valid month would be double-digit, single-digit months would be appear as "M/" which is an invalid
    # integer
    try:
      int(test_month)
    except: # M/DD/YYYY
      month_slc = slice(0, 1)
      day_slc = slice(2, 4)
      year_slc = slice(5, 9)
    else: # MM/D/YYYY
      month_slc = slice(0, 2)
      day_slc = slice(3, 4)
      year_slc = slice(5, 9)
  elif len(mdy_datetime_str) == 8: # M/D/YYYY
    month_slc = slice(0, 1)
    day_slc = slice(2, 3)
    year_slc = slice(4, 8)

  return month_slc, day_slc, year_slc

def convert_str_to_dt(datetime_str, time_str, code = 0):
  # Code 0 works with datetimes in format YYYY-MM-DD or YYYY/MM/DD (default)
  # Code 1 works with datetimes in format MM/DD/YYYY 
  # Create slice objects for parsing datetime string
  if code == 0: # YYYY-MM-DD
    year_slc = slice(0, 4)
    month_slc = slice(5, 7)
    day_slc = slice(8, 10)
  elif code == 1: # MM/DD/YYYY 
    # Check format (MM/DD/YYYY, M/DD/YYYY, etc.) and format accordingly
    month_slc, day_slc, year_slc = format_date_mdy(datetime_str) 
    
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

  # Account for "M" in missing precip data... code = 1 is precip, code = 0 is discharge

  # Loop through entries and convert datetime and precipitation data
  for i, datetime in enumerate(datetimes_str):  
    datetimes.append(convert_str_to_dt(datetimes_str[i], times_str[i], code))
  
    parameters.append(float(parameters_str[i]))
                                                                  
  return datetimes, parameters

def get_mean_data(para_datetimes, para_data, ncfr_dt1, ncfr_dt2):
  # Convert parameter datetimes and data to numpy array for array operations                      
  para_datetimes = np.array(para_datetimes)
  para_data = np.array(para_data)
    
   # Look for indices where parameter (streamflow, precip) time is within the ncfr start and end time
  time_inds = np.where(para_datetimes >= ncfr_dt1 and para_datetimes <= ncfr_dt2)[0][0]
    
  # Find the mean of the two mean data points
  mean_data = stat.mean(para_data[time_inds])
  print(mean_data) # delete later

  return mean_data

# def get_NCFR_precip(ncfr_dt, ncfr_dt2, RAWS_prcp):
  # while (
  

def convert_discharge_runoff(discharge, drainage_area, period = 86400):
  """
    Converts mean discharge (ft^3/s) to runoff (mm).

    Parameters:
    discharge - mean discharge in ft^3/s
    drainage_area - area of the watershed in m^2
    period - the time period where discharge is measured (default: 86400 seconds or 1 day)
  """
  # Convert discharge from ft^3/s to m^3/s
  discharge_cms = discharge * (0.3048)**3

  # Multiply by the period to obtain the depth only
  discharge_m3 = discharge_cms * period

  # Divide by drainage area (m^2) and multiply 1000 mm to obtain runoff in mm
  runoff = (discharge_m3 / drainage_area) * 1000

  return runoff

def get_runoff_ratio(discharge, precip_mm, drainage_area, period = 86400):
  # Get the runoff by converting discharge from ft^3/s to mm
  runoff = convert_discharge_runoff(discharge, drainage_area)

  # Get the runoff raio by dividing runoff by the precipitation
  runoff_ratio = runoff / precip_mm

  return runoff, runoff_ratio

def get_stats(stream_dts, stream_Qs, gauge_dts, gauge_prcps, ncfr_dt, ncfr_dt2, drainage_area):
  # Get mean_discharge, mean_precip, runoff, and runoff_ratio for stream and gauge
  try:
    mean_discharge = get_mean_data(stream_dts, stream_Qs, ncfr_dt, ncfr_dt2)
    mean_precip = get_mean_data(gauge_dts, gauge_prcps, ncfr_dt, ncfr_dt2)
    runoff, runoff_ratio = get_runoff_ratio(mean_discharge, mean_precip, drainage_area) 
  except:
    mean_discharge = np.nan
    mean_precip = np.nan
    runoff = np.nan
    runoff_ratio = np.nan

  # Check if the runoff_ratio equals infinity; set to NaN
  if runoff_ratio == float('inf'):
    runoff_ratio = np.nan

  return mean_discharge, mean_precip, runoff, runoff_ratio

def main():
  ## Load the discharge data from USGS gauges
  # Get the file paths for the 15 min discharge data

  # Load 15 min discharge data


  ## Load the hourly RAWS data
  # Get the file paths
  cheeseboro_SP_fp = "./data/WRCC_CampElliot_RAWS_Data.csv"

  datetime = dt.datetime(2023, 11, 26, 18, 30)
  print(datetime) # delete later

  # Load hourly precipitation data
  cheeseboro_SP_dts, cheeseboro_SP_prcp = load_data(cheeseboro_SP_fp, "Date", "Time", "Precip_in", 1) # Error, see note
  print(cheeseboro_SP_dts, cheeseboro_SP_prcp)
  # Load for other 3 RAWS sites (left off here... eventually once RAWS data are available)

  # Convert precipitation to mm

  ## Load data from NCFR_Stats2
  ncfr_fp = "./data/NCFR_Stats2.csv" # perhaps rename to NCFR_Stats2; see entries below to see the default
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
      # Create two NCFR datetime objects of the same day initially 
      ncfr_dt = dt.datetime(int(years[i]), int(months[i]), int(days[i]), int(start_hours[i]))
      ncfr_dt2 = dt.datetime(int(years[i]), int(months[i]), int(days[i]), int(end_hours[i]))

      # Change the 2nd NCFR datetime if it goes overnight or if it ends past 9 pm
      if isOvernight(start_hours[i], end_hours[i]) or end_hours[i] >= 21:
        ncfr_dt2 = ncfr_dt + dt.timedelta(days = 1)

      # Retrieve the mean discharge and daily precipitation from the watershed that recorded peak streamflow
      # Calculate runoff ratio immediately after retrieving mean discharge and daily precipitation
      # Use try/except blocks to incorporate "Plan B" if "Plan A" does not have measurements
      mean_Q_SP, mean_prcp_SP, runoff_SP, run_ratio_SP = get_stats(sepulveda_dts, sepulveda_Qs, cheeseboro_SP_dts, \
          cheeseboro_SP_prcp, ncfr_dt, ncfr_dt2, drainage_areas[0])
      mean_Q_WN, mean_prcp_WN, runoff_WN, run_ratio_WN = get_stats(whittier_dts, whittier_Qs, santa_fe_WN_dts, \
          santa_fe_WN_prcp, ncfr_dt, ncfr_dt2, drainage_areas[1])
      mean_Q_SA, mean_prcp_SA, runoff_SA, run_ratio_SA = get_stats(santa_ana_dts, santa_ana_Qs, fremont_SA_dts, \
          fremont_SA_prcp, ncfr_dt, ncfr_dt2, drainage_areas[2])
      mean_Q_SD, mean_prcp_SD, runoff_SD, run_ratio_SD = get_stats(san_diego_dts, san_diego_Qs, elliot_SD_dts, \
          elliot_SD_prcp, ncfr_dt, ncfr_dt2, drainage_areas[3])

      # Calculate the average precipitation and discharge during NCFR event (INSERT into function above)

    else: # For entries with no max_ref
      # Set all parameters to "NaN"
      mean_Q_SP = np.nan
      mean_Q_WN = np.nan
      mean_Q_SA = np.nan
      mean_Q_SD = np.nan
      mean_prcp_SP = np.nan
      mean_prcp_WN = np.nan
      mean_prcp_SA = np.nan
      mean_prcp_SD = np.nan
      runoff_SP = np.nan
      runoff_WN = np.nan
      runoff_SA = np.nan
      runoff_SD = np.nan
      run_ratio_SP = np.nan
      run_ratio_WN = np.nan
      run_ratio_SA = np.nan
      run_ratio_SD = np.nan

    # Append to lists
    mean_Qs_SP.append(mean_Q_SP)
    mean_Qs_WN.append(mean_Q_WN)
    mean_Qs_SA.append(mean_Q_SA)
    mean_Qs_SD.append(mean_Q_SD)
    mean_prcps_SP.append(mean_prcp_SP)
    mean_prcps_WN.append(mean_prcp_WN)
    mean_prcps_SA.append(mean_prcp_SA)
    mean_prcps_SD.append(mean_prcp_SD)
    runoffs_SP.append(runoff_SP)
    runoffs_WN.append(runoff_WN)
    runoffs_SA.append(runoff_SA)
    runoffs_SD.append(runoff_SD)
    run_ratios_SP.append(run_ratio_SP)
    run_ratios_WN.append(run_ratio_WN)
    run_ratios_SA.append(run_ratio_SA)
    run_ratios_SD.append(run_ratio_SD)

  ## Save and update the NCFR_Stats.csv file
  # Save as new columns in pandas dataframe
  ncfr_entries['mean_discharge_SP'] = mean_Qs_SP
  ncfr_entries['mean_discharge_WN'] = mean_Qs_WN 
  ncfr_entries['mean_discharge_SA'] = mean_Qs_SA
  ncfr_entries['mean_discharge_SD'] = mean_Qs_SD 
  ncfr_entries['mean_precip_SP'] = mean_prcps_SP
  ncfr_entries['mean_precip_WN'] = mean_prcps_WN
  ncfr_entries['mean_precip_SA'] = mean_prcps_SA
  ncfr_entries['mean_precip_SD'] = mean_prcps_SD
  ncfr_entries['runoff_SP'] = runoffs_SP
  ncfr_entries['runoff_WN'] = runoffs_WN
  ncfr_entries['runoff_SA'] = runoffs_SA
  ncfr_entries['runoff_SD'] = runoffs_SD
  ncfr_entries['runoff_ratio_SP'] = run_ratios_SP
  ncfr_entries['runoff_ratio_WN'] = run_ratios_WN
  ncfr_entries['runoff_ratio_SA'] = run_ratios_SA
  ncfr_entries['runoff_ratio_SD'] = run_ratios_SD

  # Export the updated dataframe to csv file
  ncfr_entries.to_csv(ncfr_fp) 

if __name__ == '__main__':
  main()

# Figure out how the RAWS data is organized and change accordingly
# Move the load precipitation process to main() into a for loop (load precip by NCFR event only)
# Left off at fixing error in load_data() function regarding precipitation "M" error