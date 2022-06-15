import math
import pandas as pd
import datetime as dt
import numpy as np

## Auxiliary Functions

def isOvernight(start_hour, end_hour):
  is_overnight = False

  if start_hour > end_hour:
    is_overnight = True

  return is_overnight

## Main Functions

def convert_str_to_dt(datetime_str, code):
  # Code 0 works with datetimes in format YYYY-MM-DD
  # Code 1 works with datetimes in format MM/DD/YYYY 
  # Create slice objects for parsing string
  if code == 0: # YYYY-MM-DD
    year_slc = slice(0, 4)
    month_slc = slice(5, 7)
    day_slc = slice(8, 10)
  elif code == 1: # MM/DD/YYYY
    month_slc = slice(0, 2)
    day_slc = slice(3, 5)
    year_slc = slice(6, 10)

  # Parse string accordingly
  year = datetime_str[year_slc]
  month = datetime_str[month_slc]
  day = datetime_str[day_slc]

  # Create datetime object
  datetime = dt.datetime(int(year), int(month), int(day))

  return datetime

def load_data(file_path, date_str, parameter_str, code):
  entries = pd.read_csv(file_path)
  datetimes_str = entries.loc[:, date_str]
  parameters_str = entries.loc[:, parameter_str]

  # Create new lists that will hold converted datetime and parameter data
  datetimes = []
  parameters = []

  # Loop through entries and convert datetime and discharge data
  for i, datetime in enumerate(datetimes_str): 
    datetimes.append(convert_str_to_dt(datetimes_str[i], code))

    # Check code to decide if to convert to int or float
    if code == 0: # int discharge data
      parameters.append(int(parameters_str[i]))
    elif code == 1: # float precipitation data
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


def main():
  ## Load mean discharge and daily precipitation data
  # Get the file paths for mean discharge
  sepulveda_fp = "./data/sepulveda_dam_discharge_2002_2020.csv"
  whittier_fp = "./data/whittier_narrows_discharge_1995_2020.csv"
  santa_ana_fp = "./data/santa_ana_discharge_1995_2020.csv"
  san_diego_fp = "./data/san_diego_discharge_1995_2020.csv"

  # Load mean discharge data
  sepulveda_dts, sepulveda_Qs = load_data(sepulveda_fp, "datetime", "mean_discharge_cfs", 0)
  whittier_dts, whittier_Qs = load_data(whittier_fp, "datetime", "mean_discharge_cfs", 0)
  santa_ana_dts, santa_ana_Qs = load_data(santa_ana_fp, "datetime", "mean_discharge_cfs", 0)
  san_diego_dts, san_diego_Qs = load_data(san_diego_fp, "datetime", "mean_discharge_cfs", 0)

  # Get the file paths for daily precipitation
  cheeseboro_SP_fp = "./data/Cheeseboro_Sepulveda_precip_1995_2020.csv"
  santa_fe_WN_fp = "./data/Santa_Fe_Whittier_precip_1995_2020.csv"
  fremont_SA_fp = "./data/Fremont_Santa_Ana_precip_1995_2020.csv"
  elliot_SD_fp = "./data/Camp_Elliot_San_Diego_precip_2004_2020.csv"

  # Load daily precipitation data
  cheeseboro_SP_dts, cheeseboro_SP_prcp = load_data(cheeseboro_SP_fp, "Date", "Precip_mm", 1)
  santa_fe_WN_dts, santa_fe_WN_prcp = load_data(santa_fe_WN_fp, "Date", "Precip_mm", 1)
  fremont_SA_dts, fremont_SA_prcp = load_data(fremont_SA_fp, "Date", "Precip_mm", 1)
  elliot_SD_dts, elliot_SD_prcp = load_data(elliot_SD_fp, "Date", "Precip_mm", 1)

  ## Load data from NCFR_Stats
  ncfr_fp = "./data/NCFR_Stats.csv"
  ncfr_entries = pd.read_csv(ncfr_fp)
  years = ncfr_entries.loc[:, "Year"]
  months = ncfr_entries.loc[:, "Month"]
  days = ncfr_entries.loc[:, "Day"]
  start_hours = ncfr_entries.loc[:, "Start_Hour"]
  end_hours = ncfr_entries.loc[:, "End_Hour"]
  max_refs = ncfr_entries.loc[:, "Max_Ref"]
  peak_watersheds = ncfr_entries.loc[:, "peak_watershed"]

  ## Format the NCFR_Stats data and check for mean discharge and daily precipitation
  # Create list of drainage areas of watersheds [Sepulveda, Whittier, Santa Ana, San Diego]
  drainage_areas = [455060911, 327892495, 6363859786, 1116284876]

  # Create empty lists to store new data to be saved in dataframe
  mean_discharges = []
  mean_precips = []
  runoffs = []
  runoff_ratios = []

  # Iterate through every NCFR entry
  for i, year in enumerate(years):
    # Check for mean discharge and daily precip, only if max_refs has an entry
    if not math.isnan(max_refs[i]):
      # Check year/month/day of ncfr entry, check if it goes overnight
      # Create datetime of NCFR entry (plus 1 more day if it goes overnight)
      # Retrieve mean discharge that matches NCFR entry
      # Retrieve daily precip that matches NCFR entry

      # Create two NCFR datetime objects of the same day initially
      ncfr_dt = dt.datetime(int(years[i]), int(months[i]), int(days[i]))
      ncfr_dt2 = dt.datetime(int(years[i]), int(months[i]), int(days[i]))
      
      # Change the 2nd NCFR datetime if it goes overnight or if it ends past 9 pm
      if isOvernight(start_hours[i], end_hours[i]) or end_hours[i] >= 21:
        ncfr_dt2 = ncfr_dt + dt.timedelta(days = 1)

      # Retrieve the mean discharge and daily precipitation from the watershed that recorded peak streamflow
      # Calculate runoff ratio immediately after retrieving mean discharge and daily precipitation
      mean_discharge = 0 # default
      mean_precip = 0 # default
      runoff = 0 # default
      runoff_ratio = 0 # default

      # Use try/except blocks to incorporate "Plan B" if "Plan A" does not have measurements
      if peak_watersheds[i] == "SP":
        try:
          mean_discharge = get_mean_data(sepulveda_dts, sepulveda_Qs, ncfr_dt, ncfr_dt2)
          mean_precip = get_mean_data(cheeseboro_SP_dts, cheeseboro_SP_prcp, ncfr_dt, ncfr_dt2)
          runoff, runoff_ratio = get_runoff_ratio(mean_discharge, mean_precip, drainage_areas[0]) 
        except:
          mean_discharge = get_mean_data(whittier_dts, whittier_Qs, ncfr_dt, ncfr_dt2)
          mean_precip = get_mean_data(santa_fe_WN_dts, santa_fe_WN_prcp, ncfr_dt, ncfr_dt2)
          runoff, runoff_ratio = get_runoff_ratio(mean_discharge, mean_precip, drainage_areas[1]) 
      elif peak_watersheds[i] == "WN":
        try:
          mean_discharge = get_mean_data(whittier_dts, whittier_Qs, ncfr_dt, ncfr_dt2)
          mean_precip = get_mean_data(santa_fe_WN_dts, santa_fe_WN_prcp, ncfr_dt, ncfr_dt2)
          runoff, runoff_ratio = get_runoff_ratio(mean_discharge, mean_precip, drainage_areas[1]) 
        except:
          mean_discharge = get_mean_data(sepulveda_dts, sepulveda_Qs, ncfr_dt, ncfr_dt2)
          mean_precip = get_mean_data(cheeseboro_SP_dts, cheeseboro_SP_prcp, ncfr_dt, ncfr_dt2)
          runoff, runoff_ratio = get_runoff_ratio(mean_discharge, mean_precip, drainage_areas[0]) 
      elif peak_watersheds[i] == "SA":
        try:
          mean_discharge = get_mean_data(santa_ana_dts, santa_ana_Qs, ncfr_dt, ncfr_dt2)
          mean_precip = get_mean_data(fremont_SA_dts, fremont_SA_prcp, ncfr_dt, ncfr_dt2)
          runoff, runoff_ratio = get_runoff_ratio(mean_discharge, mean_precip, drainage_areas[2]) 
        except:
          mean_discharge = get_mean_data(whittier_dts, whittier_Qs, ncfr_dt, ncfr_dt2)
          mean_precip = get_mean_data(santa_fe_WN_dts, santa_fe_WN_prcp, ncfr_dt, ncfr_dt2)
          runoff, runoff_ratio = get_runoff_ratio(mean_discharge, mean_precip, drainage_areas[1]) 
      elif peak_watersheds[i] == "SD":
        try:
          mean_discharge = get_mean_data(san_diego_dts, san_diego_Qs, ncfr_dt, ncfr_dt2)
          mean_precip = get_mean_data(elliot_SD_dts, elliot_SD_prcp, ncfr_dt, ncfr_dt2)
          runoff, runoff_ratio = get_runoff_ratio(mean_discharge, mean_precip, drainage_areas[3]) 
        except:
          mean_discharge = np.nan
          mean_precip = np.nan
          runoff = np.nan
          runoff_ratio = np.nan

      # Check if the runoff_ratio equals infinity; set to NaN
      if runoff_ratio == float('inf'):
        runoff_ratio = np.nan
    else: # For entries with no max_ref
      # Set all parameters to "NaN"
      mean_discharge = np.nan
      mean_precip = np.nan
      runoff = np.nan
      runoff_ratio = np.nan

    # Append to lists
    mean_discharges.append(mean_discharge)
    mean_precips.append(mean_precip)
    runoffs.append(runoff)
    runoff_ratios.append(runoff_ratio)

  ## Save and update the NCFR_Stats.csv file
  # Save as new columns in pandas dataframe
  ncfr_entries['mean_discharge_cfs'] = mean_discharges 
  ncfr_entries['mean_precip_mm'] = mean_precips
  ncfr_entries['runoff_mm'] = runoffs
  ncfr_entries['runoff_ratio'] = runoff_ratios

  # Export the updated dataframe to csv file
  ncfr_entries.to_csv(ncfr_fp) 


if __name__ == '__main__':
  main()
