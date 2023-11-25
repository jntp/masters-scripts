import pandas as pd
import numpy as np
import datetime as dt

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

def check_duplicate_events(start_dates, start_dt):
  # Create a bool variable to check whether the FFW is a "duplicate" (same day) event 
  duplicates_flag = False

  # Check if start_dates list is empty (if at beginning of for loop) 
  if not start_dates:
    # Store start datetime of FFW in start_dates list
    # Return start_dates list and duplicates flag as it is
    start_dates.append(start_dt)
    return start_dates, duplicates_flag 
  else: # For not empty lists
    # Check if the day of the last start datetime matches the start datetime of this FFW
    if start_dates[-1].year == start_dt.year and start_dates[-1].month == start_dt.month and start_dates[-1].day == start_dt.day:
      # If matching, flag as duplicate event and immediately return the start_dates list and duplicates_flag variable
      duplicates_flag = True
      return start_dates, duplicates_flag
    else:
      # If not matching, leave duplicates flag unchanged and append the start datetime to end of list
      start_dates.append(start_dt) 
      return start_dates, duplicates_flag

def check_threshold(stream_datetimes, stream_Qs, start_datetime, end_datetime, discharge_threshold):
  # Convert datetimes and streamflows to numpy array for array operations
  stream_datetimes = np.array(stream_datetimes)
  stream_Qs = np.array(stream_Qs)

  # Look for indices where the time falls within the time of interest and its respective streamflow data
  time_inds = np.where((stream_datetimes >= start_datetime) & (stream_datetimes <= end_datetime))[0]
  time_Qs = stream_Qs[time_inds]
  
  # Return true if streamflow exceeds threshold, otherwise return false
  if np.any(time_Qs >= discharge_threshold):
    return True
  else:
    return False


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

  ## Find flash flood warnings from WWA catalog
  # Load from csv file
  wwa_fp = "./data/WWA_All_1995_2020.csv"
  wwa_entries = pd.read_csv(wwa_fp)
  wfos = wwa_entries.loc[:, "WFO"]
  beg_years = wwa_entries.loc[:, "BegYear"]
  beg_months = wwa_entries.loc[:, "BegMon"]
  beg_days = wwa_entries.loc[:, "BegDay"]
  beg_hours = wwa_entries.loc[:, "BegHur"]
  beg_mins = wwa_entries.loc[:, "BegMin"]
  end_years = wwa_entries.loc[:, "EndYear"]
  end_months = wwa_entries.loc[:, "EndMon"]
  end_days = wwa_entries.loc[:, "EndDay"]
  end_hours = wwa_entries.loc[:, "EndHur"]
  end_mins = wwa_entries.loc[:, "EndMin"]
  phenoms = wwa_entries.loc[:, "PHENOM"]
  statuses = wwa_entries.loc[:, "STATUS"] 

  # Get indices where entry meets criteria of being a new or continued FFW
  ffw_inds = phenoms[phenoms == "FF"].index.tolist() 
  new_inds = statuses[statuses == "NEW"].index.tolist()
  con_inds = statuses[statuses == "CON"].index.tolist()

  # Convert lists to numpy array for array operations
  ffw_inds = np.array(ffw_inds)
  new_inds = np.array(new_inds)
  con_inds = np.array(con_inds) 

  # Append the new and continued indices and sort them numerically
  new_con_inds = np.append(new_inds, con_inds)
  new_con_inds = np.sort(new_con_inds)

  # Create a list to store all the indices that meet the criteria
  right_inds = []

  # Find the indices where new_con_inds and ffw_inds match
  for new_con_ind in new_con_inds:
    if np.any(ffw_inds == new_con_ind):
      right_inds.append(new_con_ind)

  ## Check if the streamflow of each watershed exceeds threshold during the each FFW event
  # Get entries from WWA catalog that meet the criteria
  wfos_ff = wfos[right_inds]
  beg_years_ff = beg_years[right_inds]
  beg_months_ff = beg_months[right_inds]
  beg_days_ff = beg_days[right_inds]
  beg_hours_ff = beg_hours[right_inds]
  beg_mins_ff = beg_mins[right_inds]
  end_years_ff = end_years[right_inds]
  end_months_ff = end_months[right_inds]
  end_days_ff = end_days[right_inds]
  end_hours_ff = end_hours[right_inds]
  end_mins_ff = end_mins[right_inds]

  # Create another list to store data that meet both FFW criteria and streamflow threshold
  unique_inds = [] # based on right_inds but without any duplicates
  flood_inds = [] # indices
  results = [] # bool values
  
  # Store starting dates of FFWs, used to prevent counting duplicating flood events
  start_dates = []

  # Loop through each FFW entry; check if the streamflow during FFW time exceeds threshold
  for i in right_inds:
    # Create datetimes for the start and end FFW time
    start_dt = dt.datetime(int(beg_years_ff[i]), int(beg_months_ff[i]), int(beg_days_ff[i]), int(beg_hours_ff[i]), \
        int(beg_mins_ff[i]))
    end_dt = dt.datetime(int(end_years_ff[i]), int(end_months_ff[i]), int(end_days_ff[i]), int(end_hours_ff[i]), \
        int(end_mins_ff[i]))

    # Check if FFW is a "duplicate" event (same day as the previous entry)
    start_dates, is_duplicate = check_duplicate_events(start_dates, start_dt)

    # Continue to next iteration if event is a duplicate
    if is_duplicate:
      continue
    else:
      unique_inds.append(i)

    # Check which WFO issued the FFW so we know which watersheds to check
    if wfos_ff[i] == "LOX": 
      # Create threshold exceedance flags for each watershed within the WFO
      is_exceedance_SP = False # Sepulveda
      is_exceeance_WN = False # Whittier Narrows

      # Check if streamflows within event exceeds threshold
      # No data available for Sepulveda before 2002, only use Whittier Narrows event if occurs earlier
      if start_dt.year < 2002:
        is_exceedance_WN = check_threshold(whittier_dts, whittier_Qs, start_dt, end_dt, 7140)
      elif start_dt.year >= 2002:
        is_exceedance_sp = check_threshold(sepulveda_dts, sepulveda_Qs, start_dt, end_dt, 7380)
        is_exceedance_wn = check_threshold(whittier_dts, whittier_Qs, start_dt, end_dt, 7140)

      # If either watershed streamflow exceeds the threshold, append a "True" to results and their respective index
      if is_exceedance_SP or is_exceedance_WN:
        results.append(True)
        flood_inds.append(i)
      else:
        # Only worry about results; append a "False" to the list
        results.append(False)
    elif wfos_ff[i] == "SGX": 
      # Check if streamflows within event exceeds threshold
      is_exceedance_SA = check_threshold(santa_ana_dts, santa_ana_Qs, start_dt, end_dt, 3780)
      is_exceedance_SD = check_threshold(san_diego_dts, san_diego_Qs, start_dt, end_dt, 1460)

      # If either watershed streamflow exceeds the threshold, append a "True" to results and their respective index
      if is_exceedance_SA or is_exceedance_SD:
        results.append(True)
        flood_inds.append(i)
      else:
        # Only worry about results; append a "False" to the list
        results.append(False)

  # Get the number of SoCal flood events during the time period
  flood_count = sum(results)

  # Get the number of FFW issued by a WFO
  FFW_count = len(unique_inds)
  FFW_dup_count = len(right_inds)
  
  # Print results
  print(flood_inds)
  print("Total Number of Flood Events (1995-2020): ", flood_count)
  print("Total Number of unique FFWs: ", FFW_count) 
  print("Total Number of non-unique FFWs: ", FFW_dup_count)

if __name__ == '__main__':
  main()
