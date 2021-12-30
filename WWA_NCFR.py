'''
Script for reading WWAs and NCFR catalog csv files. Checks to see if time matches for flash flood events
'''
import os
import datetime as dt
import numpy as np 
from statistics import mean
from array import * 
from csv import writer
from csv import reader

def adjustNCFRtime(ncfr_start_hours, ncfr_end_hours):
  """
  Check if NCFR event spans more than 1 day (goes overnight). Returns bool array.

  """

  # Create array to store boolean values
  is_overnight = []

  # Loop through ncfr start and end hours 
  for i, hour in enumerate(ncfr_start_hours):
    # print(ncfr_start_hours[i], ncfr_end_hours[i]) 
    # Check if the end time is "less" than the start time
    if ncfr_start_hours[i] > ncfr_end_hours[i]:
      is_overnight.append(1) # True for overnight spillover
    else:
      is_overnight.append(0) # False for overnight spillover

  is_overnight = np.array(is_overnight, dtype = 'bool') 
  
  return is_overnight 

def isMonthEnd(ncfr_month, ncfr_day):
  """
  Check if given month and day is the last day of the month. Returns bool value.
  """

  # Initialize variables; variables check whether month and day could mean end of month
  right_day = False
  right_month = False

  # Check if day could entail the "end of month"
  if ncfr_day == 28 or ncfr_day == 30 or ncfr_day == 31:
    right_day = True

  # Check if the day is the end of the month for that specific month
  if right_day:
    # Check if the month is February
    if ncfr_month == 2:
      right_month = True

    # Check order months, whether a month has 30 or 31 days gets swapped in August
    if ncfr_month <= 7:
      # Jan/Mar/May/July has 31 days
      if ncfr_month % 2 == 1 and ncfr_day == 31:
        right_month = True 
      # April/June has 30 days
      elif ncfr_month % 2 == 0 and ncfr_day == 30:
        right_month = True
    elif ncfr_month >= 8:
      # Aug/Oct/Dec has 31 days
      if ncfr_month % 2 == 0 and ncfr_day == 31:
        right_month = True
      # Sept/Nov has 30 days
      elif ncfr_month % 2 == 1 and ncfr_day == 30:
        right_month = True

  # If month and day correspond to end of month, they return True
  if right_day and right_month:
    # print(right_day, right_month) 
    return True
  else:
    # print(right_day, right_month) 
    return False

def createDateTimes(ncfr_years, ncfr_months, ncfr_days, ncfr_start_hours, ncfr_end_hours):
  """
  Creates NCFR datetimes given year, month, day, start hour, and end hour. Returns four arrays consisting of
  datetimes--which including two that are the dates only (no hour/minutes). 
  """

  # Initialize lists
  ncfr_start_times = []
  ncfr_end_times = []
  ncfr_start_dates = []
  ncfr_end_dates = []

  # Call is_overnight to check if NCFR end time is on the next day of the start time
  overnighters = adjustNCFRtime(ncfr_start_hours, ncfr_end_hours)

  # Loop through overnighters, reorganize datetime? consider making datetime function 
  for i, overnighter in enumerate(overnighters):
    ncfr_start_time = dt.datetime(ncfr_years[i], ncfr_months[i], ncfr_days[i], ncfr_start_hours[i])
    ncfr_end_time = dt.datetime(ncfr_years[i], ncfr_months[i], ncfr_days[i], ncfr_end_hours[i]) 

    # Check if the NCFR end hour is on the following day of the start time
    if overnighter:
      if isMonthEnd(ncfr_months[i], ncfr_days[i]) == True:
        # Set to 1st day of the next month
        ncfr_end_time = dt.datetime(ncfr_years[i], ncfr_months[i] + 1, 1, ncfr_end_hours[i])
      else:
        # Set to next day (add 1 day)
        ncfr_end_time = dt.datetime(ncfr_years[i], ncfr_months[i], ncfr_days[i] + 1, ncfr_end_hours[i]) 

    ncfr_start_times.append(ncfr_start_time)
    ncfr_end_times.append(ncfr_end_time)
    ncfr_start_dates.append(ncfr_start_time.date())
    ncfr_end_dates.append(ncfr_end_time.date())

  return ncfr_start_times, ncfr_end_times, ncfr_start_dates, ncfr_end_dates

def isFFwithinNCFR(FF_start_time, FF_end_time, NCFR_start_time, NCFR_end_time, FF_status, ff_threshold):
  """
  Checks if a flash flood warning is issued within a time bounds of an NCFR. Returns a bool value.
  """

  FF_end_bound = FF_end_time + ff_threshold
  isFFNCFR = False
 
  if FF_start_time > NCFR_start_time and FF_start_time <= FF_end_bound and "NEW" in FF_status: 
    isFFNCFR = True

  return isFFNCFR

def add_column_in_csv(input_file, output_file, transform_row):
  """
  Adds a column to an existing csv file
  """

  # Open input file in read mode and output file in write mode
  with open(input_file, 'r') as read_obj, open(output_file, 'w', newline='') as write_obj:
    # Create a csv reader object from input file
    csv_reader = reader(read_obj)

    # Create a csv writer object from output file
    csv_writer = writer(write_obj)

    # Read each row of the input csv file as list
    for row in csv_reader:
      # Transform the row to "column" 
      transform_row(row, csv_reader.line_num)

      # Write "row" to output file
      csv_writer.writerow(row) 

def main():
  # Specify file paths 
  wwa_fp = './data/WWA_All_1995_2020.csv'
  NCFR_fp = './data/NCFR_Catalog.csv'
  new_fp1 = './output/NCFR_Catalog_1.csv'
  new_fp2 = './output/NCFR_Catalog_2.csv'
  new_fp3 = './output/NCFR_Catalog_3.csv' 
  new_fp4 = './output/NCFR_Catalog_final.csv' 

  # Collect WWA data 
  wwa_file = open(wwa_fp, 'r')
  wwa_lines = wwa_file.readlines()
  wwa_file.flush()
  wwa_file.close()

  # Collect NCFR data
  ncfr_file = open(NCFR_fp, 'r')
  ncfr_lines = ncfr_file.readlines()
  ncfr_file.flush()
  ncfr_file.close() 

  # Lists to obtain from the WWA file
  ff_WFOs = []
  ff_start_datetimes = [] 
  ff_end_datetimes = []
  ff_start_dates = []
  ff_end_dates = [] 
  ff_time_lengths = []
  ff_statuses = []

  # Initialize stored variables
  ff_time_sum = dt.timedelta() # start with a duration of 0:00:00; used as sum for mean calculation
  num_FFWs = 0 # counts the total number of FFWs assoicated with NCFRs

  for line in wwa_lines:
    components = line.split(",")
    # 1 = year, 2 = month, 3 = day, 4 = hour, 5 = min for start times
    # 6 = year, 7, = month, 8 = day, 9 = hour, 10 = min for end times, 12 for status

    if 'FF' in components:
      try:
        start_datetime = dt.datetime(int(components[1]), int(components[2]), int(components[3]), int(components[4]), \
            int(components[5]))
        end_datetime = dt.datetime(int(components[6]), int(components[7]), int(components[8]), int(components[9]), \
            int(components[10]))
      except:
        continue
      else:
        ff_WFOs.append(components[0]) 
        ff_start_datetimes.append(start_datetime) 
        ff_end_datetimes.append(end_datetime)
        ff_start_dates.append(start_datetime.date())
        ff_end_dates.append(end_datetime.date())
        ff_statuses.append(components[12]) 

        ff_length = end_datetime - start_datetime
        ff_time_lengths.append(ff_length)
        ff_time_sum += ff_length 

  # Find the mean of the flash flood warning time lengths to get the "threshold" 
  ff_threshold = ff_time_sum / len(ff_time_lengths)  

  # Lists to obtain from the NCFR file
  ncfr_years = []
  ncfr_months = []
  ncfr_days = []
  ncfr_start_hours = []
  ncfr_end_hours = []

  # Now time for NCFR bitches!!! 
  for line in ncfr_lines:
    components_ncfr = line.split(",")
    # 1 = year, 2 = month, 3 = day, 4 = start_hour, 5 = end_hour

    try: 
      ncfr_years.append(int(components_ncfr[1]))
      ncfr_months.append(int(components_ncfr[2]))
      ncfr_days.append(int(components_ncfr[3]))
      ncfr_start_hours.append(int(components_ncfr[4]))
      ncfr_end_hours.append(int(components_ncfr[5])) 
    except:
      continue 

  # Create ncfr datetimes, add start_dates, end_dates
  ncfr_start_datetimes, ncfr_end_datetimes, ncfr_start_dates, ncfr_end_dates = createDateTimes(ncfr_years, ncfr_months, \
      ncfr_days, ncfr_start_hours, ncfr_end_hours)

  ncfr_months = np.array(ncfr_months)
  ncfr_days = np.array(ncfr_days) 
  ncfr_if_ff = [] # stores True/False on whether ncfr is associated with flash flood warning
  ncfr_ff_start_times = [] # string list that stores starting datetimes of associated FFWs
  ncfr_ff_end_times = [] # string list that stores ending datetimes of associated FFWs
  ncfr_ff_WFOs = [] # indicates which weather forecasting office (LOX/SGX) issued the FFW 

  # Outline
  # Search in ff_start_time where start_time month and day match ncfr one
  for i, ncfr_start_date in enumerate(ncfr_start_dates):
    match_flag = False
    ncfr_ff_start_time = "N/A"
    ncfr_ff_end_time = "N/A" 
    ncfr_ff_WFO = "N/A"

    # Call isFFwithinNCFR function for matching days
    # Match ff_start_time with ncfr_start_time 
    for j, ff_start_date in enumerate(ff_start_dates): 
      if ff_start_date == ncfr_start_date or ff_start_date == ncfr_end_dates[i]:
        match_flag = isFFwithinNCFR(ff_start_datetimes[j], ff_end_datetimes[j], ncfr_start_datetimes[i], \
            ncfr_end_datetimes[i], ff_statuses[j], ff_threshold)
        
        if match_flag == True:
          ncfr_ff_start_time = str(ff_start_datetimes[j])
          ncfr_ff_end_time = str(ff_end_datetimes[j])
          ncfr_ff_WFO = ff_WFOs[j]
          num_FFWs += 1

          # Note: Remove break below if searching for num_FFWs and its proportion
          # INCLUDE break if searching for num_NCFRs and its proportion
          break # end search process for current NCFR event and continue to next event

    ncfr_if_ff.append(match_flag) 
    ncfr_ff_start_times.append(ncfr_ff_start_time)
    ncfr_ff_end_times.append(ncfr_ff_end_time)
    ncfr_ff_WFOs.append(ncfr_ff_WFO) 

  # Get NCFR and FFW statistics
  prop_FFWs = num_FFWs / len(ff_start_datetimes) # proportion of FFWs associated with NCFRs
  num_NCFRs = sum(ncfr_if_ff) # number of "True" values
  prop_NCFRs = num_NCFRs / len(ncfr_start_datetimes)

  # Print Statistics
  print("Number of FFWs linked with NCFRs: ", num_FFWs) 
  print("Proportion of FFWs linked with NCFRs: ", prop_FFWs)
  print("Number of NCFRs linked with FFWs ", num_NCFRs)
  print("Proportion of NCFRs linked with FFWs ", prop_NCFRs) 

  # Print to csv file
  header1 = "Associated_FFW"
  header2 = "FFW_Start_Time"
  header3 = "FFW_End_Time" 
  header4 = "WFO" 

  # Add columns to CSV files
  add_column_in_csv(NCFR_fp, new_fp1, lambda row, line_num: \
      row.append(header1) if line_num == 1 else row.append(ncfr_if_ff[line_num - 2]))
  add_column_in_csv(new_fp1, new_fp2, lambda row, line_num: \
      row.append(header2) if line_num == 1 else row.append(ncfr_ff_start_times[line_num - 2]))
  add_column_in_csv(new_fp2, new_fp3, lambda row, line_num: \
      row.append(header3) if line_num == 1 else row.append(ncfr_ff_end_times[line_num - 2]))
  add_column_in_csv(new_fp3, new_fp4, lambda row, line_num: \
      row.append(header4) if line_num == 1 else row.append(ncfr_ff_WFOs[line_num - 2]))

  # Remove earlier versions of CSV files 
  os.remove(new_fp1)
  os.remove(new_fp2) 
  os.remove(new_fp3) 

if __name__ == '__main__':
  main()
 
