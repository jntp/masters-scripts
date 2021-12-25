'''
Script for reading WWAs and NCFR catalog csv files. Checks to see if time matches for flash flood events
'''
import datetime as dt
import numpy as np 
from statistics import mean
from array import * 


def adjustNCFRtime(ncfr_start_hours, ncfr_end_hours):
  # Create array to store boolean values
  is_overnight = []

  # Loop through ncfr start and end hours 
  for i, hour in enumerate(ncfr_start_hours):
    # Check if the end time is "less" than the start time
    if ncfr_start_hours[i] < ncfr_end_hours[i]:
      is_overnight.append(1) # True for overnight spillover
    else:
      is_overnight.append(0) # False for overnight spillover

  is_overnight = np.array(is_overnight, dtype = 'bool') 
  
  return is_overnight 

def isMonthEnd(ncfr_month, ncfr_day):
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
    return True
  else:
    return False

def createDateTimes(ncfr_years, ncfr_months, ncfr_days, ncfr_start_hours, ncfr_end_hours):
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

def isFFwithinNCFR(FF_start_time, FF_end_time, NCFR_start_time, NCFR_end_time, ff_threshold):
  FF_end_bound = FF_end_time + ff_threshold
  isFFNCFR = []

  if NCFR_start_time <= FF_end_bound and NCFR_end_time <= FF_end_bound:
    isFFNCFR.append("True")
  else: 
    isFFNCFR.append("False") 

  return isFFNCFR

def main():
  wwa_fp = './data/WWA_All_1995_2020.csv'
  NCFR_fp = './data/NCFR_Catalog.csv'

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
  ff_start_datetimes = [] 
  ff_end_datetimes = []
  ff_start_dates = []
  ff_end_dates = [] 
  ff_time_lengths = [] 

  # Initialize stored variables
  ff_time_sum = dt.timedelta() # start with a duration of 0:00:00; used as sum for mean calculation

  for line in wwa_lines:
    components = line.split(",")
    # 1 = year, 2 = month, 3 = day, 4 = hour, 5 = min for start times
    # 6 = year, 7, = month, 8 = day, 9 = hour, 10 = min for end times

    if 'FF' in components:
      try:
        start_datetime = dt.datetime(int(components[1]), int(components[2]), int(components[3]), int(components[4]), \
            int(components[5]))
        end_datetime = dt.datetime(int(components[6]), int(components[7]), int(components[8]), int(components[9]), \
            int(components[10]))
      except:
        continue
      else:
        ff_start_datetimes.append(start_datetime) 
        ff_end_datetimes.append(end_datetime)
        ff_start_dates.append(start_datetime.date())
        ff_end_dates.append(end_datetime.date())

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

  print(ff_start_dates) 
  # Outline
  # Search in ff_start_time where start_time month and day match ncfr one
  for i, ncfr_start_date in enumerate(ncfr_start_dates):
    match_flag = False

    # Call isFFwithinNCFR function for matching days
    # Match ff_start_time with ncfr_start_time 
    for j, ff_start_date in enumerate(ff_start_dates): 
      if ff_start_date == ncfr_start_date or ff_start_date == ncfr_end_dates[i]:
        # Call Function isFFwithinNCFR
        print("Match", i)  

if __name__ == '__main__':
  main()

 
