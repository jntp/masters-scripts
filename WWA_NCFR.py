'''
Script for reading WWAs and NCFR catalog csv files. Checks to see if time matches for flash flood events
'''
import datetime as dt
from statistics import mean 

# def isFFwithinNCFR(FF_start_time, FF_end_time, NCFR_start_time, NCFR_end_time):
# First figure out an appropriate threshold... you will need to justify this in your thesis!
# also the NCFR catalog is weird... if the end time is past 00:00 it will still be considered the same date
# also review FF criteria... because FF be issued well after NCFR has passed
# review what type of urban flooding you're look at in yo thesis

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

        ff_length = end_datetime - start_datetime
        ff_time_lengths.append(ff_length)
        ff_time_sum += ff_length 

  # Find the mean of the flash flood warning time lengths to get the "threshold" 
  ff_treshold = ff_time_sum / len(ff_time_lengths)
    
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

  test1 = dt.datetime(2020, 2, 16, 5, 30)
  test2 = dt.datetime(2020, 2, 16, 8, 40)

  
   

if __name__ == '__main__':
  main()

# Next step... Write isFFwithinNCFR function 
