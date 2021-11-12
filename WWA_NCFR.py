'''
Script for reading WWAs and NCFR catalog csv files. Checks to see if time matches for flash flood events
'''
import datetime as dt 

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

  # Lists
  ff_start_years = []
  ff_start_months = []
  ff_start_days = []
  ff_start_hours = []
  ff_start_mins = []
  ff_end_years = []
  ff_end_months = []
  ff_end_days = []
  ff_end_hours = []
  ff_end_mins = []
 
  for line in wwa_lines:
    components = line.split(",")
    # 1 = year, 2 = month, 3 = day, 4 = hour, 5 = min for start times
    # 6 = year, 7, = month, 8 = day, 9 = hour, 10 = min for end times

    if 'FF' in components:
      try:
        ff_start_years.append(int(components[1]))
        ff_start_months.append(int(components[2]))
        ff_start_days.append(int(components[3]))
        ff_start_hours.append(int(components[4]))
        ff_start_mins.append(int(components[5]))
        ff_end_years.append(int(components[6]))
        ff_end_months.append(int(components[7]))
        ff_end_days.append(int(components[8]))
        ff_end_hours.append(int(components[9]))
        ff_end_mins.append(int(components[10]))
      except:
        continue
    
  print(ff_start_years) # test these 

  # print(dt.datetime(2020, 2, 16, 5, 30)) 

if __name__ == '__main__':
  main()

# Next step... for loop for NCFR_lines, then for loop for matching or something? 
