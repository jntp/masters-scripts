import pygrib
import pandas as pd
from netCDF4 import Dataset

## Auxiliary functions
def check_unit_time(unit_time):
  unit_time_str = str(unit_time)

  if unit_time < 10:
    unit_time_str = "0" + unit_time_str

  return unit_time_str

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

## Main Functions

def create_date_fp(year, month, day):
  year_fp = str(year)
  month_fp = check_unit_time(month)
  day_fp = check_unit_time(day)

  date_fp = year_fp + month_fp + day_fp

  return year_fp, date_fp

def reformat_time(date_fp, year, month, day, start_hour, end_hour):
  # Check if the NCFR goes overnight
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

    # Modify the date_fp given new time parameters
    year_fp, date_fp = create_date_fp(year, month, day)

    # Return modified date_fp and year (data sorted by years)
    return year_fp, date_fp
  else:
    # Return the date_fp unmodified 
    return year_fp, date_fp 

def get_GRIB_data(year, month, day, start_hour, end_hour, type_fp = "01h"):
  """
  Parameters:
    year - int
    month - int
    day - int 
    start_hour - int
    end_hour - int
    type_fp - string (default: hourly) 
  """
  source_fp = "/media/jntp/D2BC15A1BC1580E1/NCFRs/QPE Data/"
  stage_fp = "ST4"
  year_fp, date_fp = create_date_fp(year, month, day)

  # Set variables that will be used to iterate through the QPE files
  current_hour = start_hour
  current_faux_hour = start_hour
  faux_end_hour = end_hour # will be used to determine when the while loop below will stop

  # Add hours to 24 if the NCFR ends overnight
  if isOvernight(start_hour, end_hour):
    faux_end_hour = 24 + end_hour

  # Retrieve data from each QPE file, so long as the file remains within the NCFR hours
  while (current_faux_hour <= faux_end_hour):
    # Check if the time reaches midnight, change the date and current_hour
    if current_hour == 24:
      # Reformat the date 
      year_fp, date_fp = reformat_time(date_fp, year, month, day, start_hour, end_hour)

      # Set the current hour to zero (24-hr time restart)
      current_hour = 0

    # Concatenate string to get the file path
    hour_fp = check_unit_time(current_hour) 
    fp = source_fp + year_fp + "/" + stage_fp + "." + date_fp + hour_fp + "." + type_fp

    # Retrieve data from file path
    grbs = pygrib.open(fp) 
    grb = grbs.message(1)
    data, lats, lons = grb.data(lat1 = 32, lat2 = 36, lon1 = -121, lon2 = -114)

    # Increment the hour
    current_hour += 1
    current_faux_hour += 1

  return data, lats, lons

def open_netcdf(): # add more args later
  # Load NEXRAD data from netcdf4 file
  source_fp = "/media/jntp/D2BC15A1BC1580E1/NCFRs/Daymet/"
  title_fp = "daymet_v4_daily_na_prcp_"
  year_fp = "1995"
  ncfile = source_fp + title_fp + year_fp + ".nc"
  nexdata = Dataset(ncfile, mode = 'r')
  print(nexdata)

def main(): 
  # source_fp = "/media/jntp/D2BC15A1BC1580E1/NCFRs/QPE Data/2002/"
  # stage_fp = "ST4"
  # date_fp = "20020101"
  # hour_fp = "00"
  # type_fp = "01h"
  # fp = source_fp + stage_fp + "." + date_fp + hour_fp + "." + type_fp

  # grbs = pygrib.open(fp)
  # grb = grbs.message(1) 
  # grb = grbs.select(name = "Total Precipitation")[0]
  # precip = grb.values
  # print(precip.shape, precip.min(), precip.max())
  # data, lats, lons = grb.data(lat1 = 32, lat2 = 36, lon1 = -121, lon2 = -114)
  # print(data.shape, data.max(), data.min(), lats.shape, lons.shape)
  # print(data[50], lats[50], lons[50])

  ## Total Precipitation for all NCFR events
  # Load times from csv file
  ncfr_fp = "./data/NCFR_Catalog.csv" 
  ncfr_entries = pd.read_csv(ncfr_fp) 
  indexes = ncfr_entries.loc[:, "Index"] 
  years = ncfr_entries.loc[:, "Year"]
  months = ncfr_entries.loc[:, "Month"]
  days = ncfr_entries.loc[:, "Day"]
  start_hours = ncfr_entries.loc[:, "Start_Hour"]
  end_hours = ncfr_entries.loc[:, "End_Hour"]
  test = ncfr_entries.loc[0, "Year":"Day"] 
  for index in indexes:
    # Get entry information
    ncfr_entry = ncfr_entries.loc[index, "Year":"End_Hour"]
    year = ncfr_entry["Year"]
    print(index, year) # test
    month = ncfr_entry["Month"]
    day = ncfr_entry["Day"]
    start_hour = ncfr_entry["Start_Hour"]
    end_hour = ncfr_entry["End_Hour"]
  
    # Check the year; this will determine what type of file to read
    if year < 2002:
      # Test (only temporary) 
      if year == 1995:
        open_netcdf() 
      else:
        continue
    elif year >= 2002:
      print(year, month, day, start_hour, end_hour)
      # Read data from Stage IV precipitation files (GRIB) 
      data, lats, lons = get_GRIB_data(year, month, day, start_hour, end_hour)

if __name__ == '__main__':
  main()

# Find:
# Total precipitation for all NCFR events
# Percent of normal annual precipitation (total NCFR precip year/total precip year)
# Average rainfall rates
# Average total precipitation per NCFR event
