import pygrib
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.plots as mpplots # temp

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

def get_spatial_bounds(y, x, y_upper = -587432, y_lower = -920432, x_upper = -1378560, x_lower = -1841560):
  # Get the x and y spatial extent (should encompass SoCal only)
  y_bounds = np.where((y > y_lower) & (y < y_upper))[0]
  x_bounds = np.where((x > x_lower) & (x < x_upper))[0]

  return y_bounds, x_bounds


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
    return year_fp, date_fp, year, month, day
  else:
    # Return the date_fp unmodified 
    return year_fp, date_fp, year, month, day

def convert_date_num(month, day):
  # Create a list representing the number of days in each month (Jan-Dec)
  # First month is 0 because it's the beginning of the year
  days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
  
  # Initialize return variable, this will the time (in number of days) of the year the date falls on
  day_num = -1 # start at -1 instead of 0 since we are dealing with arrays in Python

  # Add the number of days based on the month
  for i in range(month):
    day_num = day_num + days_per_month[i]

  # Add number of days within a month
  day_num = day_num + day

  return day_num

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

  # Initialize variables
  total_prcp = np.array([])
  total_prcp_time = faux_end_hour - current_faux_hour # time length of NCFR event in hours

  # Retrieve data from each QPE file, so long as the file remains within the NCFR hours
  while (current_faux_hour <= faux_end_hour):
    # Check if the time reaches midnight, change the date and current_hour
    if current_hour == 24:
      # Reformat the date 
      year_fp, date_fp, year, month, day = reformat_time(date_fp, year, month, day, start_hour, end_hour)

      # Set the current hour to zero (24-hr time restart)
      current_hour = 0

    # Concatenate string to get the file path
    hour_fp = check_unit_time(current_hour) 
    fp = source_fp + year_fp + "/" + stage_fp + "." + date_fp + hour_fp + "." + type_fp

    # Retrieve data from file path
    grbs = pygrib.open(fp) 
    grb = grbs.message(1)
    data, lats, lons = grb.data(lat1 = 32, lat2 = 36, lon1 = -121, lon2 = -114)

    # Check if total_prcp is empty, set data to total_prcp
    # Otherwise, add data to the total_prcp
    if total_prcp.size == 0:
      total_prcp = data
    else:
      total_prcp = np.add(total_prcp, data)

    # Increment the hour
    current_hour += 1
    current_faux_hour += 1

  return total_prcp, total_prcp_time, lats, lons

def get_netcdf_prcp(year, month, day, start_hour, end_hour):
  # Initial variable used to keep time length (in hours) of NCFR event, will be used to calculate rain rates
  prcp_hours = 24 # default: 24 hours for 1 day

  # Load NEXRAD data from netcdf4 file
  source_fp = "/media/jntp/D2BC15A1BC1580E1/NCFRs/Daymet/"
  title_fp = "daymet_v4_daily_na_prcp_"
  year_fp, date_fp = create_date_fp(year, month, day)
 
  ncfile = source_fp + title_fp + year_fp + ".nc"
  nexdata = Dataset(ncfile, mode = 'r')
  print(nexdata)

  # Get y and x data from netcdf files and specify SoCal spatial bounds
  ys = nexdata['y'][:]
  xs = nexdata['x'][:]
  y_bounds, x_bounds = get_spatial_bounds(ys, xs) 

  # Get lat and lon data from netcdf file
  lons = nexdata['lon'][y_bounds, x_bounds]
  lats = nexdata['lat'][y_bounds, x_bounds]

  # Get the shape of either lons or lats to construct new matrix
  y, x = lons.shape

  # Initialize new numpy matrices
  prcp1 = np.zeros((y, x)) 
  prcp2 = np.zeros((y, x))
 
  # Check if the NCFR goes overnight
  # If so, will obtain precipitation data for 2 days
  if isOvernight(start_hour, end_hour):
    # Find the index of the prcp array given the month and day
    ntim1 = convert_date_num(month, day)

    # Reformat the time, will be used to obtain the 2nd prcp array
    year_fp, date_fp, year, month, day = reformat_time(date_fp, year, month, day, start_hour, end_hour)

    # Find the index of the 2nd prcp array given the month and day
    ntim2 = convert_date_num(month, day)

    # Get the precipitation data
    prcp1 = nexdata['prcp'][ntim1, y_bounds, x_bounds]
    prcp2 = nexdata['prcp'][ntim2, y_bounds, x_bounds]

    # Set prcp_hours to 48 to count for 2 days
    prcp_hours = 48
  else:
    # If NCFR only spans 1 day, simply obtain precipitation for only 1 day
    ntim1 = convert_date_num(month, day)

    # Get the precipitation dadta
    prcp1 = nexdata['prcp'][ntim1, y_bounds, x_bounds]

  # Add the prcp matrices together and return the results
  prcp = np.add(prcp1, prcp2)
  return prcp, prcp_hours, lats, lons

## Plotting Functions

# Create a base map to diplay QPE data
def new_map(fig, lon, lat):
  # Create projection centered on the radar. Allows us to use x and y relative to the radar
  proj = ccrs.LambertConformal(central_longitude = lon, central_latitude = lat)

  # New axes with the specified projection
  ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection = proj)

  # Add coastlines and states
  ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 2)
  ax.add_feature(cfeature.STATES.with_scale('50m'))

  return ax


def main():  
  ## Get precipitation climatology; will be used to calculate percent of normal precipitation
  # Load NEXRAD data from netcdf4 file
  climo_fp = "/media/jntp/D2BC15A1BC1580E1/NCFRs/Daymet/sdat_climatology.nc"
  climo_data = Dataset(climo_fp, mode = 'r')
  print(climo_data)

  # Specify SoCal spatial bounds in climatology file
  y_climo = climo_data['y'][:]
  x_climo = climo_data['x'][:] 
  y_bounds, x_bounds = get_spatial_bounds(y_climo, x_climo)

  # Get the prcp data from climo_data netcdf file
  climo_prcp = climo_data['Band1'][y_bounds, x_bounds]
 
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

  # Initialize variables
  grib_total_prcps = np.array([])
  grib_lons = np.array([])
  grib_lats = np.array([])
  grib_total_hours = 0
  grib_num_events = 0
  grib_num_years = 1 # change this later (when running entire code)
  stage4_total_prcps = np.array([])
  stage4_lons = np.array([])
  stage4_lats = np.array([]) 
  stage4_total_hours = 0
  stage4_num_events = 0

  for index in indexes:
    # Get entry information
    ncfr_entry = ncfr_entries.loc[index, "Year":"End_Hour"]
    year = ncfr_entry["Year"]
    month = ncfr_entry["Month"]
    day = ncfr_entry["Day"]
    start_hour = ncfr_entry["Start_Hour"]
    end_hour = ncfr_entry["End_Hour"]
  
    # TEMPORARY - PLEASE CHANGE THE YEARS IN IF STATEMENT WHEN DONE!!!
    # Check the year; this will determine what type of file to read
    if year != 2002: # Use NetCDF Daymet files
      # Test (only temporary) 
      if year == 1995:
        # Get precipitation and number of hours for each NCFR event 
        prcp, event_hours, lats, lons = get_netcdf_prcp(year, month, day, start_hour, end_hour)

        # Add event_hours to total number of hours; will be used to calculate avg rain rates
        grib_total_hours += event_hours

        # Aggregate prcp to grib_total_prcps array
        # Check if the grib_total_prcps array is empty, set that equal to prcp (first iteration)
        if grib_total_prcps.size == 0:
          grib_total_prcps = prcp 
        else:
          # Add the precip values together
          grib_total_prcps = np.add(grib_total_prcps, prcp)

        # Set lons and lats to grib_lons and grib_lats; will be used for plotting
        if grib_lons.size == 0 and grib_lats.size == 0:
          grib_lons = lons
          grib_lats = lats

        # Increment the number of events by 1
        grib_num_events += 1
      else:
        continue
    elif year == 2002: # Use Stage IV GRIB data
      # Read data from Stage IV precipitation files (GRIB) 
      event_prcp, event_prcp_time, lats, lons = get_GRIB_data(year, month, day, start_hour, end_hour)
      event_prcp_size = len(event_prcp)

      # Add hours to stage4_total_hours
      stage4_total_hours += event_prcp_time 
 
      ## Find total precipitation for all NCFR events
      # Check if total precipitation array is empty, set that equal to event_prcp (first iteration)
      if stage4_total_prcps.size == 0:
        stage4_total_prcps = event_prcp
      elif event_prcp_size < len(stage4_total_prcps):
        # Since the size of data is not always the same per iteration, check for discrepancies in size
        # Subtract to get difference and make a new array of zeros of that size
        new_len = len(stage4_total_prcps) - event_prcp_size
        zero_vals = np.zeros(new_len)
        
        # Make a masked array out of the numpy array and append to data (also masked)
        zero_vals_ma = np.ma.masked_array(data = zero_vals, mask = True)
        event_prcp = np.ma.append(event_prcp, zero_vals_ma)

        # Add precip values together
        stage4_total_prcps = np.add(stage4_total_prcps, event_prcp)
      else:
        # Simply add precip values together if size of data matches original array
        stage4_total_prcps = np.add(stage4_total_prcps, event_prcp)

      # Set lons and lats to grib_lons and grib_lats; will be used for plotting
      if stage4_lons.size == 0 and stage4_lats.size == 0:
        stage4_lons = lons
        stage4_lats = lats

      # Increment the number of events by 1
      stage4_num_events += 1

  ## Find the percent of normal annual precipitation (total NCFR precip year/total precip year)
  prop_normal_prcp = grib_total_prcps / (climo_prcp * grib_num_years)

  ## Find the average rainfall rate for each NCFR event
  grib_rain_rate = grib_total_prcps / grib_total_hours
  stage4_rain_rate = stage4_total_prcps / stage4_total_hours
  
  ## Find the average total precipitation per NCFR event
  grib_avg_prcp = grib_total_prcps / grib_num_events 
  stage4_avg_prcp = stage4_total_prcps / stage4_num_events
  
  ## Plot the data
  # Specify a central longitude and latitude (i.e. reference point)
  central_lon = -117.636 
  central_lat = 33.818

  # Create a new figure and map 
  fig = plt.figure(figsize = (10, 10))
  ax = new_map(fig, central_lon, central_lat) # -117.636, 33.818 

  # Set limits in lat/lon 
  ax.set_extent([-121, -114, 32, 36]) # SoCal

  # Get color table and value mapping info for the NWS Reflectivity data (test and temporary)
  ref_norm, ref_cmap = mpplots.ctables.registry.get_with_steps('NWSReflectivity', 5, 5)

  # Transform to this projection
  use_proj = ccrs.LambertConformal(central_longitude = central_lon, central_latitude = central_lat)

  # Transfer lats, lons matrices from geodetic lat/lon to LambertConformal
  out_xyz = use_proj.transform_points(ccrs.Geodetic(), grib_lons, grib_lats) 
  
  # Separate x, y from out_xyz
  grib_x = out_xyz[:, :, 0] 
  grib_y = out_xyz[:, :, 1]

  # More Test
  # print(stage4_lats[0:15])
  # Use np.reshape to convert to 2d array (also check documentation)

  # Test 
  test_contour = ax.contourf(stage4_lons, stage4_lats, stage4_total_prcps)

  plt.show()
  

if __name__ == '__main__':
  main()

# Also make functions for plotting
