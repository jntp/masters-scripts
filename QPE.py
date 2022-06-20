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
  ax = fig.add_axes([0.08, 0.04, 0.96, 0.96], projection = proj)

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
  grib_num_years = 26

  for index in indexes:
    # Get entry information
    ncfr_entry = ncfr_entries.loc[index, "Year":"End_Hour"]
    year = ncfr_entry["Year"]
    month = ncfr_entry["Month"]
    day = ncfr_entry["Day"]
    start_hour = ncfr_entry["Start_Hour"]
    end_hour = ncfr_entry["End_Hour"]
  
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
    
  ## Find the percent of normal annual precipitation (total NCFR precip year/total precip year)
  prop_normal_prcp = (grib_total_prcps / (climo_prcp * grib_num_years)) * 100

  ## Find the average rainfall rate for each NCFR event
  grib_rain_rate = grib_total_prcps / grib_total_hours
  
  ## Find the average total precipitation per NCFR event
  grib_avg_prcp = grib_total_prcps / grib_num_events
  
  ### Plot the data
  ## Get x and y coordinates
  # Specify a central longitude and latitude (i.e. reference point)
  central_lon = -117.636 
  central_lat = 33.818

  # Transform to this projection
  use_proj = ccrs.LambertConformal(central_longitude = central_lon, central_latitude = central_lat)

  # Transfer lats, lons matrices from geodetic lat/lon to LambertConformal
  out_xyz = use_proj.transform_points(ccrs.Geodetic(), grib_lons, grib_lats) 
  
  # Separate x, y from out_xyz
  grib_x = out_xyz[:, :, 0] 
  grib_y = out_xyz[:, :, 1]
 
  ## Total Precipitation from NCFR Events (1995-2020)
  # Create a new figure and map 
  fig = plt.figure(1, figsize = (10, 10))
  ax = new_map(fig, central_lon, central_lat) # -117.636, 33.818 

  # Set limits in lat/lon 
  ax.set_extent([-121, -114, 32, 36]) # SoCal

  # Create the contour plot
  grib_contour = ax.contourf(grib_x, grib_y, grib_total_prcps)

  # Set title and labels for x and y axis
  ax.set_title("QPE - Total Precipitation from NCFR Events (1995-2020)")  

  # Add color bar
  cbar = fig.colorbar(grib_contour, pad = 0.05, shrink = 0.6)
  cbar.ax.set_ylabel("Precipitation [mm]")
  
  # Set x and y ticks and labels
  xtics = np.arange(-300000, 400000, 100000)
  ytics = np.arange(-200000, 300000, 100000)
  ax.set_xticks(xtics)
  ax.set_xticklabels([r"$121^\circ W$", r"$120^\circ W$", r"$119^\circ W$", r"$118^\circ W$", \
          r"$117^\circ W$", r"$116^\circ W$", r"$115^\circ W$"])
  ax.set_yticks(ytics)
  ax.set_yticklabels([r"$32^\circ N$", r"$33^\circ N$", r"$34^\circ N$", r"$35^\circ N$", r"$36^\circ N$"])

  # Save Plot
  plt.savefig('./plots/QPE_total_precip')

  ## Percent of Normal Annual Precipitation
  # Create a new figure and map 
  fig = plt.figure(2, figsize = (10, 10))
  ax = new_map(fig, central_lon, central_lat) # -117.636, 33.818 

  # Set limits in lat/lon 
  ax.set_extent([-121, -114, 32, 36]) # SoCal

  # Create the contour plot
  grib_contour = ax.contourf(grib_x, grib_y, prop_normal_prcp)

  # Set title and labels for x and y axis
  ax.set_title("QPE - Percent of Normal Annual Precipitation from NCFR Events (1995-2020)")  

  # Add color bar
  cbar = fig.colorbar(grib_contour, pad = 0.05, shrink = 0.6)
  cbar.ax.set_ylabel("Percent of Normal Precipitation [%]")
  
  # Set x and y ticks and labels
  xtics = np.arange(-300000, 400000, 100000)
  ytics = np.arange(-200000, 300000, 100000)
  ax.set_xticks(xtics)
  ax.set_xticklabels([r"$121^\circ W$", r"$120^\circ W$", r"$119^\circ W$", r"$118^\circ W$", \
          r"$117^\circ W$", r"$116^\circ W$", r"$115^\circ W$"])
  ax.set_yticks(ytics)
  ax.set_yticklabels([r"$32^\circ N$", r"$33^\circ N$", r"$34^\circ N$", r"$35^\circ N$", r"$36^\circ N$"])

  # Save Plot
  plt.savefig('./plots/QPE_percent_normal_precip')

  ## Average Rainfall Rate
  # Create a new figure and map 
  fig = plt.figure(3, figsize = (10, 10))
  ax = new_map(fig, central_lon, central_lat) # -117.636, 33.818 

  # Set limits in lat/lon 
  ax.set_extent([-121, -114, 32, 36]) # SoCal

  # Create the contour plot
  grib_contour = ax.contourf(grib_x, grib_y, grib_rain_rate)

  # Set title and labels for x and y axis
  ax.set_title("QPE - Average Rain Rate of NCFR Events (1995-2020)")  

  # Add color bar
  cbar = fig.colorbar(grib_contour, pad = 0.05, shrink = 0.6)
  cbar.ax.set_ylabel("Rain rate [mm/h]")
  
  # Set x and y ticks and labels
  xtics = np.arange(-300000, 400000, 100000)
  ytics = np.arange(-200000, 300000, 100000)
  ax.set_xticks(xtics)
  ax.set_xticklabels([r"$121^\circ W$", r"$120^\circ W$", r"$119^\circ W$", r"$118^\circ W$", \
          r"$117^\circ W$", r"$116^\circ W$", r"$115^\circ W$"])
  ax.set_yticks(ytics)
  ax.set_yticklabels([r"$32^\circ N$", r"$33^\circ N$", r"$34^\circ N$", r"$35^\circ N$", r"$36^\circ N$"])

  # Save Plot
  plt.savefig('./plots/QPE_rain_rate')

  ## Average Total Precipitation per NCFR Event
  # Create a new figure and map 
  fig = plt.figure(4, figsize = (10, 10))
  ax = new_map(fig, central_lon, central_lat) # -117.636, 33.818 

  # Set limits in lat/lon 
  ax.set_extent([-121, -114, 32, 36]) # SoCal

  # Create the contour plot
  grib_contour = ax.contourf(grib_x, grib_y, grib_avg_prcp)

  # Set title and labels for x and y axis
  ax.set_title("QPE - Average Total Precipitation per NCFR Event (1995-2020)")  

  # Add color bar
  cbar = fig.colorbar(grib_contour, pad = 0.05, shrink = 0.6)
  cbar.ax.set_ylabel("Precipitation per event [mm/event]")
  
  # Set x and y ticks and labels
  xtics = np.arange(-300000, 400000, 100000)
  ytics = np.arange(-200000, 300000, 100000)
  ax.set_xticks(xtics)
  ax.set_xticklabels([r"$121^\circ W$", r"$120^\circ W$", r"$119^\circ W$", r"$118^\circ W$", \
          r"$117^\circ W$", r"$116^\circ W$", r"$115^\circ W$"])
  ax.set_yticks(ytics)
  ax.set_yticklabels([r"$32^\circ N$", r"$33^\circ N$", r"$34^\circ N$", r"$35^\circ N$", r"$36^\circ N$"])

  # Save Plot
  plt.savefig('./plots/QPE_avg_total_precip')
  

if __name__ == '__main__':
  main()

# Consider changing the color scheme and also fix the size? (looks out of proportion)
# Figure out to make multiple plots at once
# Save the files (test before running all files)
