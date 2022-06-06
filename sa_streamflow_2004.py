import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def convert_str_to_dt(datetime_str):
  # Create slice objects for parsing string
  year_slc = slice(0, 4)
  month_slc = slice(5, 7)
  day_slc = slice(8, 10)
  hour_slc = slice(11, 13)
  min_slc = slice(14, 16)

  # Parse string accordingly
  year = int(datetime_str[year_slc])
  month = int(datetime_str[month_slc])
  day = int(datetime_str[day_slc])
  hour = int(datetime_str[hour_slc]) + 8 # add 8 hrs to convert to UTC
  minute = int(datetime_str[min_slc])

  # Check if the UTC time goes past midnight; adjust the date if that's the case
  if hour >= 24:
    day += 1 
    hour -= 24

  # Create datetime object
  datetime = dt.datetime(year, month, day, hour, minute)

  return datetime


def main():
  # Load the streamflow data
  # Get the file path
  santa_ana_fp = "./data/santa_ana_15min_discharge_1995_2020.csv"

  # Load the streamflow data
  entries = pd.read_csv(santa_ana_fp) 
  datetimes_str = entries.loc[348547:348613, "datetime"] 
  discharges_cfs_str = entries.loc[348547:348613, "discharge_cfs"]
  print(datetimes_str)

  # Create empty lists that will hold converted datetimes and discharge data
  datetimes = [] 
  discharges_cms = []

  # Convert values from string to datetime and float
  for i, datetime_str in enumerate(datetimes_str):
    # Add to offset indices
    i += 348547

    # Convert discharge from cfs to cms
    discharge_cms = float(discharges_cfs_str[i]) * 0.02832

    # Append to lists
    datetimes.append(convert_str_to_dt(datetimes_str[i]))
    discharges_cms.append(discharge_cms)

  # Values of interest; will be plotted later
  initial_Q = 15.859200000000001
  initial_time = dt.datetime(2005, 4, 28, 13, 45)
  peak_Q = 135.936
  peak_time = dt.datetime(2005, 4, 28, 16, 15) 

  # Adjust the plot size
  plt.rcParams["figure.figsize"] = [7.00, 3.50]
  plt.rcParams["figure.autolayout"] = True

  # Create subplot
  fig, ax = plt.subplots() 

  # Format the datetimes on the x axis
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
  ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))

  # Plot the lines and markers
  ax.plot(datetimes, discharges_cms, color = "blue")

  ax.axhline(y = 107, label = "Flood Threshold", color = "red", linestyle = "--", alpha = 0.5)

  ax.plot(initial_time, initial_Q, "gs", label = "NCFR Intersection")
  ax.plot(peak_time, peak_Q, "m^", label = "Peak Reading")
 
  # Add legend
  ax.legend()

  # Add axis labels and title
  plt.xlabel("Time [hh:mm]")
  plt.ylabel("Streamflow "r"$[m^{3}/s]$")
  plt.title("28 April 2005 Observed Streamflow in the Santa Ana River Watershed") 

  # Save the plot
  plt.savefig('./plots/sa_streamflow_2004.png')

  plt.show()

if __name__ == '__main__':
  main()
