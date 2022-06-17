import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_regression(x, y, min_x, max_x):
  # Fit linear regression via least squares
  # Returns m (slope) and b (y-intercept)
  m, b = np.polyfit(x, y, 1)

  # Create a sequence of numbers between min x and max x
  max_x_seq = np.linspace(min_x, max_x, 100)
  
  # Get the predictor values for peak streamflow
  # Input max_ref in y=mx+b equation
  y_preds = []

  for x_val in x:
    y_pred = m * x_val + b

    y_preds.append(y_pred)

  # Get the R-squared value
  r_squared = r2_score(y, y_preds)
  print(r_squared)

  # Return the stats
  return m, b, max_x_seq, r_squared

def round_for_text(m, b, r_squared):
  # Round variables to properly display as text in plot
  m_round = round(m, 3)
  b_round = round(b, 3)
  r_squared_round = round(r_squared, 5)

  return m_round, b_round, r_squared_round


def main():
  ## Load the hydrometeorological data
  # Get the file path
  fp = "./data/NCFR_Stats.csv"

  # Load the data
  entries = pd.read_csv(fp)
  max_refs_str = entries.loc[:, 'Max_Ref']
  peak_Qs_str = entries.loc[:, 'peak_streamflow']
  rainfalls_str = entries.loc[:, 'mean_precip_mm']
  runoffs_str = entries.loc[:, 'runoff_mm']
  runoff_ratios_str = entries.loc[:, 'runoff_ratio']

  ## Organize the data
  # Create empty lists to store convert data types
  max_refs = []
  peak_Qs = []
  rainfalls = []
  runoffs = []
  runoff_ratios = [] 

  # Convert data to int and float
  for i, max_ref_str in enumerate(max_refs_str):
    # If number is "NaN," simply continue to next iteration
    try:
      max_ref = int(max_refs_str[i])
      peak_Q = float(peak_Qs_str[i]) * 0.028316847 # convert from cfs to cms
      rainfall = float(rainfalls_str[i])
      runoff = float(runoffs_str[i])
      runoff_ratio = float(runoff_ratios_str[i])
    except:
      continue

    # Second check for "NaN" because can convert float(NaN) to NaN
    if math.isnan(peak_Q) or math.isnan(rainfall) or math.isnan(runoff) or math.isnan(runoff_ratio):
      continue

    # Append to lists
    max_refs.append(max_ref)
    peak_Qs.append(peak_Q)
    rainfalls.append(rainfall)
    runoffs.append(runoff)
    runoff_ratios.append(runoff_ratio)

  ## Max Reflectivity vs. Peak Streamflow 
  # Fit the linear regression and get stats
  m1, b1, maxref_seq, r_squared1 = get_regression(max_refs, peak_Qs, min(max_refs), max(max_refs))

  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs, peak_Qs, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq, m1 * maxref_seq + b1, color = "k", lw = 2.5)

  # Round variables for better text display
  m1_round, b1_round, r_squared1_round = round_for_text(m1, b1, r_squared1)

  # Add text showing linear equation and R-squared value
  ax.text(61.5, 550, r"$y = {0}x {1}$" "\n" r"$R^{2} = {3}$".format(m1_round, b1_round, 2, \
    r_squared1_round)) 

  # Add axis labels
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Peak Streamflow "r"$[m^{3}/s]$")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)

  # Save Plot
  plt.savefig('./plots/reflectivity_streamflow.png')

  # Clear axes
  ax.clear()

  ## Max Reflectivity vs. Runoff Ratio
  # Fit the linear regression and get stats
  m2, b2, maxref_seq, r_squared2 = get_regression(max_refs, runoff_ratios, min(max_refs), max(max_refs))

  # Plot the data and linear model
  ax.scatter(max_refs, runoff_ratios, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq, m2 * maxref_seq + b2, color = "k", lw = 2.5)

  # Round variables for better text display
  m2_round, b2_round, r_squared2_round = round_for_text(m2, b2, r_squared2)

  # Add text showing linear equation and R-squared value
  ax.text(61, 1.13, r"$y = {0}x + {1}$" "\n" r"$R^{2} = {3}$".format(m2_round, b2_round, 2, \
    r_squared2_round)) 

  # Add axis labels
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Runoff Ratio")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)

  # Save Plot
  plt.savefig('./plots/reflectivity_runoffratio.png')
  
  # Clear axes
  ax.clear()

  ## Total Event Rainfall vs. Runoff
  # Fit the linear regression and get stats
  m3, b3, rain_seq, r_squared3 = get_regression(rainfalls, runoffs, 0, max(rainfalls))

  # Plot the data and linear model
  ax.scatter(rainfalls, runoffs, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(rain_seq, m3 * rain_seq + b3, color = "k", lw = 2.5)

  # Round variables for better text display
  m3_round, b3_round, r_squared3_round = round_for_text(m3, b3, r_squared3)

  # Add text showing linear equation and R-squared value
  ax.text(54, 52.5, r"$y = {0}x {1}$" "\n" r"$R^{2} = {3}$".format(m3_round, b3_round, 2, \
    r_squared3_round)) 

  # Add axis labels
  ax.set_xlabel("Rainfall (mm)")
  ax.set_ylabel("Runoff (mm)")

  # Save Plot
  plt.savefig('./plots/rainfall_runoff')


if __name__ == '__main__':
  main()
