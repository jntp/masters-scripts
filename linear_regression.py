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

# def round_for_text():
# Next function to write!

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

    # Append to lists
    max_refs.append(max_ref)
    peak_Qs.append(peak_Q)
    rainfalls.append(rainfall)
    runoffs.append(runoff)
    runoff_ratios.append(runoff_ratio)

  ## Max Reflectivity vs. Peak Streamflow 
  # Fit linear regression via least squares
  # Returns m (slope) and b (y-intercept)
  # m1, b1 = np.polyfit(max_refs, peak_Qs, 1)

  # Create a sequence of numbers between min max_ref and max_ref
  # maxref_seq = np.linspace(min(max_refs), max(max_refs), 100)
  
  # Get the predictor values for peak streamflow
  # Input max_ref in y=mx+b equation
  # peak_Qs_preds = []

  # for max_ref in max_refs:
    # peak_Qs_pred = m1 * max_ref + b1

    # peak_Qs_preds.append(peak_Qs_pred)

  # Get the R-squared value
  # r_squared1 = r2_score(peak_Qs, peak_Qs_preds)
  # print(r_squared1)

  m1, b1, maxref_seq, r_squared1 = get_regression(max_refs, peak_Qs, min(max_refs), max(max_refs))

  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs, peak_Qs, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq, m1 * maxref_seq + b1, color = "k", lw = 2.5)

  # Round variables for better text display
  m1_round = round(m1, 3)
  b1_round = round(b1, 3)
  r_squared1_round = round(r_squared1, 5)

  # Add text showing linear equation and R-squared value
  ax.text(61.5, 550, r"$y = {0}x {1}$" "\n" r"$R^{2} = {3}$".format(m1_round, b1_round, 2, \
    r_squared1_round)) 

  # Add axis labels
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Peak Streamflow "r"$[m^{3}/s]$")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)

  # ax.clear()

  ## Max Reflectivity vs. Runoff Ratio

  plt.show()

  



if __name__ == '__main__':
  main()

# Max Reflectivity vs. Peak Streamflow
# Max Reflectivity vs. Runoff Ratio
# Total Event Rainfall vs. Runoff (Yang et al., 2016)
