import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def organize_nan_data(df):
  """
    Checks for "NaN" in data and returns lists of indices without "NaN" values.

    Parameters:
    df - pandas dataframe
  """
  # Create a new list, soon to be array, that will store updated indices
  new_inds = [] 

  # Check for "NaN" in the data
  for i in df.index:
    if not math.isnan(df[i]):
      new_inds.append(i)

  # Convert to numpy array for index operations
  new_inds = np.array(new_inds)

  return new_inds

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

def run_LA_regression(predictor, response, min_pred, max_pred):
  """
    Implements multiple regression using matrix multiplications (i.e. Linear Algebra or 'LA')

    Returns:
    a - coefficients of multiple linear regression equation
  """
  # Create a modified X array
  X = np.ones((len(predictor), 2))
  X[:, 0] = predictor

  # Create a y array for calculations
  y = response

  # Create a sequence of numbers between the minimum and maximum predictor values
  pred_seq = np.linspace(min_pred, max_pred, 100)

  # Use Linear Algebra to solve
  a = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
  predictedY = np.dot(X, a)

  # Calculate the R-squared value
  SSres = y - predictedY
  SStot = y - y.mean()
  r_squared = 1 - (SSres.dot(SSres) / SStot.dot(SStot))
  print("R Squared (LA): ", r_squared)
  print("Coefficients (LA): ", a)

  # Set to variables of y=mx+b equation
  m = a[0]
  b = a[1]

  return m, b, r_squared, pred_seq

def round_for_text(m, b, r_squared):
  # Round variables to properly display as text in plot
  m_round = round(m, 3)
  b_round = round(b, 3)
  r_squared_round = round(r_squared, 5)

  return m_round, b_round, r_squared_round


def main():
  ## Load the hydrometeorological data
  # Get the file path
  fp = "./data/NCFR_Stats2.csv"

  # Load the data
  entries = pd.read_csv(fp)
  max_refs = entries.loc[:, 'Max_Ref']
  peak_Qs_SP = entries.loc[:, 'peak_Q_SP']
  peak_Qs_WN = entries.loc[:, 'peak_Q_WN']
  peak_Qs_SA = entries.loc[:, 'peak_Q_SA']
  peak_Qs_SD = entries.loc[:, 'peak_Q_SD']
  precip_SP = entries.loc[:, 'mean_precip_SP']
  precip_WN = entries.loc[:, 'mean_precip_WN']
  precip_SA = entries.loc[:, 'mean_precip_SA']
  precip_SD = entries.loc[:, 'mean_precip_SD']
  runoffs_SP = entries.loc[:, 'runoff_SP']
  runoffs_WN = entries.loc[:, 'runoff_WN']
  runoffs_SA = entries.loc[:, 'runoff_SA']
  runoffs_SD = entries.loc[:, 'runoff_SD']
  run_ratios_SP = entries.loc[:, 'runoff_ratio_SP']
  run_ratios_WN = entries.loc[:, 'runoff_ratio_WN']
  run_ratios_SA = entries.loc[:, 'runoff_ratio_SA']
  run_ratios_SD = entries.loc[:, 'runoff_ratio_SD']

  ## Filter out data with "NaN" values (Two Step Process)
  ref_inds = organize_nan_data(max_refs) # max_refs also has the "reference" indices

  # Peak Streamflow
  inds_Q_SP = organize_nan_data(peak_Qs_SP[ref_inds])
  inds_Q_WN = organize_nan_data(peak_Qs_WN[ref_inds])
  inds_Q_SA = organize_nan_data(peak_Qs_SA[ref_inds])
  inds_Q_SD = organize_nan_data(peak_Qs_SD[ref_inds])

  # Runoff Ratio
  inds_r_SP = organize_nan_data(run_ratios_SP[ref_inds])
  inds_r_WN = organize_nan_data(run_ratios_WN[ref_inds])
  inds_r_SA = organize_nan_data(run_ratios_SA[ref_inds])
  inds_r_SD = organize_nan_data(run_ratios_SD[ref_inds])

  # Get data with "NaN" filtered out
  max_refs_SP0 = max_refs[inds_Q_SP]
  peak_Qs_SP0 = peak_Qs_SP[inds_Q_SP] * 0.028316847 # convert from cfs to cms

  max_refs_WN0 = max_refs[inds_Q_WN]
  peak_Qs_WN0 = peak_Qs_WN[inds_Q_WN] * 0.028316847 # convert from cfs to cms

  max_refs_SA0 = max_refs[inds_Q_SA]
  peak_Qs_SA0 = peak_Qs_SA[inds_Q_SA] * 0.028316847 # convert from cfs to cms

  max_refs_SD0 = max_refs[inds_Q_SD]
  peak_Qs_SD0 = peak_Qs_SD[inds_Q_SD] * 0.028316847 # convert from cfs to cms

  max_refs_SP01 = max_refs[inds_r_SP]
  run_ratios_SP0 = run_ratios_SP[inds_r_SP]

  max_refs_WN01 = max_refs[inds_r_WN]
  run_ratios_WN0 = run_ratios_WN[inds_r_WN]

  max_refs_SA01 = max_refs[inds_r_SA]
  run_ratios_SA0 = run_ratios_SA[inds_r_SA]

  max_refs_SD01 = max_refs[inds_r_SD]
  run_ratios_SD0 = run_ratios_SD[inds_r_SD]

  # Use runoff ratio indices for rainfall vs. runoff
  precip_SP0 = precip_SP[inds_r_SP]
  runoffs_SP0 = runoffs_SP[inds_r_SP]

  precip_WN0 = precip_WN[inds_r_WN]
  runoffs_WN0 = runoffs_WN[inds_r_WN] 

  precip_SA0 = precip_SA[inds_r_SA]
  runoffs_SA0 = runoffs_SA[inds_r_SA]

  precip_SD0 = precip_SD[inds_r_SD]
  runoffs_SD0 = runoffs_SD[inds_r_SD] 

  ### Max Reflectivity vs. Peak Streamflow 
  # Fit the linear regression and get stats
  m1, b1, r_squared1, maxref_seq1 = run_LA_regression(max_refs_SP0, peak_Qs_SP0, min(max_refs_SP0), \
      max(max_refs_SP0))
  m2, b2, r_squared2, maxref_seq2 = run_LA_regression(max_refs_WN0, peak_Qs_WN0, min(max_refs_WN0), \
      max(max_refs_WN0))
  m3, b3, r_squared3, maxref_seq3 = run_LA_regression(max_refs_SA0, peak_Qs_SA0, min(max_refs_SA0), \
      max(max_refs_SA0))
  m4, b4, r_squared4, maxref_seq4 = run_LA_regression(max_refs_SD0, peak_Qs_SD0, min(max_refs_SD0), \
      max(max_refs_SD0))

  ## Sepulveda Dam
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs_SP0, peak_Qs_SP0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq1, m1 * maxref_seq1 + b1, color = "k", lw = 2.5)

  # Round variables for better text display
  m1_round, b1_round, r_squared1_round = round_for_text(m1, b1, r_squared1)

  # Add text showing linear equation and R-squared value
  ax.text(61.5, 550, r"$Q = {0}*Z {1}$" "\n" r"$R^{2} = {3}$".format(m1_round, b1_round, 2, \
    r_squared1_round)) 

  # Add axis labels and title
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.set_title("Sepulveda Dam", horizontalalignment = "right")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)

  # Save Plot
  plt.savefig('./plots/reflectivity_streamflow_SP2.png')

  # Clear axes
  ax.clear()

  ## Whittier Narrows Dam
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs_WN0, peak_Qs_WN0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq2, m2 * maxref_seq2 + b2, color = "k", lw = 2.5)

  # Round variables for better text display
  m2_round, b2_round, r_squared2_round = round_for_text(m2, b2, r_squared2)

  # Add text showing linear equation and R-squared value
  ax.text(61.5, 245, r"$Q = {0}*Z {1}$" "\n" r"$R^{2} = {3}$".format(m2_round, b2_round, 2, \
    r_squared2_round)) 

  # Add axis labels and title
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.set_title("Whittier Narrows Dam", horizontalalignment = "right")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)  

  # Save Plot
  plt.savefig('./plots/reflectivity_streamflow_WN2.png')

  # Clear axes
  ax.clear()

  ## Santa Ana River
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs_SA0, peak_Qs_SA0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq3, m3 * maxref_seq3 + b3, color = "k", lw = 2.5)

  # Round variables for better text display
  m3_round, b3_round, r_squared3_round = round_for_text(m3, b3, r_squared3)

  # Add text showing linear equation and R-squared value
  ax.text(61.5, 380, r"$Q = {0}*Z {1}$" "\n" r"$R^{2} = {3}$".format(m3_round, b3_round, 2, \
    r_squared3_round)) 

  # Add axis labels and title
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.set_title("Santa Ana River", horizontalalignment = "right")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks) 

  # Save Plot
  plt.savefig('./plots/reflectivity_streamflow_SA2.png')

  # Clear axes
  ax.clear()

  ## San Diego River
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs_SD0, peak_Qs_SD0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq4, m4 * maxref_seq4 + b4, color = "k", lw = 2.5)

  # Round variables for better text display
  m4_round, b4_round, r_squared4_round = round_for_text(m4, b4, r_squared4)

  # Add text showing linear equation and R-squared value
  ax.text(61.5, 108, r"$Q = {0}*Z {1}$" "\n" r"$R^{2} = {3}$".format(m4_round, b4_round, 2, \
    r_squared4_round)) 

  # Add axis labels and title
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.set_title("San Diego River", horizontalalignment = "right")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks) 

  # Save Plot
  plt.savefig('./plots/reflectivity_streamflow_SD2.png')

  # Clear axes
  ax.clear()

  print("\n") # new line for readability on command prompt

  ### Max Reflectivity vs. Runoff Ratio
  # Fit the linear regression and get stats
  m5, b5, r_squared5, maxref_seq5 = run_LA_regression(max_refs_SP01, run_ratios_SP0, min(max_refs_SP01), \
      max(max_refs_SP01))
  m6, b6, r_squared6, maxref_seq6 = run_LA_regression(max_refs_WN01, run_ratios_WN0, min(max_refs_WN01), \
      max(max_refs_WN01))
  m7, b7, r_squared7, maxref_seq7 = run_LA_regression(max_refs_SA01, run_ratios_SA0, min(max_refs_SA01), \
      max(max_refs_SA01))
  m8, b8, r_squared8, maxref_seq8 = run_LA_regression(max_refs_SD01, run_ratios_SD0, min(max_refs_SD01), \
      max(max_refs_SD01))

  ## Sepulveda Dam
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs_SP01, run_ratios_SP0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq5, m5 * maxref_seq5 + b5, color = "k", lw = 2.5)

  # Round variables for better text display
  m5_round, b5_round, r_squared5_round = round_for_text(m5, b5, r_squared5)

  # Add text showing linear equation and R-squared value
  ax.text(62, 4.65, r"$r = {0}*Z {1}$" "\n" r"$R^{2} = {3}$".format(m5_round, b5_round, 2, \
    r_squared5_round)) 

  # Add axis labels and title
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Runoff Ratio")
  ax.set_title("Sepulveda Dam", horizontalalignment = "right")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)

  # Save Plot
  plt.savefig('./plots/reflectivity_runoffratio_SP2.png')
  
  # Clear axes
  ax.clear()

  ## Whittier Narrows Dam
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs_WN01, run_ratios_WN0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq6, m6 * maxref_seq6 + b6, color = "k", lw = 2.5)

  # Round variables for better text display
  m6_round, b6_round, r_squared6_round = round_for_text(m6, b6, r_squared6)

  # Add text showing linear equation and R-squared value
  ax.text(62, 3.08, r"$r = {0}*Z + {1}$" "\n" r"$R^{2} = {3}$".format(m6_round, b6_round, 2, \
    r_squared6_round)) 

  # Add axis labels and title
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Runoff Ratio")
  ax.set_title("Whittier Narrows Dam", horizontalalignment = "right")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)

  # Save Plot
  plt.savefig('./plots/reflectivity_runoffratio_WN2.png')
  
  # Clear axes
  ax.clear()

  ## Santa Ana River
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs_SA01, run_ratios_SA0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq7, m7 * maxref_seq7 + b7, color = "k", lw = 2.5)

  # Round variables for better text display
  m7_round, b7_round, r_squared7_round = round_for_text(m7, b7, r_squared7)

  # Add text showing linear equation and R-squared value
  ax.text(62, 1.08, r"$r = {0}*Z + {1}$" "\n" r"$R^{2} = {3}$".format(m7_round, b7_round, 2, \
    r_squared7_round)) 

  # Add axis labels and title
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Runoff Ratio")
  ax.set_title("Santa Ana River", horizontalalignment = "right")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)

  # Save Plot
  plt.savefig('./plots/reflectivity_runoffratio_SA2.png')
  
  # Clear axes
  ax.clear()

  ## San Diego River
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 

  # Plot the data and linear model
  ax.scatter(max_refs_SD01, run_ratios_SD0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(maxref_seq8, m8 * maxref_seq8 + b8, color = "k", lw = 2.5)

  # Round variables for better text display
  m8_round, b8_round, r_squared8_round = round_for_text(m8, b8, r_squared8)

  # Add text showing linear equation and R-squared value
  ax.text(62, 1.65, r"$r = {0}*Z + {1}$" "\n" r"$R^{2} = {3}$".format(m8_round, b8_round, 2, \
    r_squared8_round)) 

  # Add axis labels and title
  ax.set_xlabel("Max Reflectivity (dbZ)")
  ax.set_ylabel("Runoff Ratio")
  ax.set_title("San Diego River", horizontalalignment = "right")

  # Adjust the tick labels on x-axis 
  maxref_ticks = np.arange(50, 70, 2)
  ax.set_xticks(maxref_ticks)

  # Save Plot
  plt.savefig('./plots/reflectivity_runoffratio_SD2.png')
  
  # Clear axes
  ax.clear()

  print("\n") 

  ## Total Event Rainfall vs. Runoff
  # Fit the linear regression and get stats
  m9, b9, r_squared9, precip_seq9 = run_LA_regression(precip_SP0, runoffs_SP0, 0, max(precip_SP0))
  m10, b10, r_squared10, precip_seq10 = run_LA_regression(precip_WN0, runoffs_WN0, 0, max(precip_WN0))
  m11, b11, r_squared11, precip_seq11 = run_LA_regression(precip_SA0, runoffs_SA0, 0, max(precip_SA0))
  m12, b12, r_squared12, precip_seq12 = run_LA_regression(precip_SD0, runoffs_SD0, 0, max(precip_SD0))

  ## Sepulveda Dam
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 
 
  # Plot the data and linear model
  ax.scatter(precip_SP0, runoffs_SP0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(precip_seq9, m9 * precip_seq9 + b9, color = "k", lw = 2.5)

  # Round variables for better text display
  m9_round, b9_round, r_squared9_round = round_for_text(m9, b9, r_squared9)
    
  # Add text showing linear equation and R-squared value
  ax.text(1.3, 26.5, r"$q = {0}*p + {1}$" "\n" r"$R^{2} = {3}$".format(m9_round, b9_round, 2, \
    r_squared9_round)) 

  # Add axis labels and title
  ax.set_xlabel("Rainfall (mm)")
  ax.set_ylabel("Runoff (mm)")
  ax.set_title("Sepulveda Dam", horizontalalignment = "right")

  # Save Plot
  plt.savefig('./plots/rainfall_runoff_SP2')

  # Clear axes
  ax.clear()

  ## Whittier Narrows Dam
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 
 
  # Plot the data and linear model
  ax.scatter(precip_WN0, runoffs_WN0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(precip_seq10, m10 * precip_seq10 + b10, color = "k", lw = 2.5)

  # Round variables for better text display
  m10_round, b10_round, r_squared10_round = round_for_text(m10, b10, r_squared10)

  # Add text showing linear equation and R-squared value
  ax.text(1.05, 13.5, r"$q = {0}*p + {1}$" "\n" r"$R^{2} = {3}$".format(m10_round, b10_round, 2, \
    r_squared10_round)) 

  # Add axis labels and title
  ax.set_xlabel("Rainfall (mm)")
  ax.set_ylabel("Runoff (mm)")
  ax.set_title("Whittier Narrows Dam", horizontalalignment = "right")

  # Save Plot
  plt.savefig('./plots/rainfall_runoff_WN2')

  # Clear axes
  ax.clear()

  ## Santa Ana River
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 
 
  # Plot the data and linear model
  ax.scatter(precip_SA0, runoffs_SA0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(precip_seq11, m11 * precip_seq11 + b11, color = "k", lw = 2.5)

  # Round variables for better text display
  m11_round, b11_round, r_squared11_round = round_for_text(m11, b11, r_squared11)

  # Add text showing linear equation and R-squared value
  ax.text(1.1, 1.92, r"$q = {0}*p + {1}$" "\n" r"$R^{2} = {3}$".format(m11_round, b11_round, 2, \
    r_squared11_round)) 

  # Add axis labels and title
  ax.set_xlabel("Rainfall (mm)")
  ax.set_ylabel("Runoff (mm)")
  ax.set_title("Santa Ana River", horizontalalignment = "right")

  # Save Plot
  plt.savefig('./plots/rainfall_runoff_SA2')

  # Clear axes
  ax.clear()

  ## San Diego River
  # Create a new figure and axes
  fig = plt.figure(figsize = (5, 4))
  ax = plt.axes() 
 
  # Plot the data and linear model
  ax.scatter(precip_SD0, runoffs_SD0, s = 60, alpha = 0.7, edgecolors = "k")
  ax.plot(precip_seq12, m12 * precip_seq12 + b12, color = "k", lw = 2.5)

  # Round variables for better text display
  m12_round, b12_round, r_squared12_round = round_for_text(m12, b12, r_squared12)

  # Add text showing linear equation and R-squared value
  ax.text(31, 3.15, r"$q = {0}*p + {1}$" "\n" r"$R^{2} = {3}$".format(m12_round, b12_round, 2, \
    r_squared12_round)) 

  # Add axis labels and title
  ax.set_xlabel("Rainfall (mm)")
  ax.set_ylabel("Runoff (mm)")
  ax.set_title("San Diego River", horizontalalignment = "right")

  # Save Plot
  plt.savefig('./plots/rainfall_runoff_SD2')


if __name__ == '__main__':
  main()

# Left off formatting rainfall_runoff for SD
# Still have max reflectivity vs runoff ratio for all watersheds
