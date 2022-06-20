import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def organize_nan_data(df_prop):
  """
    Checks for "NaN" in data and returns lists of indices without "NaN" values.

    Parameters:
    df_prop - pandas dataframe cropped to size of propagation statistics dataframes (30 entries)
  """
  # Create a new list, soon to be array, that will store updated indices
  new_inds = [] # hydro file consisting of 94 entries
  prop_inds = [] # meteo file consisting of 30 entries

  # Check for "NaN" in streamflow; update the new_inds list of indices where streamflow is not "NaN"
  j = 0 # start counting for "prop_inds"

  for i in df_prop.index:
    if not math.isnan(df_prop[i]):
      new_inds.append(i)
      prop_inds.append(j)

    j += 1

  # Convert to numpy array for index operations
  new_inds = np.array(new_inds)
  prop_inds = np.array(prop_inds)

  return new_inds, prop_inds

def run_multi_regression(max_refs_prop, azimuth_prop, speed_prop, response_prop):
  # Set the values for X and y
  X = np.zeros((len(speed_prop), 3))
  X[:, 0] = max_refs_prop
  X[:, 1] = azimuth_prop
  X[:, 2] = speed_prop
  y = response_prop

  # Split the dataset into training and testing data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

  # Fit the multiple linear regression model
  mlr = LinearRegression()
  mlr.fit(X_train, y_train)

  print("Intercept: ", mlr.intercept_)
  print("Coefficients: ", mlr.coef_)

  # Round the intercepts and coefficients
  intercept = round(mlr.intercept_, 3)
  coef_ref = round(mlr.coef_[0], 3)
  coef_azu = round(mlr.coef_[1], 3)
  coef_vel = round(mlr.coef_[2], 3)

  # Don't need to include eqtn str in function; delete later
  eqtn_str = "Q = " + str(intercept) + " + " + str(coef_ref) + "*R + " + str(coef_azu) + \
      "*a + " + str(coef_vel) + "*v + e"
  print(eqtn_str)

  # Pass values of X_test to get the predicted y values
  y_pred_mlr = mlr.predict(X_test)

  # Get the R-squared value
  r_squared = mlr.score(X, y)
  print("R Squared: ", r_squared)

  return X, y, intercept, coef_ref, coef_azu, coef_vel, r_squared

def run_LA_regression(azimuth_prop, speed_prop, response_prop, max_ref_prop = []):
  """
    Implements multiple regression using matrix multiplications (i.e. Linear Algebra or 'LA')

    Returns:
    a - coefficients of multiple linear regression equation
  """
  # Create a modified X array
  if len(max_ref_prop) != 0:
    X1 = np.ones((len(speed_prop), 4))
    X1[:, 0] = max_ref_prop
    X1[:, 1] = azimuth_prop
    X1[:, 2] = speed_prop
  else:
    X1 = np.ones((len(speed_prop), 3))
    X1[:, 0] = azimuth_prop
    X1[:, 1] = speed_prop

  # Create a y array for calculations
  y = response_prop

  # Use Linear Algebra to solve
  a = np.linalg.solve(np.dot(X1.T, X1), np.dot(X1.T, y))
  predictedY = np.dot(X1, a)

  # Calculate the R-squared value
  SSres = y - predictedY
  SStot = y - y.mean()
  r_squared1 = 1 - (SSres.dot(SSres) / SStot.dot(SStot))
  print("R Squared (LA): ", r_squared1)
  print("Coefficients (LA): ", a)

  # Round the coefficients and intercept
  for i, val in enumerate(a): 
    a[i] = round(val, 3)

  # Round the R-squared value
  r_squared1 = round(r_squared1, 5)

  return X1, y, predictedY, r_squared1, a

def create_3D_mesh(X1, a):
  # Create a wiremesh for the plane where the predicted values will lie
  xx, yy, zz = np.meshgrid(X1[:, 0], X1[:, 1], X1[:, 2])
  combinedArrays = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
  Z = combinedArrays.dot(a) 
 
  return combinedArrays, Z


def main():
  ### Multiple Regression Stats (Four Variables)
  ## Load the hydrometerological data
  hydro_fp = "./data/NCFR_Stats.csv"
  met_fp = "./data/NCFR_Propagation_Stats_MR.csv"

  # Load the hydrological data
  hydro_entries = pd.read_csv(hydro_fp)
  hydro_inds = hydro_entries.loc[:, 'Index']
  max_refs = hydro_entries.loc[:, 'Max_Ref']
  peak_Qs_SP = hydro_entries.loc[:, 'peak_Q_SP']
  peak_Qs_WN = hydro_entries.loc[:, 'peak_Q_WN']
  peak_Qs_SA = hydro_entries.loc[:, 'peak_Q_SA']
  peak_Qs_SD = hydro_entries.loc[:, 'peak_Q_SD']
  run_ratios_SP = hydro_entries.loc[:, 'runoff_ratio_SP']
  run_ratios_WN = hydro_entries.loc[:, 'runoff_ratio_WN']
  run_ratios_SA = hydro_entries.loc[:, 'runoff_ratio_SA']
  run_ratios_SD = hydro_entries.loc[:, 'runoff_ratio_SD']

  # Load the meteorological data
  met_entries = pd.read_csv(met_fp)
  met_inds = met_entries.loc[:, 'Index']
  azimuth = met_entries.loc[:, 'Azimuth_deg']
  speed = met_entries.loc[:, 'Speed_ms']

  ## Organize the Data
  # Obtain peak streamflow of entries that have NCFR Propagation Stats
  peak_Qs_SP_prop = peak_Qs_SP[met_inds]
  peak_Qs_WN_prop = peak_Qs_WN[met_inds]
  peak_Qs_SA_prop = peak_Qs_SA[met_inds]
  peak_Qs_SD_prop = peak_Qs_SD[met_inds] 

  # Obtain runoff ratio of entries that have NCFR Propagation Stats
  run_ratios_SP_prop = run_ratios_SP[met_inds]
  run_ratios_WN_prop = run_ratios_WN[met_inds]
  run_ratios_SA_prop = run_ratios_SA[met_inds]
  run_ratios_SD_prop = run_ratios_SD[met_inds]

  # Check for "NaN" values and return fresh lists of indices without "NaN"  
  new_inds1, prop_inds1 = organize_nan_data(peak_Qs_SP_prop)
  new_inds2, prop_inds2 = organize_nan_data(peak_Qs_WN_prop)
  new_inds3, prop_inds3 = organize_nan_data(peak_Qs_SA_prop)
  new_inds4, prop_inds4 = organize_nan_data(peak_Qs_SD_prop)

  new_inds5, prop_inds5 = organize_nan_data(run_ratios_SP_prop)
  new_inds6, prop_inds6 = organize_nan_data(run_ratios_WN_prop)
  new_inds7, prop_inds7 = organize_nan_data(run_ratios_SA_prop)
  new_inds8, prop_inds8 = organize_nan_data(run_ratios_SD_prop)

  # Obtain updated values
  peak_Qs_SP_prop = peak_Qs_SP_prop[new_inds1] * 0.028316847 # convert from cfs to cms
  max_refs_SP_prop = max_refs[new_inds1]
  azimuth_SP_prop = azimuth[prop_inds1]
  speed_SP_prop = speed[prop_inds1]

  peak_Qs_WN_prop = peak_Qs_WN_prop[new_inds2] * 0.028316847 # convert from cfs to cms
  max_refs_WN_prop = max_refs[new_inds2]
  azimuth_WN_prop = azimuth[prop_inds2]
  speed_WN_prop = speed[prop_inds2]

  peak_Qs_SA_prop = peak_Qs_SA_prop[new_inds3] * 0.028316847 # convert from cfs to cms
  max_refs_SA_prop = max_refs[new_inds3]
  azimuth_SA_prop = azimuth[prop_inds3]
  speed_SA_prop = speed[prop_inds3]

  peak_Qs_SD_prop = peak_Qs_SD_prop[new_inds4] * 0.028316847 # convert from cfs to cms
  max_refs_SD_prop = max_refs[new_inds4]
  azimuth_SD_prop = azimuth[prop_inds4]
  speed_SD_prop = speed[prop_inds4]

  run_ratios_SP_prop = run_ratios_SP_prop[new_inds5]
  max_refs_SP_prop2 = max_refs[new_inds5]
  azimuth_SP_prop2 = azimuth[prop_inds5]
  speed_SP_prop2 = speed[prop_inds5]

  run_ratios_WN_prop = run_ratios_WN_prop[new_inds6]
  max_refs_WN_prop2 = max_refs[new_inds6]
  azimuth_WN_prop2 = azimuth[prop_inds6]
  speed_WN_prop2 = speed[prop_inds6]

  run_ratios_SA_prop = run_ratios_SA_prop[new_inds7]
  max_refs_SA_prop2 = max_refs[new_inds7]
  azimuth_SA_prop2 = azimuth[prop_inds7]
  speed_SA_prop2 = speed[prop_inds7]

  run_ratios_SD_prop = run_ratios_SD_prop[new_inds8]
  max_refs_SD_prop2 = max_refs[new_inds8]
  azimuth_SD_prop2 = azimuth[prop_inds8]
  speed_SD_prop2 = speed[prop_inds8]

  ## Implement Multiple Linear Regression
  X1, y1, predictedY1, r_squared1, a1 = run_LA_regression(azimuth_SP_prop, speed_SP_prop, peak_Qs_SP_prop, \
      max_refs_SP_prop) 
  X2, y2, predictedY2, r_squared2, a2 = run_LA_regression(azimuth_WN_prop, speed_WN_prop, peak_Qs_WN_prop, \
      max_refs_WN_prop)
  X3, y3, predictedY3, r_squared3, a3 = run_LA_regression(azimuth_SA_prop, speed_SA_prop, peak_Qs_SA_prop, \
      max_refs_SA_prop)
  X4, y4, predictedY4, r_squared4, a4 = run_LA_regression(azimuth_SD_prop, speed_SD_prop, peak_Qs_SD_prop, \
      max_refs_SD_prop)

  X5, y5, predictedY5, r_squared5, a5 = run_LA_regression(azimuth_SP_prop2, speed_SP_prop2, run_ratios_SP_prop, \
      max_refs_SP_prop2)
  X6, y6, predictedY6, r_squared6, a6 = run_LA_regression(azimuth_WN_prop2, speed_WN_prop2, run_ratios_WN_prop, \
      max_refs_WN_prop2)
  X7, y7, predictedY7, r_squared7, a7 = run_LA_regression(azimuth_SA_prop2, speed_SA_prop2, run_ratios_SA_prop, \
      max_refs_SA_prop2)
  X8, y8, predictedY8, r_squared8, a8 = run_LA_regression(azimuth_SD_prop2, speed_SD_prop2, run_ratios_SD_prop, \
      max_refs_SD_prop2)

  print("\n") # New line for readability

  ### 3D Plots
  ## Storm speed and Direction vs. Peak Discharge
  X11, y1, predictedY11, r_squared11, a11 = run_LA_regression(azimuth_SP_prop, speed_SP_prop, peak_Qs_SP_prop)
  combinedArrays1, Z1 = create_3D_mesh(X11, a11)
  X21, y2, predictedY21, r_squared21, a21 = run_LA_regression(azimuth_WN_prop, speed_WN_prop, peak_Qs_WN_prop)
  combinedArrays2, Z2 = create_3D_mesh(X21, a21)
  X31, y3, predictedY31, r_squared31, a31 = run_LA_regression(azimuth_SA_prop, speed_SA_prop, peak_Qs_SA_prop)
  combinedArrays3, Z3 = create_3D_mesh(X31, a31)
  X41, y4, predictedY41, r_squared41, a41 = run_LA_regression(azimuth_SD_prop, speed_SD_prop, peak_Qs_SD_prop)
  combinedArrays4, Z4 = create_3D_mesh(X41, a41)

  # Graph everything together
  ## Sepulveda Dam
  # Create fig and axes
  fig = plt.figure(1)
  ax = fig.add_subplot(111, projection = '3d')
  
  # Plot the data points and wire mesh
  ax.scatter(X11[:, 0], X11[:, 1], y1, color = "r", label = "Actual Streamflow")
  ax.scatter(X11[:, 0], X11[:, 1], predictedY11, color = "g", label = "Predicted Streamflow")
  ax.plot_trisurf(combinedArrays1[:, 0], combinedArrays1[:, 1], Z1, alpha = 0.5)
  
  # Add labels, legend, and title
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.set_zlim(0) # Set the minimum value on z axis to 0
  ax.legend()
  ax.set_title("Sepulveda Dam")

  # Add R-squared value and equation to plot
  ax.text(-145, 5, 535, r"$Q = {0} {1}*a {2}*v + e$" "\n" r"$R^{3} = {4}$".format(a11[2], a11[0], \
      a11[1], 2, r_squared11))
 
  # Set the default viewing position for 3D plot
  ax.view_init(elev = 14, azim = -49)

  # Save Plot
  plt.savefig('./plots/3D_streamflow_SP')

  ## Whittier Narrows Dam
  fig = plt.figure(2)
  ax = fig.add_subplot(111, projection = '3d')
  
  # Plot the data points and wire mesh
  ax.scatter(X21[:, 0], X21[:, 1], y2, color = "r", label = "Actual Streamflow")
  ax.scatter(X21[:, 0], X21[:, 1], predictedY21, color = "g", label = "Predicted Streamflow")
  ax.plot_trisurf(combinedArrays2[:, 0], combinedArrays2[:, 1], Z2, alpha = 0.5)
  
  # Add labels, legend, and title
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.set_zlim(0) # Set the minimum value on z axis to 0
  ax.legend()
  ax.set_title("Whittier Narrows Dam")

  # Add R-squared value and equation to plot
  ax.text(-145, 5, 235, r"$Q = {0} {1}*a {2}*v + e$" "\n" r"$R^{3} = {4}$".format(a21[2], a21[0], \
      a21[1], 2, r_squared21))
 
  # Set the default viewing position for 3D plot
  ax.view_init(elev = 14, azim = -49)

  # Save Plot
  plt.savefig('./plots/3D_streamflow_WN')

  ## Santa Ana River
  fig = plt.figure(3)
  ax = fig.add_subplot(111, projection = '3d')
  
  # Plot the data points and wire mesh
  ax.scatter(X31[:, 0], X31[:, 1], y3, color = "r", label = "Actual Streamflow")
  ax.scatter(X31[:, 0], X31[:, 1], predictedY31, color = "g", label = "Predicted Streamflow")
  ax.plot_trisurf(combinedArrays3[:, 0], combinedArrays3[:, 1], Z3, alpha = 0.5)
  
  # Add labels, legend, and title
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.set_zlim(0) # Set the minimum value on z axis to 0
  ax.legend()
  ax.set_title("Santa Ana River")

  # Add R-squared value and equation to plot
  ax.text(-145, 5, 405, r"$Q = {0} + {1}*a + {2}*v + e$" "\n" r"$R^{3} = {4}$".format(a31[2], a31[0], \
      a31[1], 2, r_squared31))
 
  # Set the default viewing position for 3D plot
  ax.view_init(elev = 14, azim = -49)

  # Save Plot
  plt.savefig('./plots/3D_streamflow_SA')

  ## San Diego River
  fig = plt.figure(4)
  ax = fig.add_subplot(111, projection = '3d')
  
  # Plot the data points and wire mesh
  ax.scatter(X41[:, 0], X41[:, 1], y4, color = "r", label = "Actual Streamflow")
  ax.scatter(X41[:, 0], X41[:, 1], predictedY41, color = "g", label = "Predicted Streamflow")
  ax.plot_trisurf(combinedArrays4[:, 0], combinedArrays4[:, 1], Z4, alpha = 0.5)
  
  # Add labels, legend, and title
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.set_zlim(0) # Set the minimum value on z axis to 0
  ax.legend()
  ax.set_title("San Diego River")

  # Add R-squared value and equation to plot
  ax.text(-145, 5, 115, r"$Q = {0} + {1}*a + {2}*v + e$" "\n" r"$R^{3} = {4}$".format(a41[2], a41[0], \
      a41[1], 2, r_squared41))
 
  # Set the default viewing position for 3D plot
  ax.view_init(elev = 14, azim = -49)

  # Save Plot
  plt.savefig('./plots/3D_streamflow_SD')

  ## Storm Speed and Direction vs. Runoff Ratio
  X51, y5, predictedY51, r_squared51, a51 = run_LA_regression(azimuth_SP_prop2, speed_SP_prop2, run_ratios_SP_prop)
  combinedArrays5, Z5 = create_3D_mesh(X51, a51)
  X61, y6, predictedY61, r_squared61, a61 = run_LA_regression(azimuth_WN_prop2, speed_WN_prop2, run_ratios_WN_prop)
  combinedArrays6, Z6 = create_3D_mesh(X61, a61)
  X71, y7, predictedY71, r_squared71, a71 = run_LA_regression(azimuth_SA_prop2, speed_SA_prop2, run_ratios_SA_prop)
  combinedArrays7, Z7 = create_3D_mesh(X71, a71)
  X81, y8, predictedY81, r_squared81, a81 = run_LA_regression(azimuth_SD_prop2, speed_SD_prop2, run_ratios_SD_prop)
  combinedArrays8, Z8 = create_3D_mesh(X81, a81)

  ## Sepulveda Dam 
  # Create fig and axes
  fig = plt.figure(5)
  ax = fig.add_subplot(111, projection = '3d')
  
  # Plot the data points and wire mesh
  ax.scatter(X51[:, 0], X51[:, 1], y5, color = "r", label = "Actual Runoff Ratio")
  ax.scatter(X51[:, 0], X51[:, 1], predictedY51, color = "g", label = "Predicted Runoff Ratio")
  ax.plot_trisurf(combinedArrays5[:, 0], combinedArrays5[:, 1], Z5, alpha = 0.5)
  
  # Add labels, legend, and title
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Runoff Ratio")
  ax.set_zlim(0) # Set the minimum value on z axis to 0
  ax.legend()
  ax.set_title("Sepulveda Dam")

  # Add R-squared value and equation to plot
  ax.text(-145, 5, 5, r"$Q = {0} + {1}*a {2}*v + e$" "\n" r"$R^{3} = {4}$".format(a51[2], a51[0], \
      a51[1], 2, r_squared51))
 
  # Set the default viewing position for 3D plot
  ax.view_init(elev = 14, azim = -49)

  # Save Plot
  plt.savefig('./plots/3D_runoff_ratio_SP')

  ## Whittier Narrows Dam 
  # Create fig and axes
  fig = plt.figure(6)
  ax = fig.add_subplot(111, projection = '3d')
  
  # Plot the data points and wire mesh
  ax.scatter(X61[:, 0], X61[:, 1], y6, color = "r", label = "Actual Runoff Ratio")
  ax.scatter(X61[:, 0], X61[:, 1], predictedY61, color = "g", label = "Predicted Runoff Ratio")
  ax.plot_trisurf(combinedArrays6[:, 0], combinedArrays6[:, 1], Z6, alpha = 0.5)
  
  # Add labels, legend, and title
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Runoff Ratio")
  ax.set_zlim(0) # Set the minimum value on z axis to 0
  ax.legend()
  ax.set_title("Whittier Narrows Dam")

  # Add R-squared value and equation to plot
  ax.text(-125, 7, 3.25, r"$Q = {0} + {1}*a {2}*v + e$" "\n" r"$R^{3} = {4}$".format(a61[2], a61[0], \
      a61[1], 2, r_squared61))
 
  # Set the default viewing position for 3D plot
  ax.view_init(elev = 14, azim = -49)

  # Save Plot
  plt.savefig('./plots/3D_runoff_ratio_WN')

  ## Santa Ana River
  # Create fig and axes
  fig = plt.figure(7)
  ax = fig.add_subplot(111, projection = '3d')
  
  # Plot the data points and wire mesh
  ax.scatter(X71[:, 0], X71[:, 1], y7, color = "r", label = "Actual Runoff Ratio")
  ax.scatter(X71[:, 0], X71[:, 1], predictedY71, color = "g", label = "Predicted Runoff Ratio")
  ax.plot_trisurf(combinedArrays7[:, 0], combinedArrays7[:, 1], Z7, alpha = 0.5)
  
  # Add labels, legend, and title
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Runoff Ratio")
  ax.set_zlim(0) # Set the minimum value on z axis to 0
  ax.legend()
  ax.set_title("Santa Ana River")

  # Add R-squared value and equation to plot
  ax.text(-125, 7, 0.075, r"$Q = {0} + {1}*a + {2}*v + e$" "\n" r"$R^{3} = {4}$".format(a71[2], a71[0], \
      a71[1], 2, r_squared71))
 
  # Set the default viewing position for 3D plot
  ax.view_init(elev = 14, azim = -49)

  # Save Plot
  plt.savefig('./plots/3D_runoff_ratio_SA')

  ## San Diego River
  # Create fig and axes
  fig = plt.figure(8)
  ax = fig.add_subplot(111, projection = '3d')
  
  # Plot the data points and wire mesh
  ax.scatter(X81[:, 0], X81[:, 1], y8, color = "r", label = "Actual Runoff Ratio")
  ax.scatter(X81[:, 0], X81[:, 1], predictedY81, color = "g", label = "Predicted Runoff Ratio")
  ax.plot_trisurf(combinedArrays8[:, 0], combinedArrays8[:, 1], Z8, alpha = 0.5)
  
  # Add labels, legend, and title
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Runoff Ratio")
  ax.set_zlim(0) # Set the minimum value on z axis to 0
  ax.legend()
  ax.set_title("San Diego River")

  # Add R-squared value and equation to plot
  ax.text(-125, 7, 0.175, r"$Q = {0} + {1}*a + {2}*v + e$" "\n" r"$R^{3} = {4}$".format(a81[2], a81[0], \
      a81[1], 2, r_squared81))
 
  # Set the default viewing position for 3D plot
  ax.view_init(elev = 14, azim = -49)

  # Save Plot
  plt.savefig('./plots/3D_runoff_ratio_SD')

  # plt.show()

if __name__ == '__main__':
  main()
