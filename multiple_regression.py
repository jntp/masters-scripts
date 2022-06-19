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

  # Check for "NaN" values and return fresh lists of indices without "NaN"  
  new_inds1, prop_inds1 = organize_nan_data(peak_Qs_SP_prop)
  new_inds2, prop_inds2 = organize_nan_data(peak_Qs_WN_prop)

  # Obtain updated values
  peak_Qs_SP_prop = peak_Qs_SP_prop[new_inds1] * 0.028316847 # convert from cfs to cms
  max_refs_SP_prop = max_refs[new_inds1]
  azimuth_SP_prop = azimuth[prop_inds1]
  speed_SP_prop = speed[prop_inds1]

  peak_Qs_WN_prop = peak_Qs_WN_prop[new_inds2] * 0.028316847 # convert from cfs to cms
  max_refs_WN_prop = max_refs[new_inds2]
  azimuth_WN_prop = azimuth[prop_inds2]
  speed_WN_prop = speed[prop_inds2]

  ## Implement Multiple Linear Regression
  X1, y1, predictedY1, r_squared1, a1 = run_LA_regression(azimuth_SP_prop, speed_SP_prop, peak_Qs_SP_prop, \
      max_refs_SP_prop) 
  X2, y2, predictedY2, r_squared2, a2 = run_LA_regression(azimuth_WN_prop, speed_WN_prop, peak_Qs_WN_prop, \
      max_refs_WN_prop)

  print("------------------") # Print dividing line for readability

  ### 3D Plots
  ## Storm speed and Direction vs. Peak Discharge
  X11, y1, predictedY11, r_squared11, a11 = run_LA_regression(azimuth_SP_prop, speed_SP_prop, peak_Qs_SP_prop)
  combinedArrays1, Z1 = create_3D_mesh(X11, a11)
  X21, y2, predictedY21, r_squared21, a21 = run_LA_regression(azimuth_WN_prop, speed_WN_prop, peak_Qs_WN_prop)
  combinedArrays2, Z2 = create_3D_mesh(X11, a21)

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
  # plt.savefig('./plots/3D_streamflow_SP')

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
  # plt.savefig('./plots/3D_streamflow_WN')

  plt.show()

if __name__ == '__main__':
  main()



# Max reflectivity, storm speed, storm direction vs. peak discharge
# Max reflectivity, storm speed, storm direction vs. runoff ratio
