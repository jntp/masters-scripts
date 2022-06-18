import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

  # Create a new list, soon to be array, that will store updated indices
  new_inds = [] # hydro file
  prop_inds = [] # meteo file

  # Check for "NaN" in streamflow; update the new_inds list of indices where streamflow is not "NaN"
  j = 0 # start counting for "prop_inds"

  for i in peak_Qs_SP_prop.index:
    if not math.isnan(peak_Qs_SP_prop[i]):
      new_inds.append(i)
      prop_inds.append(j)

    j += 1

  # Convert to numpy array for index operations
  new_inds = np.array(new_inds)
  prop_inds = np.array(prop_inds)
  
  # Obtain updated values
  peak_Qs_SP_prop = peak_Qs_SP_prop[new_inds]
  max_refs_prop = max_refs[new_inds]
  azimuth_prop = azimuth[prop_inds]
  speed_prop = speed[prop_inds]

  ## Implement Multiple Linear Regression
  # Set the values for X and y
  X = np.zeros((len(speed_prop), 3))
  X[:, 0] = max_refs_prop
  X[:, 1] = azimuth_prop
  X[:, 2] = speed_prop
  y = peak_Qs_SP_prop

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

  # Don't need to include eqtn str in function
  eqtn_str = "Q = " + str(intercept) + " + " + str(coef_ref) + "*R + " + str(coef_azu) + \
      "*a + " + str(coef_vel) + "*v + e"
  print(eqtn_str)

  ## Get prediction and evaluate the model
  # Pass values of X_test to get the predicted y values
  y_pred_mlr = mlr.predict(X_test)

  r_squared = mlr.score(X, y)
  print("R Squared: ", r_squared)

  ### 3D Plots
  ## Storm speed and Direction vs. Peak Discharge
  # Graph the data
  fig = plt.figure(1)
  ax = fig.add_subplot(111, projection = '3d')
  ax.scatter(X[:, 1], X[:, 2], y)
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Peak Streamflow "r"$[m^{3}/s]$")

  # Create a modified X array
  X1 = np.ones((len(speed_prop), 3))
  X1[:, 0] = azimuth_prop
  X1[:, 1] = speed_prop

  # Use Linear Algebra to solve
  a = np.linalg.solve(np.dot(X1.T, X1), np.dot(X1.T, y))
  predictedY = np.dot(X1, a)

  # Calculate the R-squared value
  SSres = y - predictedY
  SStot = y - y.mean()
  r_squared1 = 1 - (SSres.dot(SSres) / SStot.dot(SStot))
  print("R Squared: ", r_squared1)
  print("Coefficients: ", a)

  # Create a wiremesh for the plane where the predicted values will lie
  xx, yy, zz = np.meshgrid(X1[:, 0], X1[:, 1], X1[:, 2])
  combinedArrays = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
  Z = combinedArrays.dot(a)

  # Graph everything together
  fig = plt.figure(2)
  ax = fig.add_subplot(111, projection = '3d')
  ax.scatter(X1[:, 0], X1[:, 1], y, color = "r", label = "Actual Streamflow")
  ax.scatter(X1[:, 0], X1[:, 1], predictedY, color = "g", label = "Predicted Streamflow")
  ax.plot_trisurf(combinedArrays[:, 0], combinedArrays[:, 1], Z, alpha = 0.5)
  ax.set_xlabel("Direction "r"$[^\circ]$")
  ax.set_ylabel("Speed [m/s]")
  ax.set_zlabel("Peak Streamflow "r"$[m^{3}/s]$")
  ax.legend()
  ax.set_title("Sepulveda Dam")
  # Add R-squared value and equation to plot

  # Figure out how to save figure in specific position (azimuth, elevation)

  plt.show()

if __name__ == '__main__':
  main()



# Max reflectivity, storm speed, storm direction vs. peak discharge
# Max reflectivity, storm speed, storm direction vs. runoff ratio
