import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
  # Max Reflectivity, Speed, Azimuth vs. Peak Streamflow (all 4 watersheds)
  X1, y1, predictedY1, r_squared1, a1 = run_LA_regression(azimuth_SP_prop, speed_SP_prop, peak_Qs_SP_prop, \
      max_refs_SP_prop)
  model1 = sm.OLS(y1, X1).fit()
  print(model1.summary())
 

  X1_df = pd.DataFrame({'max_ref': X1[:, 0], 'azimuth': X1[:, 1], 'speed': X1[:, 2]})
  
  # Set figure size
  # plt.figure(figsize = (10, 7))

  # Generate a mask to only show the bottom triangle
  print(X1_df.corr())

  vif_data = pd.DataFrame()
  vif_data['feature'] = X1_df.columns
  
  # Calculate the VIF for each feature
  vif_data['VIF'] = [variance_inflation_factor(X1_df.values, i) for i in range(len(X1_df.columns))]
  print(vif_data)

  # Left off removing a feature and then calculating VIF again

  X2, y2, predictedY2, r_squared2, a2 = run_LA_regression(azimuth_WN_prop, speed_WN_prop, peak_Qs_WN_prop, \
      max_refs_WN_prop)
  model2 = sm.OLS(y2, X2).fit()
  print(model2.summary())

  X3, y3, predictedY3, r_squared3, a3 = run_LA_regression(azimuth_SA_prop, speed_SA_prop, peak_Qs_SA_prop, \
      max_refs_SA_prop)
  model3 = sm.OLS(y3, X3).fit()
  print(model3.summary())

  X4, y4, predictedY4, r_squared4, a4 = run_LA_regression(azimuth_SD_prop, speed_SD_prop, peak_Qs_SD_prop, \
      max_refs_SD_prop)
  model4 = sm.OLS(y4, X4).fit()
  print(model4.summary())

  # Max Reflectivity, Speed, Azimuth vs. Runoff Ratio (all 4 watersheds)
  X5, y5, predictedY5, r_squared5, a5 = run_LA_regression(azimuth_SP_prop2, speed_SP_prop2, run_ratios_SP_prop, \
      max_refs_SP_prop2)
  model5 = sm.OLS(y5, X5).fit()
  print(model5.summary())

  X6, y6, predictedY6, r_squared6, a6 = run_LA_regression(azimuth_WN_prop2, speed_WN_prop2, run_ratios_WN_prop, \
      max_refs_WN_prop2)
  model6 = sm.OLS(y6, X6).fit()
  print(model6.summary())

  X7, y7, predictedY7, r_squared7, a7 = run_LA_regression(azimuth_SA_prop2, speed_SA_prop2, run_ratios_SA_prop, \
      max_refs_SA_prop2)
  model7 = sm.OLS(y7, X7).fit()
  print(model7.summary())

  X8, y8, predictedY8, r_squared8, a8 = run_LA_regression(azimuth_SD_prop2, speed_SD_prop2, run_ratios_SD_prop, \
      max_refs_SD_prop2)
  model8 = sm.OLS(y8, X8).fit()
  print(model8.summary())


if __name__ == '__main__':
  main()
