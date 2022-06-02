import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Need to convert to metric system ai yah!!!

def load_flood_freq_data(file_path):
  entries = pd.read_csv(file_path) 
  peaks_Q_str = entries.loc[:, "peak_va"]
  probs_P_str = entries.loc[:, "prob_p"]

  # Create empty lists that will hold converted streamflow and probability data
  peaks_Q = []
  probs_P = []
  
  # Convert values from string to int and float
  for i, peak_Q_str in enumerate(peaks_Q_str):
    peaks_Q.append(int(peaks_Q_str[i]))
    probs_P.append(float(probs_P_str[i]))

  # Convert to numpy array
  peaks_Q = np.array(peaks_Q)
  probs_P = np.array(probs_P)

  return peaks_Q, probs_P


def main():
  ## Load the flood frequency data
  # Get the file paths
  sepulveda_fp = "./data/sepulveda_dam_flood_frequency_1930_2020.csv"
  whittier_fp = "./data/whittier_narrows_flood_frequency_1957_2021.csv"
  santa_ana_fp = "./data/santa_ana_flood_frequency_1923_2020.csv"
  san_diego_fp = "./data/san_diego_flood_frequency_1913_2021.csv"

  # Load flood frequency data
  sepulveda_Qs, sepulveda_Ps = load_flood_freq_data(sepulveda_fp)
  whittier_Qs, whittier_Ps = load_flood_freq_data(whittier_fp)
  santa_ana_Qs, santa_ana_Ps = load_flood_freq_data(santa_ana_fp)
  san_diego_Qs, san_diego_Ps = load_flood_freq_data(san_diego_fp)

  ## Prepare data for plotting
  # Transform the line into a smooth curve
  # Estimate spline curve coefficients, then use coeffs to determine y-values for evenly-spaced x-values
  P_Q_spline = make_interp_spline(sepulveda_Ps, sepulveda_Qs) 
  sepulveda_Ps_cv = np.linspace(sepulveda_Ps.min(), sepulveda_Ps.max(), 500)
  sepulveda_Qs_cv = P_Q_spline(sepulveda_Ps_cv)

  # Create a figure and a subplot
  # fig, ax = plt.subplots() 

  # Properties
  # ax.set_yscale("log")

  # Plot the data
  plt.plot(sepulveda_Ps, sepulveda_Qs)
  plt.xlabel("Exceedance [%]")
  plt.ylabel("Flow rate")
  plt.show()

if __name__ == '__main__':
  main()

# Perhaps plot as a multi-panel plot? 
