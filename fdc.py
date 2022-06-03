import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d 

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

  # Convert streamflow from cfs to cms
  peaks_Q = peaks_Q * 0.028316847

  return peaks_Q, probs_P

def make_line_smooth(prob_Ps, discharge_Qs, intervals = 200):
  # Estimate spline curve coefficients, then use coeffs to determine y-values for evenly-spaced x-values
  P_Q_spline = make_interp_spline(prob_Ps, discharge_Qs) 
  prob_Ps_cv = np.linspace(prob_Ps.min(), prob_Ps.max(), intervals)
  discharge_Qs_cv = P_Q_spline(prob_Ps_cv)

  return prob_Ps_cv, discharge_Qs_cv

def get_median_streamflow(prob_Ps_cv, discharge_Qs_cv):
  watershed_interp = interp1d(prob_Ps_cv, discharge_Qs_cv)
  watershed_median = watershed_interp(50)

  return watershed_median


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
  # Transform the lines into a smooth curves
  sepulveda_Ps_cv, sepulveda_Qs_cv = make_line_smooth(sepulveda_Ps, sepulveda_Qs)
  whittier_Ps_cv, whittier_Qs_cv = make_line_smooth(whittier_Ps, whittier_Qs)
  santa_ana_Ps_cv, santa_ana_Qs_cv = make_line_smooth(santa_ana_Ps, santa_ana_Qs)
  san_diego_Ps_cv, san_diego_Qs_cv = make_line_smooth(san_diego_Ps, san_diego_Qs)

  # Interpolate values
  sepulveda_median = get_median_streamflow(sepulveda_Ps_cv, sepulveda_Qs_cv)
  whittier_median = get_median_streamflow(whittier_Ps_cv, whittier_Qs_cv)
  santa_ana_median = get_median_streamflow(santa_ana_Ps_cv, santa_ana_Qs_cv)
  san_diego_median = get_median_streamflow(san_diego_Ps_cv, san_diego_Qs_cv)

  # Plot the lines (including median lines)
  plt.plot(sepulveda_Ps_cv, sepulveda_Qs_cv, label = "Sepulveda Dam", color = "blue", zorder = 2)
  plt.axhline(y = sepulveda_median, xmax = 0.5, color = "blue", linestyle = "--", zorder = 1, alpha = 0.5)
  
  plt.plot(whittier_Ps_cv, whittier_Qs_cv, label = "Whittier Narrows Dam", color = "green", zorder = 2)
  plt.axhline(y = whittier_median, xmax = 0.5, color = "green", linestyle = "--", zorder = 1, alpha = 0.5)
  
  plt.plot(santa_ana_Ps_cv, santa_ana_Qs_cv, label = "Santa Ana River", color = "orange", zorder = 2)
  plt.axhline(y = santa_ana_median, xmax = 0.5, color = "orange", linestyle = "--", zorder = 1, alpha = 0.5)
  
  plt.plot(san_diego_Ps_cv, san_diego_Qs_cv, label = "San Diego River", color = "purple", zorder = 2)
  plt.axhline(y = san_diego_median, xmax = 0.5, color = "purple", linestyle = "--", zorder = 1, alpha = 0.5)

  plt.axvline(x = 50, label = "Median (2-yr)", color = "red", linestyle = "--", zorder = 2, alpha = 0.5)

  # Add legend
  plt.legend()

  # Add axis labels and title
  plt.xlabel("Exceedance probability [%]")
  plt.ylabel("Flow rate "r"$[m^{3}/s]$")

  # Add Title
  plt.title("Flow Duration Curve of Four Urban SoCal Watersheds")

  # Save the plot
  plt.savefig('./plots/fdc.png')

  plt.show()

if __name__ == '__main__':
  main()
