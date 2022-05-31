import pandas as pd

def main():
  ## Find flash flood warnings from WWA catalog
  # Load from csv file
  wwa_fp = "./data/WWA_All_1995_2020.csv"
  wwa_entries = pd.read_csv(wwa_fp)
  wfos = wwa_entries.loc[:, "WFO"]
  beg_years = wwa_entries.loc[:, "BegYear"]
  beg_months = wwa_entries.loc[:, "BegMon"]
  beg_days = wwa_entries.loc[:, "BegDay"]
  beg_hours = wwa_entries.loc[:, "BegHur"]
  beg_mins = wwa_entries.loc[:, "BegMin"]
  end_years = wwa_entries.loc[:, "EndYear"]
  end_months = wwa_entries.loc[:, "EndDay"]
  end_days = wwa_entries.loc[:, "EndHur"]
  end_mins = wwa_entries.loc[:, "EndMin"]
  phenoms = wwa_entries.loc[:, "PHENOM"]
  statuses = wwa_entries.loc[:, "STATUS"] 

  # Run through catalog, look for new or cont ffws 

if __name__ == '__main__':
  main()
