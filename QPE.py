import pygrib
import pandas as pd

def main(): 
  source_fp = "/media/jntp/D2BC15A1BC1580E1/NCFRs/QPE Data/2002/"
  stage_fp = "ST4"
  date_fp = "20020101"
  hour_fp = "00"
  type_fp = "01h"
  fp = source_fp + stage_fp + "." + date_fp + hour_fp + "." + type_fp

  grbs = pygrib.open(fp)
  grb = grbs.message(1) 
  # grb = grbs.select(name = "Total Precipitation")[0]
  # precip = grb.values
  # print(precip.shape, precip.min(), precip.max())
  data, lats, lons = grb.data(lat1 = 32, lat2 = 36, lon1 = -121, lon2 = -114)
  print(data.shape, data.max(), data.min(), lats.shape, lons.shape)
  print(data[50], lats[50], lons[50])

  ## Total Precipitation for all NCFR events
  # Load times from csv file
  ncfr_fp = "./data/NCFR_Catalog.csv" 
  ncfr_entries = pd.read_csv(ncfr_fp) 
  ncfrs = ncfr_entries.set_index("Index", drop = False)
  years = ncfr_entries.loc[:, "Year"]
  months = ncfr_entries.loc[:, "Month"]
  days = ncfr_entries.loc[:, "Days"]
  start_hours = ncfr_entries.loc[:, "Start_Hour"]
  end_hours = ncfr_entries.loc[:, "End_Hour"]

if __name__ == '__main__':
  main()

# Find:
# Total precipitation for all NCFR events
# Percent of normal annual precipitation (total NCFR precip year/total precip year)
# Average rainfall rates
# Average total precipitation per NCFR event
