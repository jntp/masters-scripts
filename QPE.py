import pygrib

source_fp = "/media/jntp/D2BC15A1BC1580E1/NCFRs/QPE Data/datac4Cliv/"
stage_fp = "ST4"
date_fp = "20020101"
hour_fp = "00"
type_fp = "01h"
fp = source_fp + stage_fp + "." + date_fp + hour_fp + "." + type_fp

grbs = pygrib.open(fp)
grbs.seek(0)
for grb in grbs:
  print(grb)
