# masters-scripts

<i> Note: Python segmentation procedure of NCFR cores is located in the <a href = "https://github.com/jntp/Catchcore">Catchcore repository</a>. </i>

## Introduction

</p>
Scripts for my MS Thesis: "The Role of Narrow Cold Frontal Rainbands (NCFRs) on Urban Flooding in Southern California"
<br> Authors: Justin Tang (student) and Hilary McMillan (PI)
<br> Institution: Department of Geography, San Diego State University

## File Listing
<p>
  QPE.py - plots the quantitative precipitation estimation (QPE) in Southern California--including figures 11 and 12, which show the percent of normal annual precipitation from NCFR events and the average total precipitation per NCFR event in       
  1995-2020, respectively
</p><p>
  QPE_old.py - old version of QPE.py
</p><p>
  WWA_NCFR.py - matches the time of a National Weather Service (NWS) issued flash flood warning (FFW) with an NCFR start and end time in the NCFR Catalog. <br><i>Note: This script only matches ONE FFW with each NCFR event; there can be multiple FFWs     per NCFR event</i>
</p><p>
  fdc.py - plots the flow duration curve (FDC) in figure 2
</p><p>
  flood_count.py - counts the number of flood events in Southern California from 1995 to 2020 based on the criteria specified in the manuscript
</p><p>
  linear_regression.py - runs the linear regression models and plots the results
</p><p>
  multiple_regression.py - runs the multiple regression models and plots the results
</p><p>
  multiple_regression_plus.py - most updated version of multiple_regression, includes statistics on p-scores and collinearity
</p><p>
  peak_flow.py - finds the peak streamflow for every NCFR event and appends to csv file
</p><p>
  runoff_ratio.py - calculates the runoff ratio given the mean discharge and daily precipitation data and appends to csv file
</p><p>
  runoff_ratio2.py - calculates the runoff ratio given the 15-minute discharge and hourly precipitation data and appends to csv file
</p><p>
  sa_streamflow_2004.py - plots figure 6, which is the "28 April 2005 Observed Streamflow in the Santa Ana River Watershed"
</p>
</p>
