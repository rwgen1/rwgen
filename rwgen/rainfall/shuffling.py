"""
Notes:
    * Aggregate input series to monthly and spatially to catchment-average if applicable
        - Requires e.g. Thiessen polygons or some other weighting (from kriging?)
        - Possible to pass monthly series directly (or indeed SARIMA model parameters)
    * Fit SARIMA (automatically)
    * Receive NSRP event totals (point or domain-average)
        - Define an optimal domain for domain-averaging?
    * Fit shuffling parameter
    * Shuffle events
    * Simulate monthly series with SARIMA
    * Reorder shuffled events
    * Simulate NSRP again but passing in each event (discretised) and locate it in the correct place
        - Write out - but issue that not necessarily going to be able to put things at the right place in a file, as
          event could come from anywhere in the series
        - Would need to then write file(s) with a date column and at the end do a sort (i.e. read file, sort, write)
        - Gridded output could be placed in the right location directly

"""