import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


def perturb_statistics(
        stat_defs, statistic_names, durations, change_factors, month_variable, easting_name, northing_name, easting,
        northing, change_factor_names, ref_stats,
):
    for _, row in stat_defs.iterrows():
        statistic_name = row['name']
        duration = row['duration']
        threshold = row['threshold']
        lag = row['lag']

        if statistic_name in ['mean', 'variance', 'skewness']:
            statistic_long_name = statistic_name
        elif statistic_name == 'probability_dry':
            statistic_long_name = statistic_name + '_' + '{0:.1f}'.format(threshold) + 'mm'
        elif statistic_name == 'autocorrelation':
            statistic_long_name = statistic_name + '_lag' + str(int(lag))

        if (statistic_long_name in statistic_names) and (duration in durations):

            # Look up change factor in netcdf file
            dset = xr.open_dataset(change_factors[duration])
            months = dset[month_variable][:].values.astype(int)
            indexers_kwargs = {easting_name: easting, northing_name: northing}
            vals = dset.sel(method='nearest', **indexers_kwargs)[change_factor_names[statistic_long_name]].values
            cfs = pd.DataFrame({'season': months, 'cf': vals})
            dset.close()

            # Apply change factor to statistic
            ref_stats = ref_stats.merge(cfs, how='left')
            if statistic_long_name in ['mean', 'variance', 'skewness']:
                ref_stats['value'] = np.where(
                    (ref_stats['name'] == statistic_name) & (ref_stats['duration'] == duration),
                    ref_stats['value'] * ref_stats['cf'],
                    ref_stats['value']
                )
            elif statistic_long_name == 'probability_dry_0.2mm':
                ref_stats['tmp_value'] = perturb_dry_probability(ref_stats['value'], ref_stats['cf'])
                ref_stats['value'] = np.where(
                    (ref_stats['name'] == statistic_name) & (ref_stats['duration'] == duration)
                    & (ref_stats['threshold'] == threshold),
                    ref_stats['tmp_value'],
                    ref_stats['value']
                )
            elif statistic_long_name == 'autocorrelation_lag1':
                ref_stats['tmp_value'] = perturb_autocorrelation(ref_stats['value'], ref_stats['cf'])
                ref_stats['value'] = np.where(
                    (ref_stats['name'] == statistic_name) & (ref_stats['duration'] == duration)
                    & (ref_stats['lag'] == lag),
                    ref_stats['tmp_value'],
                    ref_stats['value']
                )
            ref_stats.drop(columns=['cf', 'tmp_value'], inplace=True, errors='ignore')

    return ref_stats


def perturb_dry_probability(ref, cf):
    pert = (cf * (ref / (1.0 - ref))) / (1.0 + cf * (ref / (1.0 - ref)))
    return pert


def perturb_autocorrelation(ref, cf):
    pert = (cf * ((1.0 + ref) / (1.0 - ref)) - 1.0) / (cf * ((1.0 + ref) / (1.0 - ref)) + 1.0)
    return pert

