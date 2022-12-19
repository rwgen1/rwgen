import datetime

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot


NAMES = {
    'mean': 'Mean', 'variance': 'Variance', 'skewness': 'Skewness',
    'probability_dry': 'Probability Dry', 'autocorrelation': 'Autocorrelation',
    'cross-correlation': 'Cross-Correlation'
}

UNITS = {
    'mean': 'mm', 'variance': 'mm**2', 'skewness': '-', 'probability_dry': '-', 'autocorrelation': '-',
    'cross-correlation': '-'
}


def plot_annual_cycle(sid, name, duration, ref=None, fit=None, sim=None):

    p = figure(title=NAMES[name] + ' (' + str(duration) + 'hr)', width=350, height=350)
    p.xaxis.axis_label = 'Month'
    p.yaxis.axis_label = NAMES[name] + ' (' + UNITS[name] + ')'

    mins = []
    maxs = []

    if ref is not None:
        df = ref.loc[ref['statistic_id'] == sid, ['season', 'value']]
        mins.append(df['value'].min())
        maxs.append(df['value'].max())
        ref_source = ColumnDataSource(df)
        p.line(x='season', y='value', source=ref_source, legend_label='ref', color='#000000')
        p.circle(x='season', y='value', source=ref_source, legend_label='ref', color='#000000')

    if fit is not None:
        df = fit.loc[fit['statistic_id'] == sid, ['season', 'value']]
        mins.append(df['value'].min())
        maxs.append(df['value'].max())
        fit_source = ColumnDataSource(df)
        p.line(x='season', y='value', source=fit_source, legend_label='fit', color='#1b9e77')
        p.circle(x='season', y='value', source=fit_source, legend_label='fit', color='#1b9e77')

    if sim is not None:
        df = sim.loc[sim['statistic_id'] == sid, ['season', 'mean']]  # , 'percentile_25', 'percentile_75'
        mins.append(df['mean'].min())
        maxs.append(df['mean'].max())
        # mins.append(df['percentile_25'].min())
        # maxs.append(df['percentile_75'].max())
        sim_source = ColumnDataSource(df)
        # p.line(x='season', y='percentile_25', source=sim_source, color='#d95f02', line_dash='dotted')
        # p.line(x='season', y='percentile_75', source=sim_source, color='#d95f02', line_dash='dotted')
        p.line(x='season', y='mean', source=sim_source, legend_label='sim', color='#d95f02')
        p.circle(x='season', y='mean', source=sim_source, legend_label='sim', color='#d95f02')

    y_min = min(mins)
    y_min = min(y_min, 0)
    y_max = max(maxs) * 1.1
    p.y_range.start = y_min
    p.y_range.end = y_max

    p.legend.location = 'top_left'
    p.legend.spacing = 2
    p.legend.padding = 5
    p.legend.margin = 5
    p.legend.background_fill_alpha = 0.2

    p.xaxis.ticker = list(range(1, 12 + 1))
    p.xaxis.major_label_overrides = {
        1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'
    }

    return p


def plot_cross_correlation(sid, name, duration, season, ref=None, fit=None, sim=None):

    p = figure(title=NAMES[name] + ' (' + str(duration) + 'hr) - Season = ' + str(season), width=350, height=350)
    p.xaxis.axis_label = 'Separation Distance (km)'
    p.yaxis.axis_label = NAMES[name] + ' (' + UNITS[name] + ')'

    x_maxs = []
    y_mins = []
    y_maxs = []

    if ref is not None:
        df = ref.loc[ref['statistic_id'] == sid, ['season', 'distance', 'value']]
        y_mins.append(df['value'].min())
        y_maxs.append(df['value'].max())
        x_maxs.append(df['distance'].max())
        ref_source = ColumnDataSource(df)
        p.circle(x='distance', y='value', source=ref_source, legend_label='ref', color='#000000', fill_alpha=0)

    if fit is not None:
        df = fit.loc[fit['statistic_id'] == sid, ['season', 'distance', 'value']]
        df = df.sort_values('distance')
        y_mins.append(df['value'].min())
        y_maxs.append(df['value'].max())
        x_maxs.append(df['distance'].max())
        fit_source = ColumnDataSource(df)
        p.line(x='distance', y='value', source=fit_source, legend_label='fit', color='#1b9e77')

    if sim is not None:
        df = sim.loc[sim['statistic_id'] == sid, ['season', 'distance', 'mean']]  # , 'percentile_25', 'percentile_75'
        df = df.sort_values('distance')
        y_mins.append(df['mean'].min())
        y_maxs.append(df['mean'].max())
        x_maxs.append(df['distance'].max())
        # mins.append(df['percentile_25'].min())
        # maxs.append(df['percentile_75'].max())
        sim_source = ColumnDataSource(df)
        # p.line(x='season', y='percentile_25', source=sim_source, color='#d95f02', line_dash='dotted')
        # p.line(x='season', y='percentile_75', source=sim_source, color='#d95f02', line_dash='dotted')

        # p.line(x='distance', y='mean', source=sim_source, legend_label='sim', color='#d95f02')
        p.circle(x='distance', y='mean', source=sim_source, legend_label='sim', color='#d95f02', fill_alpha=0)

    y_min = min(y_mins)
    y_min = min(y_min, 0)
    y_max = 1  # max(maxs) * 1.1
    p.y_range.start = y_min
    p.y_range.end = y_max

    x_max = max(x_maxs) * 1.05
    p.x_range.start = 0
    p.x_range.end = x_max

    p.legend.location = 'bottom_left'
    p.legend.spacing = 2
    p.legend.padding = 5
    p.legend.margin = 5
    p.legend.background_fill_alpha = 0.2

    return p


def construct_gridplot(plots, n_in_row):
    nested_list = []
    col = 1
    row = 1
    for p in plots:
        if col == 1:
            nested_list.append([p])
            col += 1
        elif col < n_in_row:
            nested_list[row-1].append(p)
            col += 1
        elif col == n_in_row:
            nested_list[row-1].append(p)
            col = 1
            row += 1

    g = gridplot(nested_list, sizing_mode='scale_both')

    return g


def show_plot(p):
    show(p)
