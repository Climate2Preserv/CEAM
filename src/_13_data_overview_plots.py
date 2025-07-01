import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
matplotlib.use('Agg')
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import importlib
from _00_CEAM_run import (directory, known_format, solar_radiation, demands_list,
                          frequency_ed, outlier_value)
from _11_initial_csv_check import directory_others, directory_graphs_01


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
# dictionary with units; can be changed if needed
units_dict = {
    'Ti': r'[$^o$C]', 'pvi': '[Pa]', 'RHi': '[%]', 'Is': r'[Wh/m$^2$]',
    # subscript: r'$_...$'; superscript: r'$^...$'
    'Te': r'[$^o$C]', 'pve': '[Pa]', 'RHe': '[%]',
    'Tde': r'[$^o$C]', 'Tdi': r'[$^o$C]', 'Twbe': r'[$^o$C]', 'Twbi': r'[$^o$C]',
    'AbsHi': r'[g/m$^3$]', 'AbsHe': r'[g/m$^3$]',
    'deltaT': r'[$^o$C]', 'deltaTd': r'[$^o$C]', 'deltaTwb': r'[$^o$C]',
    'deltaRH': '[%]', 'deltaAbsH': r'[g/m$^3$]',
    'TIST': r'[$^o$C]', 'TISTd': r'[$^o$C]', 'TISTwb': r'[$^o$C]',
    'D1': '[Wh]', 'D2': '[Wh]', 'D3': '[Wh]', 'D4': '[Wh]', 'D5': '[Wh]',
    'HDD': r'[$^o$Cdays]', 'CDD': r'[$^o$Cdays]'
}

all_demands = {'D1', 'D2', 'D3', 'D4', 'D5'}
exclude_columns = all_demands.difference(demands_list)  # compute columns to exclude

def handle_exceptions(column_name, column_data):
    '''function to define y-axis range for different variables'''
    column_data = np.array(column_data)  # convert to np array
    column_data = column_data[np.isfinite(column_data)]  # remove NaN and Inf values

    if column_name in ['RHi', 'RHe']:  # for humidity
        y_min = 0
        y_max = 100
    elif column_name in ['D1', 'D2', 'D3', 'D4', 'D5', 'Is', 'HDD',
                         'CDD', 'AbsHi', 'AbsHe']:  # for energy consumption and solar radiation
        y_min = 0
        y_max = np.nanmax(column_data) * 1.25 if len(column_data) > 0 else 100  # example adjustment with 100
    else:  # automatic range adjustment for other cases
        if len(column_data) > 0:
            min_val = np.nanmin(column_data)
            max_val = np.nanmax(column_data)
            y_min = min_val - abs(min_val) * 0.25 if min_val < 0 else min_val * 0.75
            y_max = max_val + abs(max_val) * 0.25
        else:
            y_min, y_max = 0, 100  # fallback in case of empty data

    return y_min, y_max

def custom_yticks(value, pos):
    '''function to define ticks notation'''
    if value >= 1000000:
        return f'{value / 100000:.0f}×10$^5$'
    elif value >= 100000:
        return f'{value / 10000:.0f}×10$^4$'
    elif value >= 10000:
        return f'{value / 1000:.0f}×10$^3$'
    elif value <= 1:
        return f'{value:.1f}'
    else:
        return f'{value:.0f}'

def highlight_missing_periods(ax, time_data, column_data, facecolor='#FAF3D3', alpha=0.75):
    '''highlight missing periods'''
    if len(column_data) > 1:
        missing_periods = []
        start_time = None
        for i in range(len(column_data)):
            if np.isnan(column_data[i]):
                if start_time is None:
                    start_time = time_data[i - 1] if i > 0 else time_data[i]
            elif start_time is not None:
                end_time = time_data[i]
                missing_periods.append((start_time, end_time))
                start_time = None

        if start_time is not None:  # for missing values till the end of the data
            end_time = time_data.iloc[-1]
            missing_periods.append((start_time, end_time))

        for start, end in missing_periods:  # plotting missing periods as shaded regions
            ax.axvspan(start, end, facecolor=facecolor, alpha=alpha)


'''FUNCTIONALITIES'''
def main():
    def create_placeholder_plot(output_file):
        '''function to create a placeholder plot when no data is available'''
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center',
                 fontsize=20, color='red')
        plt.axis('off')
        plt.savefig(output_file)
        plt.close('all')  # close all figures

    def create_scatter_plot_base(time_data, column_data, column_name, output_file,
                                 marker_style='d', marker_size=2.5, marker_edge_width=1,
                                 line_style='--', line_width=1, line_color='grey',
                                 tick_size={'x': 16, 'y': 16}, label_properties={'size': 24, 'bold': True},
                                 grid_properties={'color': 'grey', 'linestyle': '--', 'linewidth': 0.5},
                                 dpi=300):
        '''function to create an hourly scatter plot, where x-axis is 'Time' and y-axis is {column_name}'''
        if column_data.isna().all():  # check if all data points are NaN
            create_placeholder_plot(output_file)
            return

        y_min, y_max = handle_exceptions(column_name, column_data)      # determine y-axis limits
        plot_height, plot_width = 32, 24  # plot dimensions in cm
        fig, ax = plt.subplots(
            figsize=(plot_height * 0.393701, plot_width * 0.393701))  # plot size in inches (matplotlib unit)

        # plotting data
        ax.set_xlim(min(time_data), max(time_data))  # x-axis limits
        ax.set_ylim(y_min, y_max)  # y-axis limits
        ax.plot(time_data, column_data, linestyle=line_style, linewidth=line_width, color=line_color,
                marker=marker_style, markersize=marker_size, markeredgewidth=marker_edge_width,
                markerfacecolor='lightgrey', markeredgecolor='dimgray', alpha=1.0)

        # title, labels, ticks
        ax.set_xlabel('Time', fontsize=label_properties['size'],
                      fontweight='bold' if label_properties['bold'] else 'normal')
        y_unit = units_dict.get(column_name.split(' ')[0], '')  # y-axis unit
        ax.set_ylabel(f'{column_name} {y_unit}', fontsize=label_properties['size'],
                      fontweight='bold' if label_properties['bold'] else 'normal')
        ax.xaxis.set_major_formatter(mdates.DateFormatter(known_d_format))  # adjust this to date format
        plt.xticks(rotation=45, fontsize=tick_size['x'])
        plt.yticks(fontsize=tick_size['y'])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_yticks))

        # the average value
        if not column_data.isna().all():
            mean_value = np.nanmean(column_data)
            ax.axhline(mean_value, color='steelblue', linestyle='--', linewidth=3.0)
            ax.annotate(f'ave: {mean_value:.2f} {y_unit}', xy=(1, mean_value), xycoords=('axes fraction', 'data'),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=label_properties['size'] * 0.5, fontweight='bold', color='steelblue')

        highlight_missing_periods(ax, time_data, column_data)   # highlight missing periods

        max_indices = np.where(column_data == np.nanmax(column_data))[0]    # max value for markers
        min_indices = np.where(column_data == np.nanmin(column_data))[0]    # min value for markers
        for index in max_indices:       # for max
            ax.scatter([time_data[index]], [column_data[index]], marker='o', color='salmon', edgecolor='firebrick',
                       alpha=1.0, s=100)
        for index in min_indices:       # for min
            ax.scatter([time_data[index]], [column_data[index]], marker='o', color='salmon', edgecolor='firebrick',
                       alpha=1.0, s=100)

        # annotation text for max and min values
        if len(max_indices) > 1:
            max_annotation = f'{np.nanmax(column_data):.1f}\n (more than one)'
        else:
            max_time = time_data[max_indices[0]]
            max_annotation = f'{np.nanmax(column_data):.1f}\n ({max_time.strftime(known_format)})'

        if len(min_indices) > 1:
            min_annotation = f'{np.nanmin(column_data):.1f}\n (more than one)'
        else:
            min_time = time_data[min_indices[0]]
            min_annotation = f'{np.nanmin(column_data):.1f}\n ({min_time.strftime(known_format)})'

        ax.annotate(max_annotation,     # annotation positioning for max
                    (time_data[max_indices[0]], column_data[max_indices[0]]),
                    textcoords="offset points",
                    xytext=(0, 20),
                    ha='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold')
        ax.annotate(min_annotation,     # annotation positioning for min
                    (time_data[min_indices[0]], column_data[min_indices[0]]),
                    textcoords="offset points",
                    xytext=(0, -20),
                    ha='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold')

        # add grid to the axes
        ax.grid(
            True, which='both', axis='both',
            color=grid_properties['color'],
            linestyle=grid_properties['linestyle'],
            linewidth=grid_properties['linewidth']
        )

        show_textbox = frequency_ed != 'h' or outlier_value is not None     # determine the outlier caption

        if show_textbox:
            graph_position = 0.075
            text_info = ""
            if frequency_ed != 'h' and column_name in all_demands:  # frequency info if applicable
                text_info += (
                    f"   !! Graphs of demand records are generated based on {frequency_ed.upper()} frequency inputs\n"
                )
            if outlier_value is not None:  # outlier info if applicable
                text_info += (
                    f"   !! The assumed outliers are considered for -{outlier_value}σ:+{outlier_value}σ range"
                )

            # caption style
            plt.figtext(
                0.1, 0.025,  # alignment
                text_info,
                wrap=True,
                horizontalalignment='left',
                fontsize=label_properties['size'] * 0.65,
                fontweight='bold',
                fontstyle='italic',
                color='firebrick'
            )
        else:
            graph_position = 0

        plt.tight_layout(rect=[0, graph_position, 1, 1])  # adjust layout for bottom space
        plt.savefig(output_file, dpi=dpi)
        plt.close('all')  # close all figures

    def create_scatter_plot_daily(time_data, column_data, column_name, output_file,
                                  marker_style='d', marker_size=5, marker_edge_width=1,
                                  line_style='--', line_width=1.5, line_color='grey',
                                  tick_size={'x': 16, 'y': 16}, label_properties={'size': 24, 'bold': True},
                                  grid_properties={'color': 'grey', 'linestyle': '--', 'linewidth': 0.5},
                                  dpi=300):
        '''function to create a daily scatter plot, where x-axis is 'Time' and y-axis is {column_name}'''
        y_min, y_max = handle_exceptions(column_name, column_data)
        plot_height, plot_width = 32, 24  # plot dimension in cm
        fig, ax = plt.subplots(figsize=(plot_height * 0.393701, plot_width * 0.393701))  # plot size in inches

        ax.set_xlim(min(time_data), max(time_data))  # limits for the x-axis
        ax.set_ylim(y_min, y_max)  # limits for the y-axis
        ax.plot(time_data, column_data, linestyle=line_style, linewidth=line_width, color=line_color,
                marker=marker_style, markersize=marker_size, markeredgewidth=marker_edge_width,
                markerfacecolor='lightgrey', markeredgecolor='dimgray', alpha=0.5)

        # title, labels, ticks
        ax.set_xlabel('Time', fontsize=label_properties['size'],
                      fontweight='bold' if label_properties['bold'] else 'normal')
        y_unit = units_dict.get(column_name.split(' ')[0], '')  # determine the unit for the y-axis
        ax.set_ylabel(f'{column_name} {y_unit}', fontsize=label_properties['size'],
                      fontweight='bold' if label_properties['bold'] else 'normal')
        ax.xaxis.set_major_formatter(mdates.DateFormatter(known_d_format))
        plt.xticks(rotation=45, fontsize=tick_size['x'])
        plt.yticks(fontsize=tick_size['y'])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_yticks))

        highlight_missing_periods(ax, time_data, column_data)  # highlight missing periods

        # determine max and min values
        max_indices = np.where(column_data == column_data.max())[0]
        min_indices = np.where(column_data == column_data.min())[0]
        for index in max_indices:
            ax.scatter([time_data[index]], [column_data[index]], marker='o', color='salmon', edgecolor='firebrick',
                       alpha=1.0, s=150)
        for index in min_indices:
            ax.scatter([time_data[index]], [column_data[index]], marker='o', color='salmon', edgecolor='firebrick',
                       alpha=1.0, s=150)

        # annotation text
        if len(max_indices) > 1:
            max_annotation = f'{column_data.max():.1f}\n (more than one)'
        else:
            max_time = time_data[max_indices[0]]
            max_time_datetime = datetime.strptime(str(max_time), '%Y-%m-%d %H:%M:%S')
            max_annotation = f'{column_data.max():.1f}\n ({max_time_datetime.strftime("%d.%m.%Y")})'

        if len(min_indices) > 1:
            min_annotation = f'{column_data.min():.1f}\n (more than one)'
        else:
            min_time = time_data[min_indices[0]]
            min_time_datetime = datetime.strptime(str(min_time), '%Y-%m-%d %H:%M:%S')
            min_annotation = f'{column_data.min():.1f}\n ({min_time_datetime.strftime("%d.%m.%Y")})'

        ax.annotate(max_annotation,
                    (time_data[max_indices[0]], column_data.max()),
                    textcoords="offset points",
                    xytext=(0, 20),
                    ha='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold')
        ax.annotate(min_annotation,
                    (time_data[min_indices[0]], column_data.min()),
                    textcoords="offset points",
                    xytext=(0, -20),
                    ha='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold')

        mean_value = np.nanmean(column_data)
        ax.axhline(mean_value, color='steelblue', linestyle='--', linewidth=3.0)
        ax.annotate(f'ave: {mean_value:.2f} {y_unit}', xy=(1, mean_value), xycoords=('axes fraction', 'data'),
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold', color='steelblue')

        # add grid to the axes
        ax.grid(
            True, which='both', axis='both',
            color=grid_properties['color'],
            linestyle=grid_properties['linestyle'],
            linewidth=grid_properties['linewidth']
        )

        show_textbox = frequency_ed != 'h' or outlier_value is not None     # determine the outlier caption

        if show_textbox:
            graph_position = 0.075
            text_info = ""
            if frequency_ed != 'h' and column_name in all_demands:  # frequency info if applicable
                text_info += (
                    f"   !! Graphs of demand records are generated based on {frequency_ed.upper()} frequency inputs\n"
                )
            if outlier_value is not None:  # outlier info if applicable
                text_info += (
                    f"   !! The assumed outliers are considered for -{outlier_value}σ:+{outlier_value}σ range"
                )

            # caption style
            plt.figtext(
                0.1, 0.025,  # alignment
                text_info,
                wrap=True,
                horizontalalignment='left',
                fontsize=label_properties['size'] * 0.65,
                fontweight='bold',
                fontstyle='italic',
                color='firebrick'
            )
        else:
            graph_position = 0

        plt.tight_layout(rect=[0, graph_position, 1, 1])  # adjust layout for bottom space
        plt.savefig(output_file, dpi=dpi)
        plt.close('all')  # close all figures

    def create_scatter_plot_monthly(time_data, column_data, column_name, output_file,
                                    marker_style='d', marker_size=10, marker_edge_width=2.5,
                                    line_style='--', line_width=5, line_color='grey',
                                    tick_size={'x': 16, 'y': 16}, label_properties={'size': 24, 'bold': True},
                                    grid_properties={'color': 'grey', 'linestyle': '--', 'linewidth': 0.5},
                                    dpi=300):
        '''function to create a monthly scatter plot, where x-axis is 'Time' and y-axis is {column_name}'''
        y_min, y_max = handle_exceptions(column_name, column_data)
        plot_height, plot_width = 32, 24  # plot dimension in cm
        plt.figure(figsize=(plot_height * 0.393701, plot_width * 0.393701))  # plot size in inches

        plt.xlim(min(time_data), max(time_data))  # limits for the x-axis
        plt.ylim(y_min, y_max)  # limits for the y-axis
        plt.plot(time_data, column_data, linestyle=line_style, linewidth=line_width, color=line_color,
                 # initial plot of data with custom markers and lines
                 marker=marker_style, markersize=marker_size, markeredgewidth=marker_edge_width,
                 markerfacecolor='lightgrey', markeredgecolor='dimgray', alpha=1.0)

        # title, labels, ticks
        plt.xlabel('Time', fontsize=label_properties['size'],
                   fontweight='bold' if label_properties['bold'] else 'normal')
        y_unit = units_dict.get(column_name.split(' ')[0], '')  # determine the unit for the y-axis
        plt.ylabel(f'{column_name} {y_unit}', fontsize=label_properties['size'],
                   fontweight='bold' if label_properties['bold'] else 'normal')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(known_m_format))  # adjust date format as needed
        plt.xticks(rotation=45, fontsize=tick_size['x'])
        plt.yticks(fontsize=tick_size['y'])
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(custom_yticks))

        highlight_missing_periods(plt.gca(), time_data, column_data)    # highlight missing periods

        # determine max and min values
        max_indices = np.where(column_data == column_data.max())[0]  # all the max indexes
        min_indices = np.where(column_data == column_data.min())[0]  # all the min indexes
        # markers
        for index in max_indices:  # iterate for max indexes
            plt.scatter([time_data[index]], [column_data[index]], marker='o', color='salmon',
                        edgecolor='firebrick', linewidths=2.5, alpha=1.0, s=500)
        for index in min_indices:  # iterate for min indexes
            plt.scatter([time_data[index]], [column_data[index]], marker='o', color='salmon',
                        edgecolor='firebrick', linewidths=2.5, alpha=1.0, s=500)

        # annotation text
        if len(max_indices) > 1:  # for max value(s)
            max_annotation = f'{column_data.max():.1f}\n (more than one)'
        else:
            max_time = time_data[max_indices[0]]
            max_time_datetime = datetime.strptime(str(max_time), '%Y-%m-%d %H:%M:%S')
            max_annotation = f'{column_data.max():.1f}'

        if len(min_indices) > 1:  # for min value(s)
            min_annotation = f'{column_data.min():.1f}\n (more than one)'
        else:
            min_time = time_data[min_indices[0]]
            min_time_datetime = datetime.strptime(str(min_time), '%Y-%m-%d %H:%M:%S')
            min_annotation = f'{column_data.min():.1f}'

        # annotate positioning
        plt.annotate(max_annotation,
                     (time_data[max_indices[0]], column_data.max()),
                     textcoords="offset points",
                     xytext=(0, 20),
                     ha='center',
                     fontsize=label_properties['size'] * 0.75, fontweight='bold')
        plt.annotate(min_annotation,
                     (time_data[min_indices[0]], column_data.min()),
                     textcoords="offset points",
                     xytext=(0, 20),
                     ha='center',
                     fontsize=label_properties['size'] * 0.75, fontweight='bold')

        # average value line
        mean_value = np.nanmean(column_data)
        plt.axhline(mean_value, color='steelblue', linestyle='--', linewidth=5.0)

        # annotate average value
        plt.annotate(f'ave: {mean_value:.2f} {y_unit}', xy=(1, mean_value), xycoords=('axes fraction', 'data'),
                     xytext=(5, 0), textcoords='offset points', ha='left', va='center',
                     fontsize=label_properties['size'] * 0.75, fontweight='bold', color='steelblue')

        ax = plt.gca()  # get the current axes object
        # add grid to the axes
        ax.grid(
            True, which='both', axis='both',
            color=grid_properties['color'],
            linestyle=grid_properties['linestyle'],
            linewidth=grid_properties['linewidth']
            )

        show_textbox = frequency_ed != 'h' or outlier_value is not None     # determine the outlier caption

        if show_textbox:
            graph_position = 0.075
            text_info = ""
            if frequency_ed != 'h' and column_name in all_demands:  # frequency info if applicable
                text_info += (
                    f"   !! Graphs of demand records are generated based on {frequency_ed.upper()} frequency inputs\n"
                )
            if outlier_value is not None:  # outlier info if applicable
                text_info += (
                    f"   !! The assumed outliers are considered for -{outlier_value}σ:+{outlier_value}σ range"
                )

            # caption style
            plt.figtext(
                0.1, 0.025,  # alignment
                text_info,
                wrap=True,
                horizontalalignment='left',
                fontsize=label_properties['size'] * 0.65,
                fontweight='bold',
                fontstyle='italic',
                color='firebrick'
            )
        else:
            graph_position = 0

        plt.tight_layout(rect=[0, graph_position, 1, 1])  # adjust layout for bottom space
        plt.savefig(output_file, dpi=dpi)
        plt.close('all')  # close all figures

    def create_bar_plot_daily(time_data, column_data, column_name, output_file,
                              bar_color='lightgrey', bar_width=1.0, edge_color='dimgray', edge_width=0.5, alpha=0.5,
                              highlight_color='salmon', highlight_edge='firebrick',
                              tick_size={'x': 16, 'y': 16}, label_properties={'size': 24, 'bold': True},
                              grid_properties={'color': 'grey', 'linestyle': '--', 'linewidth': 0.5},
                              dpi=300):
        '''function to create a daily bar plot, where x-axis is 'Time' and y-axis is {column_name}'''
        y_min, y_max = handle_exceptions(column_name, column_data)
        plot_height, plot_width = 32, 24  # plot dimension in cm
        fig, ax = plt.subplots(figsize=(plot_height * 0.393701, plot_width * 0.393701))  # plot size in inches
        ax.set_xlim(min(time_data), max(time_data))  # limits for the x-axis
        ax.set_ylim(y_min, y_max)  # limits for the y-axis

        # highlight max and min values
        bar_colors = [highlight_color if (val == column_data.max() or val == column_data.min()) else bar_color for val in column_data]
        edge_colors = [highlight_edge if (val == column_data.max() or val == column_data.min()) else edge_color for val in column_data]
        ax.bar(time_data, column_data, color=bar_colors, edgecolor=edge_colors, linewidth=edge_width, alpha=alpha, width=bar_width)

        highlight_missing_periods(ax, time_data, column_data)   # highlight missing periods

        # title, labels, ticks
        ax.set_xlabel('Time', fontsize=label_properties['size'], fontweight='bold' if label_properties['bold'] else 'normal')
        y_unit = units_dict.get(column_name.split(' ')[0], '')  # determine the unit for the y-axis
        ax.set_ylabel(f'{column_name} {y_unit}', fontsize=label_properties['size'], fontweight='bold' if label_properties['bold'] else 'normal')
        ax.xaxis.set_major_formatter(mdates.DateFormatter(known_d_format))
        plt.xticks(rotation=45, fontsize=tick_size['x'])
        plt.yticks(fontsize=tick_size['y'])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))  # Change to actual formatter

        # determine max and min values
        max_indices = np.where(column_data == column_data.max())[0]  # all the max indexes
        min_indices = np.where(column_data == column_data.min())[0]  # all the min indexes

        # annotation text
        if len(max_indices) > 1:  # for max value(s)
            max_annotation = f'{column_data.max():.1f}\n (more than one)'
        else:
            max_time = time_data[max_indices[0]]
            max_time_datetime = datetime.strptime(str(max_time), '%Y-%m-%d %H:%M:%S')
            max_annotation = f'{column_data.max():.1f}\n ({max_time_datetime.strftime("%d.%m.%Y")})'
        if len(min_indices) > 1:  # for min value(s)
            min_annotation = f'{column_data.min():.1f}\n (more than one)'
        else:
            min_time = time_data[min_indices[0]]
            min_time_datetime = datetime.strptime(str(min_time), '%Y-%m-%d %H:%M:%S')
            min_annotation = f'{column_data.min():.1f}\n ({min_time_datetime.strftime("%d.%m.%Y")})'

        # annotate positioning
        ax.annotate(max_annotation,
                    (time_data[max_indices[0]], column_data.max()),
                    textcoords="offset points",
                    xytext=(0, 20),
                    ha='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold')
        ax.annotate(min_annotation,
                    (time_data[min_indices[0]], column_data.min()),
                    textcoords="offset points",
                    xytext=(0, 20),
                    ha='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold')

        plt.tight_layout()  # adjust layout
        ax.grid(True, which='both', axis='both', color=grid_properties['color'], linestyle=grid_properties['linestyle'],
                linewidth=grid_properties['linewidth'])

        plt.savefig(output_file, dpi=dpi)
        plt.close('all')  # close all figures

    def create_bar_plot_monthly(time_data, column_data, column_name, output_file,
                                bar_color='lightgrey', alpha=1.0, bar_width=15.0, edge_color='dimgray', edge_width=2.5,
                                highlight_color='salmon', highlight_edge='firebrick',
                                tick_size={'x': 16, 'y': 16}, label_properties={'size': 24, 'bold': True},
                                grid_properties={'color': 'grey', 'linestyle': '--', 'linewidth': 0.5},
                                dpi=300):
        '''function to create a monthly bar plot, where x-axis is 'Time' and y-axis is {column_name}'''
        y_min, y_max = handle_exceptions(column_name, column_data)
        plot_height, plot_width = 32, 24  # plot dimension in cm
        fig, ax = plt.subplots(figsize=(plot_height * 0.393701, plot_width * 0.393701))  # plot size in inches
        ax.set_xlim(min(time_data), max(time_data))  # limits for the x-axis
        ax.set_ylim(y_min, y_max)  # limits for the y-axis

        # highlight max and min values
        bar_colors = [highlight_color if (val == column_data.max() or val == column_data.min()) else bar_color for val in column_data]
        edge_colors = [highlight_edge if (val == column_data.max() or val == column_data.min()) else edge_color for val in column_data]
        ax.bar(time_data, column_data, color=bar_colors, edgecolor=edge_colors, linewidth=edge_width, alpha=alpha, width=bar_width)

        highlight_missing_periods(ax, time_data, column_data)   # highlight missing periods

        # title, labels, ticks
        ax.set_xlabel('Time', fontsize=label_properties['size'], fontweight='bold' if label_properties['bold'] else 'normal')
        y_unit = units_dict.get(column_name.split(' ')[0], '')  # determine the unit for the y-axis
        ax.set_ylabel(f'{column_name} {y_unit}', fontsize=label_properties['size'], fontweight='bold' if label_properties['bold'] else 'normal')
        ax.xaxis.set_major_formatter(mdates.DateFormatter(known_m_format))
        plt.xticks(rotation=45, fontsize=tick_size['x'])
        plt.yticks(fontsize=tick_size['y'])
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_yticks))

        # determine max and min values
        max_indices = np.where(column_data == column_data.max())[0]  # all the max indexes
        min_indices = np.where(column_data == column_data.min())[0]  # all the min indexes

        # annotation text
        if len(max_indices) > 1:  # for max value(s)
            max_annotation = f'{column_data.max():.1f}\n (more than one)'
        else:
            max_time = time_data[max_indices[0]]
            max_time_datetime = datetime.strptime(str(max_time), '%Y-%m-%d %H:%M:%S')
            max_annotation = f'{column_data.max():.1f}\n ({max_time_datetime.strftime("%d.%m.%Y")})'
        if len(min_indices) > 1:  # for min value(s)
            min_annotation = f'{column_data.min():.1f}\n (more than one)'
        else:
            min_time = time_data[min_indices[0]]
            min_time_datetime = datetime.strptime(str(min_time), '%Y-%m-%d %H:%M:%S')
            min_annotation = f'{column_data.min():.1f}\n ({min_time_datetime.strftime("%d.%m.%Y")})'

        # annotate positioning
        ax.annotate(max_annotation,
                    (time_data[max_indices[0]], column_data.max()),
                    textcoords="offset points",
                    xytext=(0, 20),
                    ha='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold')
        ax.annotate(min_annotation,
                    (time_data[min_indices[0]], column_data.min()),
                    textcoords="offset points",
                    xytext=(0, 20),
                    ha='center',
                    fontsize=label_properties['size'] * 0.5, fontweight='bold')

        plt.tight_layout()  # adjust layout
        ax.grid(True, which='both', axis='both', color=grid_properties['color'], linestyle=grid_properties['linestyle'],
                linewidth=grid_properties['linewidth'])

        plt.savefig(output_file, dpi=dpi)
        plt.close('all')  # close all figures

    def process_files_base(csv_directory, output_directory, examined_files=None):
        '''Function to iterate through the hourly input files and create the plots,
        excluding specified columns as defined in demands_list.'''
        if not os.path.exists(output_directory):  # check if the directory exists
            os.makedirs(output_directory)  # generate the folder if not existed

        csv_files = [file for file in os.listdir(csv_directory) if
                     file.startswith('h_')]  # list with all monthly csv files
        if examined_files is not None:
            csv_files = [file for file in csv_files if file in examined_files]
        if not csv_files:
            print("No CSV files found in the directory")
            return

        for file in csv_files:  # loop through each csv file
            file_path = os.path.join(csv_directory, file)
            df = pd.read_csv(file_path)

            if df.empty:
                print(f"No data in {file}, creating placeholder plot.")
                output_file = os.path.join(output_directory, f'{file[:-4]}_NoData.png')
                create_placeholder_plot(output_file)
                continue

            if not pd.api.types.is_datetime64_any_dtype(df['Time']):  # check if 'Time' column is datetime
                df['Time'] = pd.to_datetime(df['Time'], format=known_format)

            for column in df.columns[1:]:  # loop through columns except 'Time'
                if column in exclude_columns:  # skip excluded columns
                    print(f"Skipping column {column} in {file} as it is in exclude_columns.")
                    continue

                if df[column].isna().all():
                    print(f"All values are NaN in {file}, column {column}, creating placeholder plot.")
                    output_file = os.path.join(output_directory, f'h_{column}_NoData.png')
                    create_placeholder_plot(output_file)
                    continue
                output_file = os.path.join(output_directory, f'h_{column}.png')
                create_scatter_plot_base(df['Time'], df[column], column, output_file)

        print("All hourly plots have been saved.")  # confirmation

    def process_files_daily(csv_directory, output_directory, examined_files=None):
        '''function to iterate through the daily input files and create the plots
        as defined in create_scatter_plot function'''
        if not os.path.exists(output_directory):  # check if the directory exists
            os.makedirs(output_directory)  # generate the folder if not existed

        csv_files = [file for file in os.listdir(csv_directory) if file.startswith('d_')]  # list with daily csv files
        if examined_files is not None:
            csv_files = [file for file in csv_files if file in examined_files]
        if not csv_files:
            print("No CSV files found in the directory")
            return

        for file in csv_files:  # loop through each csv file
            file_path = os.path.join(csv_directory, file)
            df = pd.read_csv(file_path, dtype={'Time': str})  # read 'Time' column as string
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):  # check if 'Time' column is datetime
                df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y')
            for column in df.columns[1:]:  # loop through columns except 'Time'
                if column in exclude_columns:  # skip excluded columns
                    print(f"Skipping column {column} in {file} as it is in exclude_columns.")
                    continue

                if column == 'Is':
                    output_file_bars = os.path.join(output_directory, f'd_{column}_bar.png')
                    create_bar_plot_daily(df['Time'], df[column], column, output_file_bars)
                else:
                    output_file = os.path.join(output_directory, f'd_{column}.png')
                    create_scatter_plot_daily(df['Time'], df[column], column, output_file)

        print("All daily plots have been saved.")  # confirmation

    def process_files_monthly(csv_directory, output_directory, examined_files=None):
        '''function to iterate through the monthly input files and create the plots
        as defined in create_scatter_plot_monthly function'''
        if not os.path.exists(output_directory):  # check if the directory exists
            os.makedirs(output_directory)  # generate the folder if not existed

        csv_files = [file for file in os.listdir(csv_directory) if file.startswith('m_')]  # list with monthly csv files
        if examined_files is not None:
            csv_files = [file for file in csv_files if file in examined_files]
        if not csv_files:
            print("No CSV files found in the directory")
            return

        for file in csv_files:  # loop through each csv file
            file_path = os.path.join(csv_directory, file)
            df = pd.read_csv(file_path, dtype={'Time': str})  # read 'Time' column as string
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):  # check if 'Time' column is datetime
                df['Time'] = pd.to_datetime(df['Time'], format='%m.%Y')
            for column in df.columns[1:]:  # loop through columns except 'Time'
                if column in exclude_columns:  # skip excluded columns
                    print(f"Skipping column {column} in {file} as it is in exclude_columns.")
                    continue

                if column == 'Is':
                    output_file_bars = os.path.join(output_directory, f'm_{column}_bar.png')
                    create_bar_plot_monthly(df['Time'], df[column], column, output_file_bars)
                else:
                    output_file = os.path.join(output_directory, f'm_{column}.png')
                    create_scatter_plot_monthly(df['Time'], df[column], column, output_file)

        print("All monthly plots have been saved.")  # confirmation

    def process_files_extreme(csv_directory, output_directory, examined_files=None):
        '''function to iterate through the hourly input files and create the plots
        as defined in create_scatter_plot_base function'''
        if not os.path.exists(output_directory):  # check if the directory exists
            os.makedirs(output_directory)  # generate the folder if not existed

        csv_files = [file for file in os.listdir(csv_directory)]  # list with hourly csv files
        if examined_files is not None:
            csv_files = [file for file in csv_files if file in examined_files]
        if not csv_files:
            print("No CSV files found in the directory")
            return

        for file in csv_files:  # loop through each csv file
            file_path = os.path.join(csv_directory, file)
            print(f"Processing {file}...")

            try:
                df = pd.read_csv(file_path, encoding='utf-8')  # specify encoding
            except UnicodeDecodeError:
                print(f"UTF-8 decoding failed for {file}. Trying ISO-8859-1.")
                try:
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')
                except Exception as e:
                    print(f"Failed to read {file} with error: {e}. Skipping.")
                    continue
            except Exception as e:
                print(f"Error reading {file}: {e}. Skipping.")
                continue

            if df.empty:
                print(f"No data in {file}, creating placeholder plot.")
                output_file = os.path.join(output_directory, f'{file[:-4]}_NoData.png')
                create_placeholder_plot(output_file)
                continue

            if not pd.api.types.is_datetime64_any_dtype(df['Time']):  # check if 'Time' column is datetime
                df['Time'] = pd.to_datetime(df['Time'], format=known_format)

            for column in df.columns[1:]:  # loop through columns except 'Time'
                if column in exclude_columns:  # skip excluded columns
                    print(f"Skipping column {column} in {file} as it is in exclude_columns.")
                    continue

                if df[column].isna().all():
                    print(f"All values are NaN in {file}, column {column}, creating placeholder plot.")
                    output_file = os.path.join(output_directory, f'{file[:-4]}_{column}_NoData.png')
                    create_placeholder_plot(output_file)
                    continue
                output_file = os.path.join(output_directory, f'{file[:-4]}_{column}.png')
                create_scatter_plot_base(df['Time'], df[column], column, output_file)

        print("All hourly plots have been saved.")  # confirmation


    directory_others_touse = os.path.join(directory_others, 'to_use')  # path to 'to_use' folder
    known_d_format = '%d.%m.%Y'     # daily format for averaging
    known_m_format = '%m.%Y'        # monthly format for averaging

    process_files_base(directory, directory_graphs_01)  # call the function to process files
    process_files_daily(directory, directory_graphs_01)  # call the function to process daily files
    process_files_monthly(directory, directory_graphs_01)  # call the function to process monthly files
    process_files_extreme(directory_others_touse, directory_others_touse)  # call the function to process extreme files

    _11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
    _11_initial_csv_check.create_folders(directory_graphs_01,
                                             folders=['01_hourly', '02_daily', '03_monthly', '04_extreme'])
    out_h = os.path.join(directory_graphs_01, '01_hourly')
    out_d = os.path.join(directory_graphs_01, '02_daily')
    out_m = os.path.join(directory_graphs_01, '03_monthly')
    out_ext = os.path.join(directory_graphs_01, '04_extreme')

    d_files = []    # list for graphs with daily results
    for file_name in os.listdir(directory_graphs_01):  # iterate over files
        if file_name.endswith('.png') and file_name.startswith('d_'):    # argumnets: .png and '_d'
            d_files.append(file_name)       # extend the list
    for d_file_name in d_files:     # loop for daily files
        _11_initial_csv_check.move_files(directory_graphs_01,out_d,file_name=d_file_name)  # moving files

    m_files = []    # list for graphs with monlty results
    for file_name in os.listdir(directory_graphs_01):  # iterate over files
        if file_name.endswith('.png') and file_name.startswith('m_'):    # argumnets: .png and '_m'
            m_files.append(file_name)       # extend the list
    for m_file_name in m_files:     # loop for daily files
        _11_initial_csv_check.move_files(directory_graphs_01,out_m,file_name=m_file_name)  # moving files

    _11_initial_csv_check.move_files(directory_graphs_01, out_h, file_extension='*.png')  # moving remaining files

    if solar_radiation == 'y':
        prefixes = ['EWW', 'ESW', 'EHW', 'EDW', 'ESunW_p', 'ESunW_n']
    else:
        prefixes = ['EWW', 'ESW', 'EHW', 'EDW']
    ext_files = []  # list for graphs with results for extreme weeks
    for file_name in os.listdir(directory_others_touse):  # iterate over files
        for prefix in prefixes:  # loop for daily files
            if file_name.endswith('.png') and prefix in file_name:  # argumnets: .png and '_d'
                ext_files.append(file_name)  # extend the list

    for ext_file_name in ext_files:  # loop for daily files
        _11_initial_csv_check.move_files(directory_others_touse, out_ext, file_name=ext_file_name)


'''APPLICATION'''
if __name__ == "__main__":
    main()
