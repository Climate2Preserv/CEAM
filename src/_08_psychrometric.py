import psychrolib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import importlib
from _07_CC_file import climate_classes
from _00_CEAM_run import directory
from _11_initial_csv_check import directory_graphs_04


def plot_psychrometric_chart(directory, input_file, header_x, header_y, T_og, RH_og, climate_class_symbol=None):
    '''function to generate the psychometric charts'''
    psychrolib.SetUnitSystem(psychrolib.SI)
    pressure = 101325

    t_array = np.arange(0, 40, 0.1)
    rh_array = np.arange(0, 1.2, 0.2)
    rh_array_sub = np.arange(0.1, 1.1, 0.2)
    twb_array = np.arange(-10, 45, 5)

    climate_class = next((cls for cls in climate_classes if cls['name'] == climate_class_symbol), None)
    if not climate_class:
        raise ValueError(f"Invalid class name: {climate_class_symbol}")

    Ti_outer_min = climate_class['Ti_outer_min']
    Ti_outer_max = climate_class['Ti_outer_max']
    RHi_outer_min = climate_class['RHi_outer_min']
    RHi_outer_max = climate_class['RHi_outer_max']
    Ti_seasonal_offset_p = climate_class.get('Ti_seasonal_offset_p')
    Ti_seasonal_offset_n = climate_class.get('Ti_seasonal_offset_n')
    RHi_seasonal_offset_p = climate_class.get('RHi_seasonal_offset_p')
    RHi_seasonal_offset_n = climate_class.get('RHi_seasonal_offset_n')

    data_file = os.path.join(directory, input_file)         # read input data
    data = pd.read_csv(data_file, delimiter=',')

    # assuming the correct column names are header_x (T) and header_y (RH)
    T_data = data[header_x]
    RH_data = data[header_y]

    # convert the columns to numeric values - verification
    T_og = data[T_og].astype(float)
    RH_og = data[RH_og].astype(float)

    avg_T = np.mean(T_og)       # average T values
    avg_RH = np.mean(RH_og)     # average RH values

    avg_HR = psychrolib.GetHumRatioFromRelHum(avg_T, avg_RH / 100, pressure)    # conversion
    HR_data = [psychrolib.GetHumRatioFromRelHum(T, RH / 100, pressure) for T, RH in zip(T_data, RH_data)]   # conversion

    # initialize inside and outside data lists
    inside_T_data = []
    inside_HR_data = []
    outside_T_data = []
    outside_HR_data = []

    # determine if points are inside or outside the bounds
    for T, RH, HR in zip(T_data, RH_data, HR_data):
        is_inside = True

        # check long-term bounds
        if (Ti_outer_min is not None and T < Ti_outer_min) or (Ti_outer_max is not None and T > Ti_outer_max):
            is_inside = False
        if (RHi_outer_min is not None and RH < RHi_outer_min) or (RHi_outer_max is not None and RH > RHi_outer_max):
            is_inside = False

        # check seasonal bounds
        if (Ti_seasonal_offset_n is not None and T < (avg_T - Ti_seasonal_offset_n)) or \
                (Ti_seasonal_offset_p is not None and T > (avg_T + Ti_seasonal_offset_p)):
            is_inside = False
        if (RHi_seasonal_offset_n is not None and RH < (avg_RH - RHi_seasonal_offset_n)) or \
                (RHi_seasonal_offset_p is not None and RH > (avg_RH + RHi_seasonal_offset_p)):
            is_inside = False

        # add the point to the corresponding list (for each bound)
        if is_inside:
            inside_T_data.append(T)
            inside_HR_data.append(HR)
        else:
            outside_T_data.append(T)
            outside_HR_data.append(HR)

    figsize = (12, 8)
    f, ax = plt.subplots(figsize=figsize, dpi=300)

    # plot constant relative humidity lines
    for rh in rh_array:
        hr_array = [psychrolib.GetHumRatioFromRelHum(t, rh, pressure) for t in t_array]
        ax.plot(t_array, hr_array, 'k', linestyle='-', linewidth=1.0)
        idx = np.abs(t_array).argmin()
        ax.annotate(f'RH$_{{{int(rh * 100)}}}$', xy=(t_array[idx], hr_array[idx]), xytext=(-5, 0),
                    textcoords='offset points', fontsize=6, fontweight='bold', color='black', ha='right')

    for rh in rh_array_sub:
        hr_array = [psychrolib.GetHumRatioFromRelHum(t, rh, pressure) for t in t_array]
        ax.plot(t_array, hr_array, 'k', linestyle='-', linewidth=0.5)
        idx = np.abs(t_array).argmin()

    rh_values_long = []         # prepare RH values based on climate class offsets
    if RHi_outer_min is not None:
        rh_values_long.append(RHi_outer_min / 100)
    if RHi_outer_max is not None:
        rh_values_long.append(RHi_outer_max / 100)

    # calculate seasonal RH values only if offsets are not None
    rh_values_season = []
    if RHi_seasonal_offset_n is not None or RHi_seasonal_offset_p is not None:
        if RHi_seasonal_offset_n is not None:
            rh_values_season.append((avg_RH - RHi_seasonal_offset_n) / 100)
        if RHi_seasonal_offset_p is not None:
            rh_values_season.append((avg_RH + RHi_seasonal_offset_p) / 100)

    line_colors_long = ['#b22222', '#b22222']
    line_colors_season = ['#77A5C9', '#77A5C9']

    for rh, color in zip(rh_values_long, line_colors_long):
        hr_array = [psychrolib.GetHumRatioFromRelHum(t, rh, pressure) for t in t_array]
        ax.plot(t_array, hr_array, color=color, linestyle='-', linewidth=2.5)

    for rh, color in zip(rh_values_season, line_colors_season):
        hr_array = [psychrolib.GetHumRatioFromRelHum(t, rh, pressure) for t in t_array]
        ax.plot(t_array, hr_array, color=color, linestyle='-', linewidth=2.5)

    # plot constant wet-bulb temperature lines
    first_label = True
    for twb in twb_array:
        hr_array = []
        t_plot_array = []
        for t in t_array:
            if twb <= t:
                hr = psychrolib.GetHumRatioFromTWetBulb(t, twb, pressure)
                hr_array.append(hr)
                t_plot_array.append(t)
        if first_label:
            ax.plot(t_plot_array, hr_array, 'dimgrey', linestyle='--', linewidth=1.0,
                    label='Wet-bulb temperature [$^o$C]')
            first_label = False
        else:
            ax.plot(t_plot_array, hr_array, 'dimgrey', linestyle='--', linewidth=1.0, label='_nolegend_')

    # plot inside and outside data points
    ax.scatter(inside_T_data, inside_HR_data, label='Inside data', marker='o', color='lightgrey',
               edgecolor='dimgray',
               alpha=0.75, s=50)
    ax.scatter(outside_T_data, outside_HR_data, label='Outside data', marker='o', color='#FAF3D3',
               edgecolor='#C4A36A',
               alpha=0.5, s=25)

    # plot average point
    ax.scatter(avg_T, avg_HR, label=f'Average ({avg_T:.1f}$^o$C, {avg_RH:.1f}%)', marker='*', color='steelblue',
               edgecolor='navy',
               alpha=0.75, s=250)

    temp_values_long = []       # prepare default T values based on climate class offsets
    if Ti_outer_min is not None:
        temp_values_long.append(Ti_outer_min)
    if Ti_outer_max is not None:
        temp_values_long.append(Ti_outer_max)

    # calculate seasonal temperature values only offset is not None
    temp_values_seasons = []
    if Ti_seasonal_offset_n is not None or Ti_seasonal_offset_p is not None:
        if Ti_seasonal_offset_n is not None:
            temp_values_seasons.append(avg_T - Ti_seasonal_offset_n)
        if Ti_seasonal_offset_p is not None:
            temp_values_seasons.append(avg_T + Ti_seasonal_offset_p)

    # add vertical lines at specific temperatures
    for temp in temp_values_long:
        if temp is not None:
            ax.axvline(x=temp, color='#b22222', linestyle='--', linewidth=2.5)

    for temp in temp_values_seasons:
        if temp is not None:
            ax.axvline(x=temp, color='#77A5C9', linestyle='--', linewidth=2.5)

    handles, labels = ax.get_legend_handles_labels()        # adding the legend with custom entry for vertical lines

    # custom legend entry for vertical lines
    handles.extend([
        plt.Line2D([0], [0], color='#b22222', linestyle='--', linewidth=2.5),
        plt.Line2D([0], [0], color='#b22222', linestyle='-', linewidth=2.5),
        plt.Line2D([0], [0], color='#77A5C9', linestyle='--', linewidth=2.5),
        plt.Line2D([0], [0], color='#77A5C9', linestyle='-', linewidth=2.5)
    ])

    # long-term T label
    min_str_long = f"{Ti_outer_min:.1f}" if Ti_outer_min is not None else "None"
    max_str_long = f"{Ti_outer_max:.1f}" if Ti_outer_max is not None else "None"
    labels.append(f'T$_{{long-term}}$ ({min_str_long},{max_str_long}) [$^o$C]')

    # long-term RH label
    min_str_rh = f"{RHi_outer_min:.1f}" if RHi_outer_min is not None else "None"
    max_str_rh = f"{RHi_outer_max:.1f}" if RHi_outer_max is not None else "None"
    labels.append(f'RH$_{{long-term}}$ ({min_str_rh},{max_str_rh}) [%]')

    # include seasonal legend
    if Ti_seasonal_offset_n is not None and Ti_seasonal_offset_p is not None:
        labels.append(
            f'T$_{{seasonal}}$ ({(avg_T - Ti_seasonal_offset_n):.1f},{(avg_T + Ti_seasonal_offset_p):.1f}) [$^o$C]')
    else:
        labels.append(f'T$_{{seasonal}}$ (None,None) [$^o$C]')
    if RHi_seasonal_offset_n is not None and RHi_seasonal_offset_p is not None:
        labels.append(
            f'RH$_{{seasonal}}$ ({(avg_RH - RHi_seasonal_offset_n):.1f},{(avg_RH + RHi_seasonal_offset_p):.1f}) [%]')
    else:
        labels.append(f'RH$_{{seasonal}}$ (None,None) [%]')

    ax.set(ylim=(0, 0.025), xlim=(0, 40))
    ax.set_xlabel('Dry-bulb Temperature [$^o$C]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Humidity Ratio [kg$_{water}$/kg$_{dry air}$]', fontsize=14, fontweight='bold')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.001))
    plt.grid(which='both', color='darkgrey', linestyle=':', linewidth=0.5)

    plt.legend(handles, labels, loc='upper left', prop={'weight': 'bold', 'size': 10}, edgecolor='black',
               facecolor='white', fancybox=False, borderpad=1, borderaxespad=1)

    plt.tight_layout()

    # construct the .png file and save
    file_name = f"psychrometric_{climate_class_symbol}.png"
    save_path = os.path.join(directory, file_name)
    plt.savefig(save_path, format='png')


'''FUNCTIONALITIES'''
def main():
    input_file = 'h_comb.csv'
    climate_class = ['AA', 'A1', 'A2', 'B', 'C', 'D']  # examine all CC

    for name_class in climate_class:
        plot_psychrometric_chart(directory, input_file, 'Ti', 'RHi', 'Ti', 'RHi', name_class)

    _11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
    _11_initial_csv_check.move_files(directory, os.path.join(directory_graphs_04, '02_psychrometric'),
                                     file_extension='*.png', filename_include='psychrometric_')


'''APPLICATION'''
if __name__ == "__main__":
    main()
