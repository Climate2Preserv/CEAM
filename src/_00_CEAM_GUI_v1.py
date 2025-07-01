# CEAM (Climate and Energy Assessment of Museums)
# Copyright (C) 2025 Marcin Zygmunt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import customtkinter as ctk         # GUI library
import webbrowser                   # opening URL
import tkinter as tk                # tkinter for the hyperlink
import os
import subprocess                   # running external Python script
import threading                    # run the subprocess in a separate thread
import time                         # to get current time
import sys
import winsound                     # for playing a sound (Windows specific)
import pandas as pd
import platform
import _00_CEAM_run                 # load the _00_CEAM_run script
from tkinter import filedialog      # for opening file dialog
from tkinter import messagebox


color_basic = "#D3D3D3"             # light grey-basic
color_inactive = '#E5E5E5'          # inactive grey
color_basic_2 = "#acd8fa"           # light vanish blue (reset button)
color_default = "#77A5C9"           # vanish blue
color_run = "#50C878"               # vanish green
color_test = '#FAF3D3'              # test color set to yellowish
window_x_size = 960                 # x size
window_y_size = 600                 # y size
rows_number = 8                     # number of rows
columns_number = 12                 # number of columns
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]   # list with days
optimization_schedule_list = ["s1", "s2", "s3", "s4", "s5"]                 # list with optimization schedules
optimizers_list = ['DT', 'NN', 'LSTM', 'GRU']                               # list with optimizers
demand_list = ['D1', 'D2', 'D3', 'D4', 'D5']                                # list with available demands
freq_options = ["h", "d", "m"]                                              # frequency options
format_h_options = ["%Y.%m.%d", "%Y.%d.%m", "%m.%d.%Y", "%d.%m.%Y"]         # hour format
format_d_options = ["%Y.%m.%d", "%Y.%d.%m", "%m.%d.%Y", "%d.%m.%Y"]         # day format
format_m_options = ["%Y.%m", "%m.%Y"]                                       # month format
y_n_options = ["y", "n"]                                                    # yes/no options
unit_options = ["SI", "IU"]                                                 # unit options
normal_distribution_options = ["-1:+1", "-2:+2", "-3:+3"]                   # normal distribution options

module_times = {
    "Module 0": 30,         # estimated time of computing (in seconds) for Module 1
    "Module 1": 500,        # estimated time of computing (in seconds) for Module 2
    "Module 2": 120,        # estimated time of computing (in seconds) for Module 3
    "Module 3": 60,         # estimated time of computing (in seconds) for Module 4
    "Module 4": 120,        # estimated time of computing (in seconds) for Module 5
    "Module 5": 4800,       # estimated time of computing (in seconds) for Module 6
}

Module_0_text = f"\tModule 0: Data overview p.1 - ESSENTIAL\n" \
                "\t\tInitial verification of the provided input data, as well as its initial examination.\n" \
                "\t\tEstimated time of computing: 5-30 seconds.\n" \
                "\t\tRequired module(s): Module 0 (initialization of assessment).\n" \
                "\t\tRequired inputs: files path, data format and frequencies, units, solar radiation, demands list, outliers, T_hdd and T_cdd."
Module_1_text = "\tModule 1: Data overview p.2 - OPTIONAL\n" \
                "\t\tGraphical overview of the provided input data.\n" \
                "\t\tEstimated time of computing: 150-500 seconds.\n" \
                "\t\tRequired module(s): Module 0.\n" \
                "\t\tRequired inputs: files path, data format and frequencies, solar radiation, demands list, outliers."
Module_2_text = "\tModule 2: Correlation assessment - OPTIONAL\n" \
                "\t\tCorrelation assessment of the related factors.\n" \
                "\t\tEstimated time of computing: 30-120 seconds.\n" \
                "\t\tRequired module(s): Module 0.\n" \
                "\t\tRequired inputs: files path, solar radiation, demands list."
Module_3_text = "\tModule 3: Energy Signature - OPTIONAL\n" \
                "\t\tEnergy-related assessment by means of Energy Signature method.\n" \
                "\t\tEstimated time of computing: 30-60 seconds.\n" \
                "\t\tRequired module(s): Module 0.\n" \
                "\t\tRequired inputs: files path, demands list."
Module_4_text = "\tModule 4: Climate Classes overview - OPTIONAL\n" \
                "\t\tIndoor climate assessment following the ASHRAE Climate Classes.\n" \
                "\t\tEstimated time of computing: 30-120 seconds.\n" \
                "\t\tRequired module(s): Module 0.\n" \
                "\t\tRequired inputs: files path, data format and frequencies."
Module_5_text = "\tModule 5: Energy optimization - OPTIONAL\n" \
                "\t\tAnalysis of a potential energy savings due to the application of indoor climate management strategies," \
                " examined by means of BBM.\n" \
                "\t\tEstimated time of computing: 300-4800 seconds.\n" \
                "\t\tRequired module(s): Module 0 and Module 4 (if CC re-evaluation is selected).\n" \
                "\t\tRequired inputs: files path, data format and frequencies, country, timetable, optimization plan, solar radiation, demands list, CC-reeval."

def button_callback(button, clicked_buttons, module_index, app):
    '''button appearance and functionality on click'''
    if not clicked_buttons[module_index]:
        button.configure(fg_color=color_default, font=("Arial", 14, "bold"))  # change color to grey when off
        clicked_buttons[module_index] = True
    else:
        button.configure(fg_color=color_basic, font=("Arial", 14, "bold"))  # revert to default color when on
        clicked_buttons[module_index] = False

    app.update_textbox_content()    # update the text box content based on the module selection

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # configure window and "grid"
        self.title("CEAM - Climate and Energy Assessment for Museums")
        screen_width = self.winfo_screenwidth()  # function: screen width
        screen_height = self.winfo_screenheight()  # function: screen height
        x = (screen_width // 2) - (window_x_size // 2)  # centering x-axis calculations
        y = (screen_height // 2) - (window_y_size // 2)  # centering y-axis calculations
        self.geometry(f"{window_x_size}x{window_y_size}+{x}+{y}")  # size

        self.grid_columnconfigure((0, 1, 10, 11), weight=1)             # side columns
        self.grid_columnconfigure((2, 3, 4, 5, 6, 7, 8, 9), weight=1)   # main columns
        self.grid_rowconfigure(0, weight=0)                             # upper sidebar row
        self.grid_rowconfigure((1, 2, 3), weight=0)                     # main content rows
        self.grid_rowconfigure(4, weight=1)                             # secondary row 4
        self.grid_rowconfigure(5, weight=1)                             # secondary row 5
        self.grid_rowconfigure(6, weight=1)                             # secondary row 6
        self.grid_rowconfigure(7, weight=1)                             # secondary row 7
        self.grid_rowconfigure(8, weight=1)                             # secondary row 8
        self.grid_rowconfigure(9, weight=0)                             # bottom sidebar row

        self.clicked_buttons = [False] * 6              # tracking the clicked state of each button
        self.module_descriptions = [Module_0_text, Module_1_text, Module_2_text,
                                    Module_3_text, Module_4_text, Module_5_text]  # module descriptions
        self.directory = ""                             # directory variable
        self.directory_time_index = ""
        self._create_top_sidebar()                      # create the upper sidebar
        self._create_bottom_sidebar()                   # create the bottom sidebar
        self._create_separators()                       # add horizontal separators
        self._create_buttons()                          # create 'module' buttons
        self._create_scrollable_textbox()               # create the scrollable textbox
        self._create_path_selection()                   # create the path selection
        self.run_button = self._create_run_button()     # create the run button
        self.graphs_button = self._create_graphs_button()     # create the run button
        self.reports_button = self._create_reports_button()     # create the run button
        self.dir_button = self._create_dir_button()     # create the dir button
        self._create_lower_part()                       # create lower part of the GUI
        self.selected_schedule_data = []                # list to store selected schedule data
        self.selected_optimization_schedules = []       # list to store selected optimization schedule data
        self.selected_optimizers = []      # list to store selected optimizers
        self.selected_optimization_schedules_1 = []     # list to store the available demands
        self.setup_comment_section(self.selected_modules)

    def _create_top_sidebar(self):
        '''upper sidebar definition'''
        self.top_sidebar_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=color_basic)  # generate upper sidebar
        self.top_sidebar_frame.grid(row=0, column=0, columnspan=12, sticky="ew")  # upper sidebar positioning

        # the text content for the upper sidebar
        title = "C2P:CEAM - Climate and Energy Assessment for Museums"
        description = "Computational tool allowing for indoor climate and energy assessment in museum"
        developer = "Developed by Marcin Zygmunt, PhD in the Climate2Preserv (C2P) project:"
        info_desk = "Software info center:"
        quick_start = "Quick start:"

        # placing the info labels
        label_title = tk.Label(self.top_sidebar_frame, text=title, bg=color_basic, font=("Arial", 14, "bold"))
        label_title.pack(anchor="w", padx=20, pady=(5, 5))

        label_description = tk.Label(self.top_sidebar_frame, text=description, bg=color_basic, font=("Arial", 10))
        label_description.pack(anchor="w", padx=20, pady=(0, 0))

        frame_developer = tk.Frame(self.top_sidebar_frame, bg=color_basic)
        frame_developer.pack(anchor="w", padx=20, pady=(0, 0))
        label_developer = tk.Label(frame_developer, text=developer, bg=color_basic, font=("Arial", 10))
        label_developer.pack(side="left")
        label_web = tk.Label(frame_developer, text="web", fg="blue", bg=color_basic, cursor="hand2",
                             font=("Arial", 10, "bold"))
        label_web.pack(side="left")
        label_web.bind("<Button-1>", self._open_website)  # bind the click to open the website

        # manuals section
        frame_manuals = tk.Frame(self.top_sidebar_frame, bg=color_basic)
        frame_manuals.pack(anchor="w", padx=20, pady=(0, 5), fill="x")
        # left-aligned manuals info
        label_manuals = tk.Label(frame_manuals, text=info_desk, bg=color_basic, font=("Arial", 10))
        label_manuals.pack(side="left")
        label_manuals_link = tk.Label(frame_manuals, text="CEAM manuals", fg="green", bg=color_basic, cursor="hand2",
                                      font=("Arial", 10, "bold"))
        label_manuals_link.pack(side="left")
        label_manuals_link.bind("<Button-1>", self._open_manuals)

        separator = tk.Label(frame_manuals, text="||", bg=color_basic, font=("Arial", 10, "bold"))
        separator.pack(side="left")

        label_youtube_link = tk.Label(frame_manuals, text="YouTube-Tutorial", fg="blue", bg=color_basic,
                                      cursor="hand2", font=("Arial", 10, "bold"))
        label_youtube_link.pack(side="left")
        label_youtube_link.bind("<Button-1>", self._open_youtube_info)

        separator = tk.Label(frame_manuals, text="||", bg=color_basic, font=("Arial", 10, "bold"))
        separator.pack(side="left")

        label_github_link = tk.Label(frame_manuals, text="GitHub", fg="blue", bg=color_basic,
                                     cursor="hand2", font=("Arial", 10, "bold"))
        label_github_link.pack(side="left")
        label_github_link.bind("<Button-1>", self._open_github_info)

        # right-aligned quick manuals section
        label_quick_link = tk.Label(frame_manuals, text='YouTube-Quick Start', fg="blue", bg=color_basic,
                                    cursor="hand2", font=("Arial", 10, "bold"))
        label_quick_link.pack(side="right", padx=(0, 0))
        label_quick_link.bind("<Button-1>", self._open_youtube_quick)

        separator = tk.Label(frame_manuals, text="||", bg=color_basic, font=("Arial", 10, "bold"))
        separator.pack(side="right")

        label_quick_manuals_link = tk.Label(frame_manuals, text="Quick Start Guide", fg="green", bg=color_basic,
                                            cursor="hand2", font=("Arial", 10, "bold"))
        label_quick_manuals_link.pack(side="right", padx=(0, 0))
        label_quick_manuals_link.bind("<Button-1>", self._open_quick_manuals)

        label_text = tk.Label(frame_manuals, text=quick_start, bg=color_basic, font=("Arial", 10))
        label_text.pack(side="right", padx=(0, 0))

    def _open_manuals(self, event):
        '''open the manuals link in the default web browser'''
        try:
            url = "https://drive.google.com/file/d/14pi2nRSRpZwOy8r0y87SqgOQ7U6qvplu/view"
            # url = "https://drive.google.com/file/d/1r2qyEid3o0hw_SW0lVqwrlV0M-fiwahD/view?usp=drive_link"
            webbrowser.open(url)
        except Exception as e:
            print(f"Error opening manual: {e}")

    def _open_quick_manuals(self, event):
        '''open the quick start manuals link in the default web browser'''
        try:
            url = "https://drive.google.com/file/d/1HdxbHnI0BT-e0AznzwNTVAQfMDlq8crr/view"
            # url = "https://drive.google.com/file/d/1r2qyEid3o0hw_SW0lVqwrlV0M-fiwahD/view?usp=drive_link"
            webbrowser.open(url)
        except Exception as e:
            print(f"Error opening manual: {e}")

    def _create_bottom_sidebar(self):
        '''bottom sidebar definition'''
        self.bottom_sidebar_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=color_basic)   # generate bottom sidebar
        self.bottom_sidebar_frame.grid(row=9, column=0, columnspan=12, sticky="ew")             # bottom sidebar positioning

        # the text content for the bottom sidebar
        contact_text = "Need help? "
        version_text = "v.1.2.1   01.07.2025"

        label_link = tk.Label(self.bottom_sidebar_frame, text='Contact info', fg="blue", bg=color_basic, cursor="hand2",
                              font=("Arial", 10, "bold"))
        label_link.pack(side="right", padx=(0, 20))
        label_link.bind("<Button-1>", self._open_contact_info)      # bind the click to open the link

        label_message = tk.Label(self.bottom_sidebar_frame, text='Message', fg="blue", bg=color_basic, cursor="hand2",
                                 font=("Arial", 10, "bold"))
        label_message.pack(side="right", padx=(0, 10))
        label_message.bind("<Button-1>", self.open_mail_client)

        label_text = tk.Label(self.bottom_sidebar_frame, text=contact_text, bg=color_basic, font=("Arial", 10, "bold"))
        label_text.pack(side="right", padx=(0, 10))

        label_text_version = tk.Label(self.bottom_sidebar_frame, text=version_text, bg=color_basic, font=("Arial", 8))
        label_text_version.pack(side="left", padx=(20, 0))

    def open_mail_client(self, event):
        '''opens the mailbox client'''
        email_address = "climate2preserv@gmail.com"
        mailto_link = f"mailto:{email_address}"
        webbrowser.open(mailto_link)

    def _create_buttons(self):
        '''create and place module buttons'''
        for i in range(6):
            button = ctk.CTkButton(self, text=f"Module {i}",
                                   font=("Arial", 14, "bold"),      # font
                                   fg_color=color_basic,            # color
                                   border_color="black",            # border color
                                   border_width=1)                  # border width
            button.configure(command=lambda i=i, b=button: button_callback(b, self.clicked_buttons, i, self))   # generate button
            button.grid(row=1, column=2*i, columnspan=2, padx=5, pady=10, sticky="ew")                          # button positioning

    def _create_scrollable_textbox(self):
        '''create a frame with the text box and scrollbar'''
        text_frame = tk.Frame(self, highlightbackground="black", highlightthickness=1, bd=0)    # border
        text_frame.grid(row=2, column=0, columnspan=12, sticky="nsew", padx=20, pady=0)         # positioning

        # create the scrollable textbox
        self.textbox = tk.Text(text_frame, wrap="word", height=10, font=("Arial", 12), state="disabled")
        self.textbox.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(text_frame, command=self.textbox.yview)    # create a vertical scrollbar
        scrollbar.pack(side="right", fill="y")                              # scrollbar positioning
        self.textbox.config(yscrollcommand=scrollbar.set)                   # configure the text widget
        self.update_textbox_content()                                       # insert initial text into the textbox

    def update_textbox_content(self):
        '''the base text in the main textbox'''

        base_text = ("Welcome to the C2P:CEAM tool! This tool provides insights into indoor climate and energy "
                     "assessment for museums.\n"
                     "Please select modules for the desired scope of analysis: look in manuals (green hyperlink)"
                     " for the complex description of each module functionality.\n\n")
        mid_text = "The selected scope of analysis consists of:\n"
        end_text = ("\n\nYou can also reach out for help or send feedback via the contact link (blue hyperlink) "
                    "at the bottom right.\n"
                    "More information about the Climate2Preserv project can be found on "
                    "the C2P website (web hyperlink) in the header.")

        selected_modules = [desc for i, desc in enumerate(self.module_descriptions)
                            if self.clicked_buttons[i]]

        self.selected_modules = selected_modules  # save the selection

        self.textbox.config(state="normal")
        self.textbox.delete("1.0", "end")  # clear existing content

        self.textbox.insert("1.0", base_text)

        if selected_modules:
            # insert selected modules
            self.textbox.insert("end", mid_text + "\n".join(selected_modules) + end_text)
        else:
            self.textbox.insert("end", "\tWarning: no analysis is selected!!", "bold")
            self.textbox.insert("end", end_text)

        self.textbox.tag_config("bold", font=("Arial", 14, "bold"))

        self.textbox.config(state="disabled")

        self.setup_comment_section(selected_modules)

    def setup_comment_section(self, selected_modules):
        self.comment_textbox = ctk.CTkTextbox(self,
                                              height=1,
                                              width=40,
                                              bg_color="white",
                                              text_color="black",
                                              font=("Arial", 12, "bold"))

        # calculate estimated time based on selected modules
        total_time = 0
        for module_desc in selected_modules:
            module_name = module_desc.split(":")[0].strip()
            if module_name in module_times:
                total_time += module_times[module_name]

        # the initial text in the textbox
        if total_time == 0:
            self.comment_textbox.insert("0.0",
                                        f"   Waiting for analysis to start but none of the modules is selected...")
        else:
            self.comment_textbox.insert("0.0",
                                        f"   Waiting for analysis to start! Computing should take no longer than {total_time} seconds.")
        self.comment_textbox.configure(state="disabled")

        self.comment_textbox.grid(row=8, column=0, columnspan=9, padx=0, pady=0, sticky="ew")   # placement

    def _create_path_selection(self):
        '''display the selected path'''
        self.path_entry = tk.Entry(self, font=("Arial", 12), highlightbackground="black",
                                   highlightthickness=1)                                            # template
        self.path_entry.grid(row=3, column=0, columnspan=10, padx=20, pady=10, sticky="nsew")       # positioning

        # browse button
        browse_button = ctk.CTkButton(self, text="Browse",
                                      font=("Arial", 14, "bold"),                               # font
                                      fg_color=color_default,                                   # color
                                      border_color="black", border_width=1,                     # border
                                      command=self._browse_directory)                           # function
        browse_button.grid(row=3, column=10, columnspan=2, padx=5, pady=10, sticky="ew")        # positioning

    def _browse_directory(self):
        '''open file dialog to select a directory'''
        self.directory = filedialog.askdirectory()
        if self.directory:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, self.directory)
            self.graphs_folder = os.path.join(self.directory, "Graphs")         # path for Graphs folder
            self.reports_folder = os.path.join(self.directory, "Reports")       # path for Reports folder

            # debugging
            print(f"Selected Directory: {self.directory}")
            print(f"Graphs Folder Exists at Top Level: {os.path.isdir(self.graphs_folder)}")
            print(f"Reports Folder Exists at Top Level: {os.path.isdir(self.reports_folder)}")

            if os.path.isdir(self.graphs_folder) and os.path.basename(self.graphs_folder) == "Graphs":
                self.graphs_button.configure(fg_color=color_run, state="normal")        # enable Graphs button
            else:
                self.graphs_button.configure(fg_color=color_basic, state="disabled")    # disable Graphs button

            if os.path.isdir(self.reports_folder) and os.path.basename(self.reports_folder) == "Reports":
                self.reports_button.configure(fg_color=color_run, state="normal")       # enable Reports button
            else:
                self.reports_button.configure(fg_color=color_basic, state="disabled")   # disable Reports button

            if self.directory:
                self.dir_button.configure(fg_color=color_run, state="normal")
            else:
                self.dir_button.configure(fg_color=color_basic, state="disabled")

            self.run_button.configure(fg_color=color_run, state="normal")               # enable Run button
            self.update_textbox_content()                   # update the textbox content
        else:
            self.run_button.configure(fg_color=color_basic, state="disabled")
            self.graphs_button.configure(fg_color=color_basic, state="disabled")
            self.reports_button.configure(fg_color=color_basic, state="disabled")
            self.dir_button.configure(fg_color=color_basic, state="disabled")

    def _create_run_button(self):
        '''create a Run button'''
        run_button = ctk.CTkButton(self, text="Run",
                                   font=("Arial", 14, "bold"),  # font
                                   fg_color=color_basic,  # color (inactive)
                                   border_color="black", border_width=2.5,  # border
                                   state="disabled",  # initial status
                                   command=self._run_analysis)                             # function
        run_button.grid(row=7, column=10, columnspan=2, padx=5, pady=10, sticky="ew")    # positioning
        return run_button

    def _open_dir_folder(self):
        '''open the Dir. folder in the file explorer'''
        if os.path.isdir(self.directory):
            os.startfile(self.directory)
        else:
            print("Dir folder does not exist.")

    def _open_graphs_folder(self):
        '''open the Graphs folder in the file explorer'''
        if os.path.isdir(self.graphs_folder):
            os.startfile(self.graphs_folder)
        else:
            print("Graphs folder does not exist.")

    def _open_reports_folder(self):
        '''open the Reports folder in the file explorer'''
        if os.path.isdir(self.reports_folder):
            os.startfile(self.reports_folder)
        else:
            print("Reports folder does not exist.")

    def _create_graphs_button(self):
        '''create a button to open the Graphs folder'''
        graphs_button = ctk.CTkButton(self, text="Graphs",
                                      font=("Arial", 14, "bold"),  # font
                                      fg_color=color_basic,  # color (inactive)
                                      border_color="black", border_width=1,  # border
                                      state="disabled",  # initial status
                                      command=self._open_graphs_folder)  # function for opening the Graphs folder
        graphs_button.grid(row=8, column=10, columnspan=1, padx=5, pady=10, sticky="ew")  # positioning
        self.graphs_button = graphs_button  # store as an attribute
        return graphs_button

    def _create_reports_button(self):
        '''create a button to open the Reports folder'''
        reports_button = ctk.CTkButton(self, text="Reports",
                                       font=("Arial", 14, "bold"),  # font
                                       fg_color=color_basic,  # color (inactive)
                                       border_color="black", border_width=1,  # border
                                       state="disabled",  # initial status
                                       command=self._open_reports_folder)  # function for opening the Reports folder
        reports_button.grid(row=8, column=11, columnspan=1, padx=5, pady=10, sticky="ew")  # positioning
        self.reports_button = reports_button  # store as an attribute
        return reports_button

    def _create_dir_button(self):
        '''create a button to open the Dir. folder'''
        dir_button = ctk.CTkButton(self, text="Dir.",
                                       font=("Arial", 14, "bold"),  # font
                                       fg_color=color_basic,  # color (inactive)
                                       border_color="black", border_width=1,  # border
                                       state="disabled",  # initial status
                                       command=self._open_dir_folder)  # function for opening the Reports folder
        dir_button.grid(row=8, column=9, columnspan=1, padx=5, pady=10, sticky="ew")  # positioning
        self.dir_button = dir_button  # store as an attribute
        return dir_button

    def _create_separators(self):
        '''create a horizontal separators'''
        separator_top = tk.Frame(self, height=2.5, bg="black")                                # black line
        separator_top.grid(row=0, column=0, columnspan=12, sticky="ew", pady=(105, 0))      # positioning

        separator_bottom = tk.Frame(self, height=2.5, bg="black")                             # black line
        separator_bottom.grid(row=9, column=0, columnspan=12, sticky="ew", pady=(0, 20))    # positioning

        separator_bottom_3 = tk.Frame(self, height=2.5, bg="black")  # black line
        separator_bottom_3.grid(row=3, column=0, columnspan=12, sticky="ew", pady=(55, 0))  # positioning

    def _open_website(self, event):
        '''link for web source'''
        webbrowser.open("https://www.kikirpa.be/en/projects/climate2preserv")

    def _open_contact_info(self, event):
        '''link for contact source'''
        webbrowser.open("https://www.linkedin.com/in/marcin-zygmunt-06596115a/")

    def _open_youtube_info(self, event):
        '''link for Youtube source'''
        webbrowser.open("https://www.youtube.com/watch?v=NRkTPR9I4WA&list=PLacjK7kX27AdleEQpv0C3xwy2wcnEp9mS")

    def _open_youtube_quick(self, event):
        '''link for Youtube source'''
        webbrowser.open("https://youtu.be/CNYjmvSHAw8")

    def _open_github_info(self, event):
        '''link for GitHub source'''
        webbrowser.open("https://github.com/Climate2Preserv")

    def _create_lower_part(self):
        '''definition of lower part of the GUI'''
        headers_row4 = {1: "ICD", 2: "ECD", 3: "ED", 4: "Country", 6: "City"}       # headers for row 4
        option_menu_width = window_x_size / (columns_number+1)                      # option menu width

        # tracking variables for inputs
        self.freq_vars = [ctk.StringVar(value="") for _ in range(3)]    # for frequencies input
        self.country_var = ctk.StringVar()                              # for Country input
        self.city_var = ctk.StringVar()                                 # for City input
        self.thdd_var = ctk.StringVar()                                 # for T_HDD input
        self.tcdd_var = ctk.StringVar()                                 # for T_CDD input
        self.row6_menus = []                                            # for references to row 6 option menus

        def update_row6_options(col):
            '''function to update row data format options based on the given frequencies'''
            selected_freq = self.freq_vars[col - 1].get()
            if selected_freq == "h":
                new_options = format_h_options
            elif selected_freq == "d":
                new_options = format_d_options
            elif selected_freq == "m":
                new_options = format_m_options
            else:
                new_options = []

            self.row6_menus[col - 1].configure(values=new_options)
            self.row6_menus[col - 1].set("")

        def reset_inputs():
            '''function to reset all the inputs of the lower part of GUI'''
            for var in self.freq_vars:      # reset frequencies
                var.set("")
            for menu in self.row6_menus:    # reset the data formats
                menu.set("")
                menu.configure(values=[])

            self.country_var.set("")                # reset country input
            self.city_var.set("")                   # reset city input
            self.thdd_var.set("")                   # reset T_HDD input
            self.tcdd_var.set("")                   # reset T_CDD input
            self.unit_menu.set("")                  # reset unit option menu
            self.solar_radiation_menu.set("")       # reset solar radiation option
            self.cc_reeval_menu.set("")      # reset solar radiation option2
            self.outlier_menu.set("")               # reset outlier option
            self.normal_distribution_menu.set("")   # reset normal distribution option

            self.normal_distribution_label.grid_remove()    # hide normal distribution label
            self.normal_distribution_menu.grid_remove()     # hide normal distribution menu

            self.selected_schedule_data = []                # clear the selected schedule data list
            self.selected_optimization_schedules = []       # clear the selected optimization schedule data list
            self.selected_optimizers = []      # clear the selected optimizers
            self.selected_optimization_schedules_1 = []     # clear the list of the available demands

        for col in range(12):
            self.grid_columnconfigure(col, weight=1, uniform="grid_col")    # adjust the grid size


            # functionalities for row 4
            if col in headers_row4:         # headers
                label = ctk.CTkLabel(self, text=headers_row4[col],
                                     font=("Arial", 14, "bold"),
                                     fg_color=color_basic, corner_radius=0)
                label.grid(row=4, column=col, padx=5, pady=5, sticky="nsew")

            elif col == 5:                  # country textbox
                country_entry = ctk.CTkEntry(self, textvariable=self.country_var,
                                             font=("Arial", 12),
                                             width=option_menu_width,
                                             border_width=2, border_color="black", corner_radius=0,     # borders
                                             fg_color="white")                                          # background color
                country_entry.grid(row=4, column=col, padx=5, pady=5, sticky="ew")
            elif col == 7:                  # city textbox
                city_entry = ctk.CTkEntry(self, textvariable=self.city_var,
                                          font=("Arial", 12),
                                          width=option_menu_width,
                                          border_width=2, border_color="black", corner_radius=0,    # borders
                                          fg_color="white")                                         # background color
                city_entry.grid(row=4, column=col, padx=5, pady=5, sticky="ew")
            elif col == 8:                  # operation schedule
                schedule_button = ctk.CTkButton(self, text="Timetable", font=("Arial", 14, "bold"),
                                                border_width=1, border_color="black", corner_radius=5,
                                                fg_color=color_default,
                                                command=self.open_schedule_window)
                schedule_button.grid(row=4, column=col, columnspan=2, padx=5, pady=5, sticky="ew")
            elif col == 10:                  # row 10: schedule, optimizers, and cc_re-eval buttons
                schedule_button = ctk.CTkButton(self, command=self.open_optimization_schedule_window,
                                                font=("Arial", 14, "bold"), text="Optimization Plan",
                                                corner_radius=5, fg_color=color_default,
                                                border_color="black", border_width=1)
                schedule_button.grid(row=4, column=col, columnspan=2, padx=5, pady=5, sticky="ew")

                schedule_button2 = ctk.CTkButton(self, command=self.open_optimizers_window,
                                                 font=("Arial", 14, "bold"), text="Optimizer(s)",
                                                 corner_radius=5, fg_color=color_default,
                                                 border_color="black", border_width=1)
                schedule_button2.grid(row=5, column=col, columnspan=2, padx=5, pady=5, sticky="ew")

                solar_button2 = ctk.CTkButton(self, command=self.open_cc_reeval_window,
                                              font=("Arial", 14, "bold"), text="CC re-eval",
                                              corner_radius=5, fg_color=color_default,
                                              border_color="black", border_width=1)
                solar_button2.grid(row=6, column=col, columnspan=2, padx=5, pady=5, sticky="ew")
            elif col == 0:                  # TIG button
                time_index_button = ctk.CTkButton(
                    self,
                    text="TIG",
                    font=("Arial", 14, "bold"),
                    fg_color=color_basic_2,
                    border_color="black",
                    border_width=1,
                    corner_radius=5,
                    command=self.open_time_index_window
                )
                time_index_button.grid(row=4, column=col, padx=5, pady=5, sticky="ew")

            # functionalities for row 5
            if col == 0:                    # frequency header
                label = ctk.CTkLabel(self, text="Freq.", font=("Arial", 14, "bold"),
                                     fg_color=color_basic, corner_radius=0)
                label.grid(row=5, column=col, padx=5, pady=5, sticky="nsew")
            elif col == 4:                  # units header
                label = ctk.CTkLabel(self, text="Units", font=("Arial", 14, "bold"),
                                     fg_color=color_basic, corner_radius=0)
                label.grid(row=5, column=col, padx=5, pady=5, sticky="nsew")
            elif col == 8:                  # to be added in future versions
                label = ctk.CTkLabel(self, text="Seasons", font=("Arial", 14, "bold"),
                                     fg_color=color_inactive, corner_radius=0, text_color=color_basic)
                label.grid(row=5, column=col, columnspan=2, padx=5, pady=5, sticky="nsew")
            elif col in {1, 2, 3}:          # frequency options
                option_menu = ctk.CTkOptionMenu(self, variable=self.freq_vars[col - 1], values=freq_options,
                                                command=lambda value, col=col: update_row6_options(col),
                                                dynamic_resizing=False, width=option_menu_width,
                                                font=("Arial", 12, "bold"), anchor='center',
                                                corner_radius=0, fg_color="white", text_color="black",
                                                dropdown_fg_color="white", dropdown_text_color="black",
                                                dropdown_hover_color=color_test,
                                                button_color=color_default, button_hover_color=color_test)
                option_menu.set("")
                option_menu.grid(row=5, column=col, padx=5, pady=5, sticky="ew")
            elif col == 5:                  # unit options
                self.unit_menu = ctk.CTkOptionMenu(self, values=unit_options,
                                                   dynamic_resizing=False, width=option_menu_width,
                                                   font=("Arial", 12, "bold"), anchor='center',
                                                   corner_radius=0, fg_color="white", text_color="black",
                                                   dropdown_fg_color="white", dropdown_text_color="black",
                                                   dropdown_hover_color=color_test,
                                                   button_color=color_default, button_hover_color=color_test)
                self.unit_menu.set("")
                self.unit_menu.grid(row=5, column=col, padx=5, pady=5, sticky="ew")

            # functionalities for row 6
            if col == 0:                    # known format header
                label = ctk.CTkLabel(self, text="Known Format", font=("Arial", 14, "bold"),
                                     fg_color=color_basic, corner_radius=0,
                                     wraplength=option_menu_width)
                label.grid(row=6, column=col, padx=5, pady=5, sticky="nsew")
            elif col == 4:                  # T_HDD header
                label = ctk.CTkLabel(self, text="T_HDD", font=("Arial", 14, "bold"),
                                     fg_color=color_basic, corner_radius=0)
                label.grid(row=6, column=col, padx=5, pady=5, sticky="nsew")
            elif col == 6:                  # T_CDD header
                label = ctk.CTkLabel(self, text="T_CDD", font=("Arial", 14, "bold"),
                                     fg_color=color_basic, corner_radius=0)
                label.grid(row=6, column=col, padx=5, pady=5, sticky="nsew")
            elif col == 8:                  # to be added in future versions
                label = ctk.CTkLabel(self, text="Inner gains", font=("Arial", 14, "bold"),
                                     fg_color=color_inactive, corner_radius=0, text_color=color_basic)
                label.grid(row=6, column=col, columnspan=2, padx=5, pady=5, sticky="nsew")
            elif col in {1, 2, 3}:          # known format options
                option_menu = ctk.CTkOptionMenu(self, values=[],
                                                dynamic_resizing=False, width=option_menu_width,
                                                font=("Arial", 12, "bold"),
                                                corner_radius=0, fg_color="white", text_color="black",
                                                dropdown_fg_color="white", dropdown_text_color="black",
                                                dropdown_hover_color=color_test,
                                                button_color=color_default, button_hover_color=color_test)
                option_menu.set("")
                option_menu.grid(row=6, column=col, padx=5, pady=5, sticky="ew")
                self.row6_menus.append(option_menu)
            elif col == 5:                  # T_HDD entry
                t_hdd_entry = ctk.CTkEntry(self, textvariable=self.thdd_var, width=option_menu_width,
                                           font=("Arial", 12), justify="center",
                                           border_width=2, corner_radius=0,
                                           fg_color="white", border_color="black")
                t_hdd_entry.grid(row=6, column=col, padx=5, pady=5, sticky="ew")
            elif col == 7:                  # T_CDD entry
                t_cdd_entry = ctk.CTkEntry(self, textvariable=self.tcdd_var, width=option_menu_width,
                                           font=("Arial", 12), justify="center",
                                           border_width=2, corner_radius=0,
                                           fg_color="white", border_color="black")
                t_cdd_entry.grid(row=6, column=col, padx=5, pady=5, sticky="ew")

            # functionalities for row 7
            if col == 0:                    # additional info header
                label = ctk.CTkLabel(self, text="Info", font=("Arial", 14, "bold"),
                                     fg_color=color_basic, corner_radius=0,
                                     wraplength=option_menu_width)
                label.grid(row=7, column=col, columnspan=1, padx=5, pady=5, sticky="nsew")
            elif col == 4:                  # outliers header
                label = ctk.CTkLabel(self, text="Outliers", font=("Arial", 14, "bold"),
                                     fg_color=color_basic, corner_radius=0)
                label.grid(row=7, column=col, padx=5, pady=5, sticky="nsew")
            elif col == 6:                  # standard deviation header
                self.normal_distribution_label = ctk.CTkLabel(self, text="\u03C3", font=("Arial", 14, "bold"),
                                                              fg_color=color_basic, corner_radius=0)
                self.normal_distribution_label.grid(row=7, column=col, padx=5, pady=5, sticky="nsew")
                self.normal_distribution_label.grid_remove()  # initially, the label is hidden
            elif col == 2:                  # solar radiation definition
                self.solar_radiation_button = ctk.CTkButton(self, text="Solar", font=("Arial", 14, "bold"),
                                                            border_width=1, border_color="black", corner_radius=5,
                                                            fg_color=color_default,
                                                            command=self.open_solar_radiation_window)
                self.solar_radiation_button.grid(row=7, column=col, columnspan=1, padx=5, pady=5, sticky="ew")
            elif col == 3:                  # demands button
                schedule_button = ctk.CTkButton(self, command=self.open_optimization_schedule_window_1,
                                                font=("Arial", 14, "bold"), text="Demand",
                                                corner_radius=5, fg_color=color_default,
                                                border_color="black", border_width=1)
                schedule_button.grid(row=7, column=col, columnspan=1, padx=5, pady=5, sticky="ew")
            elif col == 5:                  # y/n menu for outliers
                self.outlier_menu = ctk.CTkOptionMenu(self, values=y_n_options,
                                                    dynamic_resizing=False, width=option_menu_width,
                                                    font=("Arial", 12, "bold"), anchor='center',
                                                    corner_radius=0, fg_color="white", text_color="black",
                                                    dropdown_fg_color="white", dropdown_text_color="black",
                                                    dropdown_hover_color=color_test,
                                                    button_color=color_default, button_hover_color=color_test,
                                                    command=self.toggle_normal_distribution_menu)   # functionality
                self.outlier_menu.set("")
                self.outlier_menu.grid(row=7, column=col, padx=5, pady=5, sticky="ew")
            elif col == 7:                  # menu for normal distribution
                self.normal_distribution_menu = ctk.CTkOptionMenu(self, values=normal_distribution_options,
                                                                  dynamic_resizing=False, width=option_menu_width,
                                                                  font=("Arial", 12, "bold"), anchor='center',
                                                                  corner_radius=0, fg_color="white", text_color="black",
                                                                  dropdown_fg_color="white",
                                                                  dropdown_text_color="black",
                                                                  dropdown_hover_color=color_test,
                                                                  button_color=color_default,
                                                                  button_hover_color=color_test)
                self.normal_distribution_menu.set("")
                self.normal_distribution_menu.grid(row=7, column=col, padx=5, pady=5, sticky="ew")
                self.normal_distribution_menu.grid_remove()  # initially, the menu is hidden
            elif col == 9:                 # reset button
                reset_button = ctk.CTkButton(self, text="Reset", font=("Arial", 14, "bold"),
                                             fg_color=color_basic_2, corner_radius=5,
                                             border_color="black", border_width=1,
                                             command=reset_inputs)                                  # functionality
                reset_button.grid(row=7, column=col, padx=5, pady=5, sticky="ew")

    def toggle_normal_distribution_menu(self, selected_value):
        """function to enable/disable the normal distribution menu based on outlier selection"""
        if selected_value == 'y':                           # if outliers: 'y' option
            self.normal_distribution_menu.grid()            # show the menu
            self.normal_distribution_label.grid()           # show the label
        else:
            self.normal_distribution_menu.grid_remove()     # hide the menu
            self.normal_distribution_label.grid_remove()    # hide the label
            self.normal_distribution_menu.set("")           # reset to the default value

    def open_schedule_window(self):
        '''function to generate a new pop-up window for operational schedule definition'''
        x_size = 350
        y_size = 400
        schedule_window = ctk.CTkToplevel(self)
        schedule_window.title("Operational Schedule")
        schedule_window.geometry(f"{x_size}x{y_size}")
        screen_width = self.winfo_screenwidth()  # function: screen width
        screen_height = self.winfo_screenheight()  # function: screen height
        x = (screen_width // 2) - (x_size // 2)  # centering x-axis calculations
        y = (screen_height // 2) - (y_size // 2)  # centering y-axis calculations
        schedule_window.geometry(f"{x_size}x{y_size}+{x}+{y}")  # final positioning

        schedule_window.grab_set()                      # the Toplevel window
        schedule_window.attributes("-topmost", True)    # keep the Toplevel window on top
        schedule_window.grid_columnconfigure(1, weight=1, uniform="equal")
        schedule_window.grid_columnconfigure(2, weight=1, uniform="equal")

        # configure rows
        for row in range(10):
            schedule_window.grid_rowconfigure(row, weight=1, uniform="equal")

        self.schedule_data = []  # list to store schedule information
        headers = ["Day", "From", "Till"]  # headers for schedule window

        for col, header in enumerate(headers):  # add headers in row 0
            label = ctk.CTkLabel(schedule_window, text=header, font=("Arial", 14, "bold"))
            label.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")

        # upper separator line (solid black line) after row 0
        separator_up = ctk.CTkFrame(schedule_window, height=2, fg_color="black")
        separator_up.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(5, 5), pady=(0, 30))

        for row, day in enumerate(days_of_week, start=1):  # data rows for schedule definition
            check_var = ctk.BooleanVar()  # checkboxes
            checkbox = ctk.CTkCheckBox(schedule_window, width=10, height=10, corner_radius=1, border_width=2,
                                       fg_color=color_default, border_color='black', hover_color=color_test,
                                       variable=check_var, text=day, font=("Arial", 12))
            checkbox.grid(row=row, column=0, padx=20, pady=(10,0), sticky="nsew")

            from_var = ctk.StringVar()  # "From" textbox
            from_entry = ctk.CTkEntry(schedule_window, textvariable=from_var, font=("Arial", 12), justify="center",
                                      width=75, height=5, border_width=1, border_color='black', corner_radius=1)
            from_entry.grid(row=row, column=1, padx=5, pady=(10,0), sticky="ns")

            till_var = ctk.StringVar()  # "Till" textbox
            till_entry = ctk.CTkEntry(schedule_window, textvariable=till_var, font=("Arial", 12), justify="center",
                                      width=75, height=5, border_width=1, border_color='black', corner_radius=1)
            till_entry.grid(row=row, column=2, padx=5, pady=(10,0), sticky="ns")

            self.schedule_data.append((check_var, from_var, till_var))  # store the provided references

        # add accept button to store inputs and close the window
        accept_button = ctk.CTkButton(schedule_window, text="Accept", font=("Arial", 12, "bold"),
                                      fg_color=color_run, border_color="black", border_width=1,
                                      command=lambda: self.store_schedule_data(schedule_window))
        accept_button.grid(row=9, column=2, columnspan=1, padx=(5, 5), pady=5, sticky="nsew")

        # text comment below the accept button
        comment_label = ctk.CTkLabel(
            schedule_window,
            text="Comment: use 24 hour timing!",
            font=("Arial", 10, "italic"),
            anchor="center",
            justify="center"
        )
        comment_label.grid(row=8, column=1, columnspan=2, padx=5, pady=0, sticky="new")

    def open_time_index_window(self):
        '''function to generate a pop-up window for time index generation'''
        x_size = 430
        y_size = 280
        self.time_index_window = ctk.CTkToplevel(self)
        self.time_index_window.title("Time Index Generator")
        self.time_index_window.geometry(f"{x_size}x{y_size}")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (x_size // 2)
        y = (screen_height // 2) - (y_size // 2)
        self.time_index_window.geometry(f"{x_size}x{y_size}+{x}+{y}")

        self.time_index_window.grab_set()
        self.time_index_window.attributes("-topmost", True)
        # variables
        self.time_index_path = tk.StringVar()
        self.start_date = tk.StringVar()
        self.end_date = tk.StringVar()
        self.date_format = tk.StringVar()
        self.frequency_minutes = tk.StringVar()

        # directory path
        label_dir = ctk.CTkLabel(self.time_index_window, text="Path", font=("Arial", 12, "bold"))
        label_dir.grid(row=0, column=0, padx=(10,5), pady=(10,5), sticky="w")
        entry_dir = ctk.CTkEntry(self.time_index_window, textvariable=self.time_index_path, font=("Arial", 12))
        entry_dir.grid(row=0, column=1, padx=5, pady=(10,5), sticky="ew")

        browse_button = ctk.CTkButton(
            self.time_index_window,
            text="Browse",
            font=("Arial", 10),
            fg_color=color_default,
            border_color="black", border_width=1,
            command=self.browse_directory_for_time_index
        )
        browse_button.grid(row=0, column=2, padx=5, pady=(10,5), sticky="ew")

        # other inputs
        entries = [
            ("Starting Date *", self.start_date),
            ("Ending Date *", self.end_date),
            ("Date Format **", self.date_format),
            ("Frequency (in min)", self.frequency_minutes)
        ]

        for i, (label_text, variable) in enumerate(entries, start=1):
            label = ctk.CTkLabel(self.time_index_window, text=label_text, font=("Arial", 12, "bold"))
            label.grid(row=i, column=0, padx=(10,5), pady=5, sticky="w")

            entry = ctk.CTkEntry(self.time_index_window, textvariable=variable, font=("Arial", 12))
            entry.grid(row=i, column=1, columnspan=1, padx=5, pady=5, sticky="ew")

            if i == 1:  # for starting date
                date_text = ctk.CTkLabel(self.time_index_window, text="example: 2024-01-01 00:00", font=("Arial", 8),
                                         anchor="w")
                date_text.grid(row=i, column=2, padx=5, pady=5, sticky="w")
            if i == 2:  # for ending date
                date_text = ctk.CTkLabel(self.time_index_window, text="example: 2024-12-31 23:59", font=("Arial", 8),
                                         anchor="w")
                date_text.grid(row=i, column=2, padx=5, pady=5, sticky="w")
            if i == 3:  # for format
                date_text = ctk.CTkLabel(self.time_index_window, text="use Python formating!", font=("Arial", 8),
                                         anchor="w")
                date_text.grid(row=i, column=2, padx=5, pady=5, sticky="w")

        # run button
        run_button = ctk.CTkButton(
            self.time_index_window,
            text="Run",
            font=("Arial", 12, "bold"),
            fg_color=color_run,
            border_color="black", border_width=1,
            command=self.run_time_index_generator
        )
        run_button.grid(row=5, column=2, padx=5, pady=5)

        # text comment below the accept button
        comment_label = ctk.CTkLabel(
            self.time_index_window,
            text="*  use 'year-month-day hour:minute' format!\n"
                 "**  '%d.%m.%Y %H:%M' format is recommended for CEAM application",
            font=("Arial", 12, "italic"),
            anchor="center",
            justify="center"
        )
        comment_label.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

    def browse_directory_for_time_index(self):
        '''browse for a directory and set it in the time index path entry'''
        self.time_index_window.attributes("-topmost", False)
        self.directory_time_index = filedialog.askdirectory()
        self.time_index_window.attributes("-topmost", True)
        if self.directory_time_index:
            self.time_index_path.set(self.directory_time_index)

    def run_time_index_generator(self):
        '''run the time index generation using user inputs'''
        # inputs
        directory = self.time_index_path.get().strip()
        if not directory:
            directory = os.path.join(os.path.expanduser("~"), "Desktop")

        start_time = self.start_date.get().strip() or "2025-01-01 00:00"
        end_time = self.end_date.get().strip() or "2025-04-30 23:59"
        timestamp_format = self.date_format.get().strip() or "%d.%m.%Y %H:%M"
        frequency_in_minutes = self.frequency_minutes.get().strip() or "60"

        try:
            frequency_in_minutes = int(frequency_in_minutes)
        except ValueError:
            messagebox.showerror("Error", "Frequency must be an integer (minutes).", parent=self.time_index_window)
            return

        if not os.path.isdir(directory):
            messagebox.showerror("Error", "Provided directory path does not exist.", parent=self.time_index_window)
            return

        try:
            time_index = pd.date_range(start=start_time, end=end_time, freq=str(frequency_in_minutes) + 'T')
            df = pd.DataFrame(time_index, columns=['Time'])
            df['Time'] = df['Time'].dt.strftime(timestamp_format)
            full_path = os.path.join(directory, f"time_index_{frequency_in_minutes}.csv")
            df.to_csv(full_path, index=False)
            messagebox.showinfo("Success", f"CSV file created:\n{full_path}", parent=self.time_index_window)
        except Exception as e:
            messagebox.showerror("Error", "Problem description", parent=self.time_index_window)

    def store_schedule_data(self, schedule_window):
        '''function to store the provided inputs from the operational schedule window'''
        self.selected_schedule_data.clear()  # clear the initial list

        for idx, (check_var, from_var, till_var) in enumerate(self.schedule_data):
            selected = check_var.get()
            day = days_of_week[idx]         # get the corresponding day

            # validate from_time
            try:
                from_time = int(from_var.get().strip())
                if from_time < 0 or from_time > 23:
                    from_time = None
            except ValueError:
                from_time = None

            # validate till_time
            try:
                till_time = int(till_var.get().strip())
                if till_time < 0 or till_time > 23 or (from_time is not None and till_time <= from_time):
                    till_time = None
            except ValueError:
                till_time = None

            if from_time is None:
                from_var.set("")  # clear invalid input for from_time
            if till_time is None:
                till_var.set("")  # clear invalid input for till_time

            # append data only for selected days with valid from_time and till_time
            if selected and from_time is not None and till_time is not None:
                schedule_entry = {
                    "Operation day": day,
                    "selected": selected,
                    "from": from_time,
                    "till": till_time
                }
                self.selected_schedule_data.append(schedule_entry)

        print("Selected Schedule Data:", self.selected_schedule_data)

        schedule_window.destroy()  # close the schedule window

    def open_optimization_schedule_window(self):
        '''function to generate a new pop-up window for optimization schedule definition'''
        x_size = 330
        y_size = 320
        optimization_schedule_window = ctk.CTkToplevel(self)
        optimization_schedule_window.title("Optimization Schedule")
        optimization_schedule_window.geometry(f"{x_size}x{y_size}")
        screen_width = self.winfo_screenwidth()  # function: screen width
        screen_height = self.winfo_screenheight()  # function: screen height
        x = (screen_width // 2) - (x_size // 2)  # centering x-axis calculations
        y = (screen_height // 2) - (y_size // 2)  # centering y-axis calculations
        optimization_schedule_window.geometry(f"{x_size}x{y_size}+{x}+{y}")  # final positioning

        optimization_schedule_window.grab_set()
        optimization_schedule_window.attributes("-topmost", True)

        self.optimization_schedule_data = []  # list to store optimization schedule information
        headers = ["Schedule", "Details"]  # headers for optimization schedule window

        # headers in row 0
        for col, header in enumerate(headers):
            label = ctk.CTkLabel(optimization_schedule_window, text=header, font=("Arial", 14, "bold"))
            label.grid(row=0, column=col, padx=25, pady=5, sticky="nsew")

        # an upper separator line after row 0
        separator_up = ctk.CTkFrame(optimization_schedule_window, height=2, fg_color="black")
        separator_up.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(15, 0), pady=(0, 40))

        # description of the predefined optimization schedules
        optimization_schedule_descriptions = [
            "Short offsets are applied for all days  *",
            "Short offsets are applied for weekdays only  *",
            "Short offsets are applied for weekends only  *",
            "Short offsets are applied besides the working hours  **",
            "Short offsets are applied besides the working hours considering national holidays  **",
        ]

        # loop through each schedule and description to create checkboxes and descriptions
        for row, (optimization_schedule, description) in enumerate(
                zip(optimization_schedule_list, optimization_schedule_descriptions), start=1):
            checkbox_frame = ctk.CTkFrame(optimization_schedule_window, fg_color="transparent")
            checkbox_frame.grid(row=row, column=0, padx=20, pady=5, sticky="nsew")

            # create the checkbox
            check_var = ctk.BooleanVar()
            checkbox = ctk.CTkCheckBox(checkbox_frame, width=10, height=10, corner_radius=1, border_width=2,
                                       fg_color=color_default, border_color='black', hover_color=color_test,
                                       variable=check_var, text=optimization_schedule, font=("Arial", 12))
            checkbox.pack(expand=True)

            description_label = ctk.CTkLabel(
                optimization_schedule_window,
                text=description,
                font=("Arial", 12),
                anchor="w",
                justify="left",
                wraplength=200
            )
            description_label.grid(row=row, column=1, padx=5, pady=5, sticky="nsw")

            self.optimization_schedule_data.append(check_var)   # store the checkbox variable reference

        # accept button
        accept_button = ctk.CTkButton(
            optimization_schedule_window,
            text="Accept",
            font=("Arial", 12, "bold"),
            fg_color=color_run,
            border_color="black",
            border_width=1,
            command=lambda: self.store_optimization_schedule_data(optimization_schedule_window)
        )
        accept_button.grid(row=8, column=1, columnspan=1, padx=(25, 25), pady=(10, 5), sticky="nsew")

        # text comment below the accept button
        comment_label = ctk.CTkLabel(
            optimization_schedule_window,
            text="* whole days (24h) are considered for s1, s2, and s3!\n"
                 "** working time defined in the Timetable tab!",
            font=("Arial", 10, "italic"),
            anchor="center",
            justify="center"
        )
        comment_label.grid(row=9, column=0, columnspan=2, padx=5, pady=0, sticky="ew")

    def open_optimizers_window(self):
        '''function to generate a new pop-up window for optimization schedule definition'''
        x_size = 350
        y_size = 280
        optimizers_window = ctk.CTkToplevel(self)
        optimizers_window.title("Optimizer(s) selection")
        optimizers_window.geometry(f"{x_size}x{y_size}")
        screen_width = self.winfo_screenwidth()  # function: screen width
        screen_height = self.winfo_screenheight()  # function: screen height
        x = (screen_width // 2) - (x_size // 2)  # centering x-axis calculations
        y = (screen_height // 2) - (y_size // 2)  # centering y-axis calculations
        optimizers_window.geometry(f"{x_size}x{y_size}+{x}+{y}")  # final positioning

        optimizers_window.grab_set()
        optimizers_window.attributes("-topmost", True)

        self.optimizers_data = []  # list to store optimization schedule information
        headers = ["Model", "Description"]  # headers for optimization schedule window

        for col, header in enumerate(headers):
            label = ctk.CTkLabel(optimizers_window, text=header, font=("Arial", 14, "bold"))
            label.grid(row=0, column=col, padx=25, pady=5, sticky="nsew")

        separator_up = ctk.CTkFrame(optimizers_window, height=2, fg_color="black")
        separator_up.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(15, 0), pady=(0, 40))

        optimization_schedule_descriptions = [
            "Decision Tree model *",
            "Neural Networks model **",
            "Long Short-Term Memory model ***",
            "Gated Recurrent Units model ***",
        ]

        for row, (optimizers_models, description) in enumerate(
                zip(optimizers_list, optimization_schedule_descriptions), start=1):
            checkbox_frame = ctk.CTkFrame(optimizers_window, fg_color="transparent")
            checkbox_frame.grid(row=row, column=0, padx=30, pady=5, sticky="nsw")

            check_var = ctk.BooleanVar()
            checkbox = ctk.CTkCheckBox(checkbox_frame, width=10, height=10, corner_radius=1, border_width=2,
                                       fg_color=color_default, border_color='black', hover_color=color_test,
                                       variable=check_var, text=optimizers_models, font=("Arial", 12))
            checkbox.pack(expand=True)

            description_label = ctk.CTkLabel(
                optimizers_window,
                text=description,
                font=("Arial", 12),
                anchor="w",
                justify="left",
                wraplength=250
            )
            description_label.grid(row=row, column=1, padx=5, pady=5, sticky="nsw")

            self.optimizers_data.append(check_var)      # store the checkbox variable reference

        accept_button = ctk.CTkButton(
            optimizers_window,
            text="Accept",
            font=("Arial", 12, "bold"),
            fg_color=color_run,
            border_color="black",
            border_width=1,
            command=lambda: self.store_optimizers_data(optimizers_window)
        )
        accept_button.grid(row=8, column=1, columnspan=1, padx=(25, 25), pady=(10, 5), sticky="nsew")

        comment_label = ctk.CTkLabel(
            optimizers_window,
            text="* for non-manageable environments\n"
                 "** for general applications\n"
                 "*** for complex applications",
            font=("Arial", 10, "italic"),
            anchor="center",
            justify="center"
        )
        comment_label.grid(row=9, column=1, columnspan=1, padx=5, pady=0, sticky="ew")

    def store_optimization_schedule_data(self, optimization_schedule_window):
        '''function to store the provided inputs from optimization schedule window'''
        self.selected_optimization_schedules.clear()       # clear the initial list

        for idx, check_var in enumerate(self.optimization_schedule_data):
            selected = check_var.get()                                          # get value from the checkbox
            self.selected_optimization_schedules.append((idx + 1, selected))    # store the data

        optimization_schedule_window.destroy()      # close the schedule window

    def store_optimizers_data(self, optimizers_window):
        '''function to store the provided inputs from optimization schedule window'''
        self.selected_optimizers.clear()       # clear the initial list

        for idx, check_var in enumerate(self.optimizers_data):
            selected = check_var.get()                                          # get value from the checkbox
            self.selected_optimizers.append((idx, selected))    # store the data

        optimizers_window.destroy()      # close the schedule window

    def open_solar_radiation_window(self):
        '''function to generate a new pop-up window for solar radiation definition'''
        x_size = 150
        y_size = 100
        self.solar_window = ctk.CTkToplevel(self)
        self.solar_window.title("Solar Radiation")
        self.solar_window.geometry(f"{x_size}x{y_size}")
        screen_width = self.winfo_screenwidth()  # function: screen width
        screen_height = self.winfo_screenheight()  # function: screen height
        x = (screen_width // 2) - (x_size // 2)  # centering x-axis calculations
        y = (screen_height // 2) - (y_size // 2)  # centering y-axis calculations
        self.solar_window.geometry(f"{x_size}x{y_size}+{x}+{y}")  # final positioning

        self.solar_window.grab_set()
        self.solar_window.attributes("-topmost", True)

        self.solar_radiation_menu = ctk.CTkOptionMenu(self.solar_window,
                                                      values=y_n_options,
                                                      dynamic_resizing=False,
                                                      width=120,
                                                      font=("Arial", 12, "bold"),
                                                      corner_radius=0, fg_color="white", text_color="black",
                                                      dropdown_fg_color="white", dropdown_text_color="black",
                                                      dropdown_hover_color=color_test,
                                                      button_color=color_default, button_hover_color=color_test,
                                                      anchor='center')
        self.solar_radiation_menu.set("")       # default value
        self.solar_radiation_menu.pack(pady=10)

        close_button = ctk.CTkButton(self.solar_window,
                                     text="Accept", font=("Arial", 12, "bold"),
                                     fg_color=color_run, border_color="black", border_width=1,
                                     command = self.solar_window.destroy
                                     )
        close_button.pack(padx=25, pady=10)

    def open_cc_reeval_window(self):
        '''function to generate a new pop-up window for CC re-evaluation definition'''
        x_size = 200
        y_size = 150
        self.cc_reeval_window = ctk.CTkToplevel(self)
        self.cc_reeval_window.title("CC re-assessment")
        self.cc_reeval_window.geometry(f"{x_size}x{y_size}")
        screen_width = self.winfo_screenwidth()  # function: screen width
        screen_height = self.winfo_screenheight()  # function: screen height
        x = (screen_width // 2) - (x_size // 2)  # centering x-axis calculations
        y = (screen_height // 2) - (y_size // 2)  # centering y-axis calculations
        self.cc_reeval_window.geometry(f"{x_size}x{y_size}+{x}+{y}")  # final positioning

        self.cc_reeval_window.grab_set()
        self.cc_reeval_window.attributes("-topmost", True)

        self.cc_reeval_menu = ctk.CTkOptionMenu(self.cc_reeval_window,
                                                values=y_n_options,
                                                dynamic_resizing=False,
                                                width=120,
                                                font=("Arial", 12, "bold"),
                                                corner_radius=0, fg_color="white", text_color="black",
                                                dropdown_fg_color="white", dropdown_text_color="black",
                                                dropdown_hover_color=color_test,
                                                button_color=color_default, button_hover_color=color_test,
                                                anchor='center')
        self.cc_reeval_menu.set("")             # default value
        self.cc_reeval_menu.pack(pady=10)

        close_button = ctk.CTkButton(self.cc_reeval_window,
                                     text="Accept", font=("Arial", 12, "bold"),
                                     fg_color=color_run, border_color="black", border_width=1,
                                     command=self.cc_reeval_window.destroy
                                     )
        close_button.pack(padx=25, pady=10)

        warning_label = ctk.CTkLabel(
            self.cc_reeval_window,
            text=f"ATTENTION: \n  Initial CC assessment (Module 4) \n  is required for CC-reevaluation !",
            font=("Arial", 10),
            text_color="black",
            width=175,
            justify="left"
        )
        warning_label.pack(padx=10, pady=(0, 10))

    def open_optimization_schedule_window_1(self):
        '''function to generate a new pop-up window for demands selection'''
        x_size = 300
        y_size = 320
        optimization_schedule_window_1 = ctk.CTkToplevel(self)
        optimization_schedule_window_1.title("Demand data")
        optimization_schedule_window_1.geometry(f"{x_size}x{y_size}")
        screen_width = self.winfo_screenwidth()  # function: screen width
        screen_height = self.winfo_screenheight()  # function: screen height
        x = (screen_width // 2) - (x_size // 2)  # centering x-axis calculations
        y = (screen_height // 2) - (y_size // 2)  # centering y-axis calculations
        optimization_schedule_window_1.geometry(f"{x_size}x{y_size}+{x}+{y}")  # final positioning

        optimization_schedule_window_1.grab_set()
        optimization_schedule_window_1.attributes("-topmost", True)

        self.optimization_schedule_data_1 = []  # list to store optimization schedule information
        headers = ["Demand", "Description"]  # headers for optimization schedule window

        for col, header in enumerate(headers):
            label = ctk.CTkLabel(optimization_schedule_window_1, text=header, font=("Arial", 14, "bold"))
            label.grid(row=0, column=col, padx=25, pady=5, sticky="nsew")

        separator_up = ctk.CTkFrame(optimization_schedule_window_1, height=2, fg_color="black")
        separator_up.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(10, 5), pady=(40, 0))

        optimization_schedule_descriptions = [
            "Heating consumption",
            "Cooling consumption",
            "Ventilation consumption",
            "(De)humidification consumption",
            "Total consumption"
        ]

        for row, (optimization_schedule, description) in enumerate(
                zip(demand_list, optimization_schedule_descriptions), start=1):
            checkbox_frame = ctk.CTkFrame(optimization_schedule_window_1, fg_color="transparent")
            checkbox_frame.grid(row=row, column=0, padx=20, pady=5, sticky="nsew")

            check_var = ctk.BooleanVar()
            checkbox = ctk.CTkCheckBox(checkbox_frame, width=10, height=10, corner_radius=1, border_width=2,
                                       fg_color=color_default, border_color='black', hover_color=color_test,
                                       variable=check_var, text=optimization_schedule, font=("Arial", 12))
            checkbox.pack(expand=True)

            description_label = ctk.CTkLabel(
                optimization_schedule_window_1,
                text=description,
                font=("Arial", 12),
                anchor="w",
                justify="left",
                wraplength=200
            )
            description_label.grid(row=row, column=1, padx=5, pady=5, sticky="nsw")

            self.optimization_schedule_data_1.append(check_var)

        accept_button = ctk.CTkButton(
            optimization_schedule_window_1,
            text="Accept",
            font=("Arial", 12, "bold"),
            fg_color=color_run,
            border_color="black",
            border_width=1,
            command=lambda: self.store_optimization_schedule_data_1(optimization_schedule_window_1)
        )
        accept_button.grid(row=8, column=1, columnspan=1, padx=(25, 25), pady=(10, 5), sticky="nsew")

        comment_label = ctk.CTkLabel(
            optimization_schedule_window_1,
            text="Select demands valid for the given ED.csv input file!",
            font=("Arial", 10, "italic"),
            anchor="center",
            justify="center"
        )
        comment_label.grid(row=9, column=0, columnspan=2, padx=5, pady=0, sticky="ew")

    def store_optimization_schedule_data_1(self, optimization_schedule_window_1):
        '''function to store the provided inputs from demands selection window'''
        self.selected_optimization_schedules_1.clear()       # clear the initial list

        for idx, check_var in enumerate(self.optimization_schedule_data_1):
            selected = check_var.get()                                          # get value from the checkbox
            self.selected_optimization_schedules_1.append((idx + 1, selected))    # store the data

        optimization_schedule_window_1.destroy()      # close the schedule window

    def _run_analysis(self):
        '''pop-up "run" window'''
        self.computation_done = False   # initialize computation_done flag at the start of the method

        test_window = tk.Toplevel(self)  # function
        test_window.title("Analysis")  # window title
        test_window.geometry("300x100")  # size
        screen_width = self.winfo_screenwidth()  # function: screen width
        screen_height = self.winfo_screenheight()  # function: screen height
        x = (screen_width // 2) - (300 // 2)  # centering x-axis calculations
        y = (screen_height // 2) - (100 // 2)  # centering y-axis calculations
        test_window.geometry(f"300x100+{x}+{y}")  # final positioning

        label = tk.Label(test_window, text='Analysis in progress...', font=("Arial", 14))  # text and font
        label.pack(expand=True)  # positioning inside the window

        start_time = time.time()    # record start time

        def update_comment_textbox():
            while not self.computation_done:  # continue updating until computation is done
                current_time = time.time()
                running_time = int(current_time - start_time)
                self.comment_textbox.configure(state="normal")  # enable editing to update text
                self.comment_textbox.delete("0.0", "end")  # clear current text
                self.comment_textbox.insert("0.0",
                                            f"Status on: {time.strftime('%H:%M:%S')} - Computing... already for {running_time} s.")
                self.comment_textbox.configure(state="disabled")  # disable editing again
                time.sleep(1)  # update every second

        # thread to update the comment textbox
        update_thread = threading.Thread(target=update_comment_textbox)
        update_thread.daemon = True  # ensure the thread closes when the main program exits
        update_thread.start()

        def run_script():
            '''create the txt input file to run the scripts'''
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            file_path = os.path.join(desktop_path, "var.txt")
            headers_row4 = {1: "ICD", 2: "ECD", 3: "ED"}
            try:
                with open(file_path, 'w') as file:
                    file.write(f"Working directory: {self.directory}\n\n")

                    for i in range(6):
                        status = "on" if self.clicked_buttons[i] else "off"
                        file.write(f"Module {i}: {status}\n")
                    file.write("\n")

                    for i, var in enumerate(self.freq_vars):
                        header = headers_row4.get(i + 1)
                        value = var.get() or "Not selected"
                        file.write(f"{header}_freq: {value}\n")
                    file.write("\n")

                    for i, menu in enumerate(self.row6_menus):
                        header = headers_row4.get(i + 1)
                        value = menu.get() or "Not selected"
                        file.write(f"{header}_format: {value}\n")
                    file.write("\n")

                    if hasattr(self, 'solar_radiation_menu'):
                        yn_value = self.solar_radiation_menu.get() or "Not selected"
                        file.write(f"Solar radiation: {yn_value}\n")
                    file.write("\n")

                    if hasattr(self, 'cc_reeval_menu'):
                        yn_value = self.cc_reeval_menu.get() or "Not selected"
                        file.write(f"CC re-eval: {yn_value}\n")
                    file.write("\n")

                    country = self.country_var.get()
                    file.write(
                        f"Country: {country.capitalize() if isinstance(country, str) and country.isalpha() else 'Not provided'}\n")
                    city = self.city_var.get()
                    file.write(
                        f"City: {city.capitalize() if isinstance(city, str) and city.isalpha() else 'Not provided'}\n")
                    file.write("\n")

                    if hasattr(self, 'unit_menu'):
                        units_value = self.unit_menu.get() or "Not selected"
                        file.write(f"Units: {units_value}\n")
                    file.write("\n")

                    def get_number(value):
                        try:
                            value = str(value).replace(',', '.')
                            return float(value)
                        except (ValueError, TypeError):
                            return 'N/A'

                    t_hdd = get_number(self.thdd_var.get())
                    t_cdd = get_number(self.tcdd_var.get())

                    if isinstance(t_hdd, float) and isinstance(t_cdd, float) and t_hdd > t_cdd:
                        t_hdd = t_cdd = 'N/A'

                    file.write(f"T_HDD: {t_hdd}\n")
                    file.write(f"T_CDD: {t_cdd}\n")
                    file.write("\n")

                    if hasattr(self, 'outlier_menu'):
                        outlier_value = self.outlier_menu.get() or "Not selected"
                        file.write(f"Outlier: {outlier_value}\n")
                        normal_distribution_value = self.normal_distribution_menu.get() or "Not selected"
                        file.write(f"Normal distribution: {normal_distribution_value}\n")
                    file.write("\n")

                    if not self.selected_schedule_data or all(
                            not schedule.get('selected', False) for schedule in self.selected_schedule_data):
                        file.write("Operation Schedule: N/A\n")
                    else:
                        for schedule in self.selected_schedule_data:
                            file.write(
                                f"Operation day: {schedule['Operation day']}, from: {schedule['from']}, till: {schedule['till']}\n"
                            )
                    file.write("\n")

                    if not self.selected_optimization_schedules or all(
                            not selected for _, selected in self.selected_optimization_schedules):
                        file.write("Optimization Schedule: N/A\n")
                    else:
                        selected_indices = [f"s{idx}" for idx, selected in self.selected_optimization_schedules if
                                            selected]
                        file.write(
                            f"Optimization Schedule: {', '.join(selected_indices)}\n" if selected_indices else "Optimization Schedule: Not Selected\n")
                    file.write("\n")

                    optimizer_mapping = {0: "DT", 1: "NN", 2: "LSTM", 3: "GRU"}
                    if not self.selected_optimizers or all(
                            not selected for _, selected in self.selected_optimizers):
                        file.write("Optimizers: N/A\n")
                    else:
                        selected_names = [optimizer_mapping[idx] for idx, selected in
                                          self.selected_optimizers if selected]
                        file.write(
                            f"Optimizers: {', '.join(selected_names)}\n" if selected_names else "Optimizers: Not Selected\n")
                    file.write("\n")

                    if not self.selected_optimization_schedules_1 or all(
                            not selected for _, selected in self.selected_optimization_schedules_1):
                        file.write("Examined demands: N/A\n")
                    else:
                        selected_indices_1 = [f"D{idx}" for idx, selected in self.selected_optimization_schedules_1 if
                                              selected]
                        file.write(
                            f"Examined demands: {', '.join(selected_indices_1)}\n" if selected_indices_1 else "Examined demands: Not Selected\n")
                    file.write("\n")

                print(f"Directory and module statuses saved to {file_path}")
            except Exception as e:
                print(f"Error saving directory or module statuses: {e}")

            script_path = os.path.join(sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(__file__),
                                       "_00_CEAM_run.py")
            try:
                _00_CEAM_run.run_main_logic()
                print("_00_CEAM_run executed successfully.")
            except Exception as e:
                print(f"Error running _00_CEAM_run: {e}")
            finally:
                end_time = time.time()
                running_time = int(end_time - start_time)
                completion_time = time.strftime('%H:%M:%S')

                self.comment_textbox.configure(state="normal")
                self.comment_textbox.delete("0.0", "end")
                self.comment_textbox.insert("0.0",
                                            f"Completed at {completion_time}! Total computing time: {running_time} s.")
                self.comment_textbox.configure(state="disabled")

                if os.path.isdir(self.graphs_folder):
                    self.graphs_button.configure(fg_color=color_run, state="normal")
                if os.path.isdir(self.reports_folder):
                    self.reports_button.configure(fg_color=color_run, state="normal")
                if os.path.isdir(self.directory):
                    self.dir_button.configure(fg_color=color_run, state="normal")

                label.config(text="Completed!", font=("Arial", 14))
                label.pack(expand=True)

                self.computation_done = True        # set the computation_done flag to True

                ok_button = ctk.CTkButton(
                    test_window,
                    text="OK",
                    command=test_window.destroy,
                    font=("Arial", 12, 'bold'),
                    fg_color=color_run,
                    border_color="black",
                    border_width=1
                )
                ok_button.pack(pady=10)

                # play a sound when computation is complete
                try:
                    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                except Exception as e:
                    print(f"Error playing sound: {e}")

        script_thread = threading.Thread(target=run_script)
        script_thread.start()


'''APPLICATION'''
if __name__ == "__main__":      # run the tool/GUI
    app = App()
    app.mainloop()