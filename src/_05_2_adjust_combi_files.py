import os
import csv
from datetime import datetime
from _00_CEAM_run import directory, known_format


'''FUNCTIONALITIES'''
def main():
    def convert_to_datetime(date_str, date_format):
        """convert string date to datetime object based on provided format"""
        return datetime.strptime(date_str, date_format)

    def read_csv(filename):
        """read the csv file"""
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [row for row in reader]

    def write_csv(filename, data):
        """write list of dictionaries to csv file"""
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def process_file(filename, date_format, output_format):
        """read, sort by 'Time', and write back to the csv file with specified date formats"""
        file_path = os.path.join(directory, filename)
        data = read_csv(file_path)

        for row in data:    # convert 'Time' to datetime object
            row['Time'] = convert_to_datetime(row['Time'], date_format)

        data_sorted = sorted(data, key=lambda x: x['Time'])             # sort data based on 'Time'

        for row in data_sorted:             # convert 'Time' back to string format
            row['Time'] = row['Time'].strftime(output_format)

        write_csv(file_path, data_sorted)
        print(f"Sorted data has been written to {file_path}.")

    # define file configurations for each case
    file_configs = [
        {"filename": "d_comb.csv", "date_format": "%d.%m.%Y", "output_format": "%d.%m.%Y"},
        {"filename": "h_comb.csv", "date_format": known_format, "output_format": known_format},
        {"filename": "m_comb.csv", "date_format": "%m.%Y", "output_format": "%m.%Y"}
    ]

    # process each file based on configuration
    for config in file_configs:
        process_file(config["filename"], config["date_format"], config["output_format"])


'''APPLICATION'''
if __name__ == "__main__":
    main()