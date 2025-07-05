import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_and_process_csv(file_path):
    # Read the CSV file, skipping the first row
    data = pd.read_csv(file_path, skiprows=1, header=None)
    # Convert nanoseconds to seconds
    data_seconds = data / 1e9
    return data_seconds

def calculate_event_frequency_per_second(data_seconds):
    # Convert DataFrame to Series for easier processing
    time_series = data_seconds.iloc[:, 0]
    # Round down the event times to the nearest second to group them
    time_series_rounded = np.floor(time_series).astype(int)
    # Count the number of events per second
    event_counts_per_second = time_series_rounded.value_counts().sort_index()
    # Fill in seconds with zero events
    full_range = np.arange(event_counts_per_second.index.min(), event_counts_per_second.index.max() + 1)
    event_counts_per_second = event_counts_per_second.reindex(full_range, fill_value=0)
    return event_counts_per_second

def plot_event_frequency_within_range(event_counts_per_second, start_time, end_time, file_path):
    # Filter the event counts for the specified time range
    filtered_event_counts = event_counts_per_second.loc[start_time:end_time]
    # Create a plot
    fig, ax = plt.subplots()
    ax.plot(filtered_event_counts.index, filtered_event_counts.values)
    ax.set_title(f"Event Frequency per Second (Range: {start_time}-{end_time}) in File: {file_path}")
    ax.set_xlabel('Time (Seconds)')
    ax.set_ylabel('Frequency (Events per Second)')
    # Save the plot as SVG
    svg_filename = file_path.rsplit('.', 1)[0] + f'_frequency_{start_time}_{end_time}.svg'
    plt.savefig(svg_filename, format='svg')
    plt.close()
    return svg_filename

# Usage example
file_path = '1#-465.csv'  # replace with your file path
data_seconds = read_and_process_csv(file_path)
event_counts_per_second = calculate_event_frequency_per_second(data_seconds)

# Define your specific start and end times here
start_time = 0  # replace with your start time
end_time = 900    # replace with your end time

svg_filename_range = plot_event_frequency_within_range(event_counts_per_second, start_time, end_time, file_path)
print(f"SVG file created: {svg_filename_range}")
