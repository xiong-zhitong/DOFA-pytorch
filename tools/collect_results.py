import os
import json
import csv

# Define the path to the logs directory
logs_dir = "logs"
output_csv = "extracted_log_data.csv"

# Prepare a list to store the extracted data
data = []

# Iterate through the logs folder
for root, dirs, files in os.walk(logs_dir):
    for file in files:
        if file == "log.txt":
            # Extract the folder name
            folder_name = os.path.basename(os.path.dirname(root))
            
            # Path to the log.txt file
            log_path = os.path.join(root, file)
            
            # Read the last line of the log.txt file
            with open(log_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    log_data = json.loads(last_line)
                    
                    # Extract the last column item (the last key in the dictionary)
                    last_column_key = list(log_data.keys())[-1]
                    last_column_value = log_data[last_column_key]
                    
                    # Append the extracted data to the list
                    data.append([folder_name, last_column_key, last_column_value])

# Write the collected data to a CSV file
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Exp Name", "Metric", "Value"])  # Header
    writer.writerows(data)

print(f"Data has been successfully written to {output_csv}")

