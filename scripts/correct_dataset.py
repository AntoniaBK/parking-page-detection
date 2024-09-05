import csv

def get_list(path:str) -> list[str]:
     with open(path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]
     
uuids_in_warninglist = get_list("pp_uuids.txt")

# Read the CSV file
input_file = 'mistake_simple_data.csv'
output_file = 'simple_data.csv'

# Open the CSV file for reading and writing
with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames  # Get the CSV header
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the header to the new file
    writer.writeheader()

    # Loop through each row in the CSV file
    for row in reader:
        if row['uuid'] in uuids_in_warninglist:
            row['in_warninglist'] = 'True' 
        writer.writerow(row)  # Write the updated row to the new file

print("CSV file updated successfully!")
