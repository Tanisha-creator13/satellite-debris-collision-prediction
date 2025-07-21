# src/data_processor.py

import pandas as pd
from sgp4.api import WGS72, Satrec
import os
from datetime import datetime 

# --- Configuration for directories ---
script_dir = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(script_dir, '..', 'data', 'raw')

PROCESSED_DATA_DIR = os.path.join(script_dir, '..', 'data', 'processed')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# --- End Configuration ---

def parse_tles_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Parses a file containing TLEs into a Pandas DataFrame.

    Args:
        file_path (str): The path to the TLE file.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a TLE,
                      with columns for NORAD_CAT_ID, Epoch, Line1, Line2,
                      and various derived orbital parameters.
    """
    tle_data_raw = [] # To store [line1, line2] pairs
    current_tle_lines = []

    print(f"Reading TLEs from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Filter out empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]

        for i in range(len(lines)):
            line = lines[i]
            if line.startswith('1 '):
                # Found the start of a TLE (Line 1)
                current_tle_lines = [line] # Start new TLE with Line 1
            elif line.startswith('2 '):
                # Found Line 2, add it to current TLE
                if len(current_tle_lines) == 1: # Ensure Line 1 was just added
                    current_tle_lines.append(line)
                    tle_data_raw.append(current_tle_lines) # Add complete TLE pair
                    current_tle_lines = [] # Reset for next TLE
                else:
                    pass
            else:
                pass 

        print(f"Found {len(tle_data_raw)} complete TLE pairs.")

        parsed_tles = []
        for line1, line2 in tle_data_raw:
            try:
                # Use WGS72 as it's the standard for NORAD TLEs
                satellite = Satrec.twoline2rv(line1, line2, WGS72)
                epoch_datetime = pd.to_datetime(satellite.jdsatepoch, unit='D', origin='julian')

                # Extract data from the Satrec objects
                parsed_tles.append({
                    'NORAD_CAT_ID': satellite.satnum,
                    'OBJECT_DESIGNATOR': satellite.intldesg, # International Designator (e.g., 98067A)
                    'EPOCH': epoch_datetime,
                    'INCLINATION_RAD': satellite.inclo,        # Inclination in radians
                    'RAAN_RAD': satellite.nodeo,               # Right Ascension of the Ascending Node in radians
                    'ECCENTRICITY': satellite.ecco,            # Eccentricity
                    'ARG_OF_PERIGEE_RAD': satellite.argpo,     # Argument of Perigee in radians
                    'MEAN_ANOMALY_RAD': satellite.mo,          # Mean Anomaly in radians
                    'MEAN_MOTION_REV_PER_DAY': satellite.no_kozai, # Mean Motion in revolutions per day
                    'REV_NUM_AT_EPOCH': satellite.revnum,      # Revolution number at epoch
                    'BSTAR': satellite.bstar,                  # BSTAR drag term
                    # 'EPHEMERIS_TYPE': satellite.ephtype,
                    # 'ELEMENT_SET_NUM': satellite.elnum,
                    # 'CHECKSUM_LINE1': satellite.checksum_line1,
                    # 'CHECKSUM_LINE2': satellite.checksum_line2,
                    'LINE1': line1,
                    'LINE2': line2
                })
            except Exception as e:
                print(f"Error parsing TLE pair (NORAD_CAT_ID might be unknown for malformed TLE):\nLine 1: {line1}\nLine 2: {line2}\nError: {e}")
                continue # Skip to the next TLE pair

        df = pd.DataFrame(parsed_tles)
        return df

    except FileNotFoundError:
        print(f"Error: TLE file not found at {file_path}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"An unexpected error occurred during file reading or initial processing: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    today_str = datetime.utcnow().strftime('%Y%m%d') 
    tle_filename = f'tles_celestrak_active_{today_str}.txt'

    # Construct the full file path for input TLEs
    tle_file_path = os.path.join(RAW_DATA_DIR, tle_filename)

    # Call the parsing function
    tles_df = parse_tles_to_dataframe(tle_file_path)

    if not tles_df.empty:
        print(f"\nSuccessfully parsed {len(tles_df)} TLEs into a DataFrame.")
        print("\nFirst 5 rows of the DataFrame:")
        print(tles_df.head())
        print(f"\nDataFrame Info:")
        tles_df.info()

        # Save the processed DataFrame to CSV
        output_csv_path = os.path.join(PROCESSED_DATA_DIR, f'processed_tles_{today_str}.csv')
        tles_df.to_csv(output_csv_path, index=False)
        print(f"\nProcessed TLEs saved to: {output_csv_path}")
    else:
        print("No TLEs were parsed into the DataFrame. Please check the input file and parsing logic.")