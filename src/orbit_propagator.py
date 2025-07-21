import pandas as pd
from sgp4.api import WGS72, Satrec, jday
import os
from datetime import datetime, timedelta

# --- Configuration for directories ---
script_dir = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(script_dir, '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(script_dir, '..', 'data', 'processed')
PROPAGATED_DATA_DIR = os.path.join(script_dir, '..', 'data', 'propagated')

# Ensure output directory exists
os.makedirs(PROPAGATED_DATA_DIR, exist_ok=True)
# --- End Configuration ---

def propagate_tle(satellite: Satrec, start_epoch: datetime, num_hours: int = 24, time_step_minutes: int = 5):
    """
    Propagates a single TLE for a given duration and time step.

    Args:
        satellite (Satrec): The SGP4 satellite object.
        start_epoch (datetime): The starting datetime for propagation (usually the TLE's epoch).
                                Must be a naive datetime (no timezone info), assumed UTC.
        num_hours (int): The total duration to propagate in hours.
        time_step_minutes (int): The time step for propagation in minutes.

    Returns:
        list of dict: A list of dictionaries, each containing propagated state (pos, vel)
                      and timestamp. Returns empty list if propagation fails consistently.
    """
    propagated_states = []
    
    for i in range(0, num_hours * 60 + 1, time_step_minutes): # +1 to include the end point
        current_time = start_epoch + timedelta(minutes=i)
        
        # Calculate Julian date for propagation using datetime parts
        jd, fr = jday(current_time.year, current_time.month, current_time.day, 
                      current_time.hour, current_time.minute, 
                      current_time.second + current_time.microsecond / 1_000_000.0)

        # Propagate
        e, r, v = satellite.sgp4(jd, fr) # r: position, v: velocity

        if e == 0: # If propagation was successful
            propagated_states.append({
                'NORAD_CAT_ID': satellite.satnum,
                'TIMESTAMP_UTC': current_time,
                'X_KM': r[0],
                'Y_KM': r[1],
                'Z_KM': r[2],
                'VX_KMS': v[0],
                'VY_KMS': v[1],
                'VZ_KMS': v[2]
            })
        else:
            pass 

    return propagated_states

if __name__ == "__main__":
    # --- Determine the filename of the processed TLEs ---
    today_str = datetime.utcnow().strftime('%Y%m%d')
    processed_tle_filename = f'processed_tles_{today_str}.csv'
    processed_tle_file_path = os.path.join(PROCESSED_DATA_DIR, processed_tle_filename)

    print(f"Loading processed TLEs from: {processed_tle_file_path}")
    try:
        tles_df = pd.read_csv(processed_tle_file_path, parse_dates=['EPOCH'])
        print(f"Successfully loaded {len(tles_df)} TLEs.")
    except FileNotFoundError:
        print(f"Error: Processed TLE file not found at {processed_tle_file_path}. Please run data_processor.py first.")
        exit() # Exit if the input file is not found
    except Exception as e:
        print(f"An error occurred loading the processed TLE file: {e}")
        exit()

    # --- Propagate orbits ---
    all_propagated_data = []
    num_satellites_to_propagate = 100 # Propagate only the first N satellites for a quick test
    propagation_hours = 1 # Propagate for this many hours from each TLE's epoch
    propagation_time_step_minutes = 5 # Calculate a state every X minutes

    print(f"\nStarting orbit propagation for the first {num_satellites_to_propagate} satellites.")
    print(f"Propagation duration: {propagation_hours} hours, time step: {propagation_time_step_minutes} minutes.")
    print("This might take a while, especially for a large number of satellites/long durations.")

    for index, row in tles_df.head(num_satellites_to_propagate).iterrows():
        try:
            line1 = row['LINE1']
            line2 = row['LINE2']
            
            satellite = Satrec.twoline2rv(line1, line2, WGS72)

            start_epoch_naive = row['EPOCH'].replace(tzinfo=None) if row['EPOCH'].tzinfo is not None else row['EPOCH']

            # Call the propagation function
            propagated_states = propagate_tle(satellite, start_epoch_naive, 
                                              num_hours=propagation_hours, 
                                              time_step_minutes=propagation_time_step_minutes)
            all_propagated_data.extend(propagated_states) # Add results to the master list

        except Exception as e:
            print(f"Error processing TLE for NORAD ID {row['NORAD_CAT_ID']}: {e}")
            continue # Continue to the next satellite

    if all_propagated_data:
        propagated_df = pd.DataFrame(all_propagated_data)
        
        print(f"\nSuccessfully propagated states for {num_satellites_to_propagate} satellites.")
        print(f"Total propagated states generated: {len(propagated_df)}.")
        
        print("\nFirst 5 rows of the Propagated DataFrame:")
        print(propagated_df.head())
        print(f"\nPropagated DataFrame Info:")
        propagated_df.info()

        # Construct the output filename
        output_propagated_filename = (
            f'propagated_states_{today_str}_'
            f'first{num_satellites_to_propagate}sats_'
            f'{propagation_hours}hr_{propagation_time_step_minutes}min_step.csv'
        )
        output_propagated_path = os.path.join(PROPAGATED_DATA_DIR, output_propagated_filename)
        
        propagated_df.to_csv(output_propagated_path, index=False)
        print(f"\nPropagated states saved to: {output_propagated_path}")
    else:
        print("No propagation data generated. Please check TLE parsing and propagation logic.")