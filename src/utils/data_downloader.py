import requests
import os
import sys
from datetime import datetime, timedelta

# --- Configuration ---
USERNAME = 'ytanisha217@gmail.com'
PASSWORD = 'hteA5!JLWDuFYAF'

# Define the output directory
RAW_DATA_DIR = 'data/raw/'

# Ensure the directory exists
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def download_tle_data(start_date: datetime, end_date: datetime, output_file: str):
    """
    Downloads TLE data from Space-Track.org for a specified date range.

    Args:
        start_date (datetime): The start date for the TLE data (inclusive).
        end_date (datetime): The end date for the TLE data (inclusive).
        output_file (str): The full path and filename to save the TLE data.
    """
    base_url = "https://www.space-track.org/basicspacedata/query"
    
    # Space-Track API date format: YYYY-MM-DD
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # params = {
    #     'class': 'gp_history',
    #     'epoch': f'{start_date_str}--{end_date_str}', # TLEs whose epoch falls in this range
    #     'format': 'tle',
    #     'orderby': 'NORAD_CAT_ID,EPOCH asc',
    #     'limit': '50000',
    #     'predicates': 'OBJECT_TYPE/SATELLITE,ROCKET BODY,DEBRIS' # Filter for common types
    # }

    # --- NEW: Specific query for ISS ---
    params = {
        'class': 'gp_history',
        'NORAD_CAT_ID': '25544', # ISS Norad ID
        'format': 'tle',
        'orderby': 'EPOCH desc', # Get most recent TLEs first
        'limit': '10' # Only need a few recent TLEs for ISS
    }

    print(f"Attempting to download TLEs from {start_date_str} to {end_date_str}...")

    try:
        with requests.Session() as session:
            # Login to Space-Track.org
            login_data = {'identity': USERNAME, 'password': PASSWORD}
            login_url = "https://www.space-track.org/ajaxauth/login"
            login_response = session.post(login_url, data=login_data)
            login_response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            if "Login Failed" in login_response.text:
                raise Exception("Space-Track.org login failed. Check your username and password.")
            print("Successfully logged into Space-Track.org.")

            # Make the data request
            print(f"Making API request to: {session.prepare_request(requests.Request('GET', base_url, params=params)).url}")
            response = session.get(base_url, params=params)
            response.raise_for_status() # Raise an exception for HTTP errors

            # DEBUG PRINT STATEMENT ---
            print("\n--- Space-Track.org API Response Content (for debugging) ---")
            print(response.text[:500]) # Print first 500 characters to avoid flooding console
            print("-----------------------------------------------------------\n")
            


            # Save the TLE data
            print(f"Attempting to save data to: {output_file}")
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"Successfully downloaded TLE data to: {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}", file=sys.stderr)
        print("Please check your internet connection, Space-Track.org credentials, and the API query.", file=sys.stderr)
        print("Also, Space-Track.org has rate limits and might return errors for very large requests.", file=sys.stderr)
        print("Consider reducing the date range or the limit.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Example Usage: Download TLEs for the last 3 months
    end_date = datetime.utcnow()
    # Let's try to get data for a reasonable window, e.g., the last 3 months.
    # Space-Track.org can be slow/unresponsive for very large historical queries.
    start_date = end_date - timedelta(days=90) # Last 3 months

    output_filename = os.path.join(RAW_DATA_DIR, f"tles_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.txt")

    # Call the download function
    # Only download if the file doesn't exist or is empty (to avoid re-downloading large files)
    if not os.path.exists(output_filename) or os.stat(output_filename).st_size == 0:
        download_tle_data(start_date, end_date, output_filename)
    else:
        print(f"File '{output_filename}' already exists and is not empty. Skipping download.")

    # --- Next Steps: Basic TLE Parsing with sgp4 ---
    print("\n--- Next Step: Basic TLE Parsing with sgp4 ---")
    print("Once the download is complete, you can use the sgp4 library to parse these TLEs.")
    print("Example of parsing the first TLE in the downloaded file:")
    try:
        from sgp4.api import Satrec

        with open(output_filename, 'r') as f:
            lines = f.readlines()

        # Remove empty lines if any
        lines = [line.strip() for line in lines if line.strip()]

        if len(lines) == 0:
            print("The downloaded TLE file is empty or contains no valid TLE lines.", file=sys.stderr)
        elif len(lines) % 2 != 0:
            print(f"Warning: The number of lines ({len(lines)}) in the TLE file is odd. This typically indicates an incomplete TLE or non-standard format. Only processing full TLE pairs.", file=sys.stderr)
            lines = lines[:-1] # Remove the last line if it's incomplete
        
        # Now, check if we have at least one full TLE (2 lines)
        if len(lines) >= 2:
            line1 = lines[0]
            line2 = lines[1]
            try:
                # The .twoline2rv() method expects the raw TLE lines
                satellite = Satrec.twoline2rv(line1, line2)
                print(f"Parsed first satellite (NORAD ID: {satellite.satnum}).")
                print(f"Inclination: {satellite.inclo:.2f} degrees")
                print(f"Mean Motion: {satellite.no_kozai:.4f} revs/day")
                # You can access other orbital elements from the 'satellite' object.
            except Exception as parse_e:
                print(f"Error parsing first TLE. This often means the TLE format isn't as expected, or the data itself is malformed: {parse_e}", file=sys.stderr)
                print(f"Problematic TLE lines:\nLine 1: {line1}\nLine 2: {line2}", file=sys.stderr)
        else:
            print("Not enough complete TLEs (expected at least two lines for one TLE) in the downloaded file to parse.", file=sys.stderr)


    except FileNotFoundError:
        print(f"Error: Downloaded file '{output_filename}' not found. It might have failed to download or was deleted.", file=sys.stderr)
    except ImportError:
        print("Please install 'sgp4' library: pip install sgp4", file=sys.stderr)