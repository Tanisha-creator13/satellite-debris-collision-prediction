import requests
import os
from datetime import datetime, timedelta 
from sgp4.api import WGS72, Satrec # Keeping for parsing example


RAW_DATA_DIR = 'data/raw/' 
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def download_data_from_url(url: str, output_file: str):
    """
    Downloads data from a given URL and saves it to a file.
    """
    print(f"Attempting to download data from: {url}")
    try:
        # A simple requests.get, no session or complex auth needed for CelesTrak
        response = requests.get(url, timeout=30) # timeout for robustness
        response.raise_for_status() # Raising an exception for HTTP errors

        with open(output_file, 'w') as f:
            f.write(response.text)
        print(f"Successfully downloaded data to: {output_file}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading from {url}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    today_str = datetime.utcnow().strftime('%Y%m%d')
    output_filename = os.path.join(RAW_DATA_DIR, f'tles_celestrak_active_{today_str}.txt')

    # CelesTrak URL for all active TLEs 
    celestrak_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"

    print("\n--- Downloading TLEs from CelesTrak.org ---")
    download_success = download_data_from_url(celestrak_url, output_filename)

    if download_success:
        print("\n--- Next Step: Basic TLE Parsing with sgp4 ---")
        try:
            with open(output_filename, 'r') as f:
                tle_lines = f.readlines()

            # Filter out empty lines and non-TLE lines (like headers if present)
            tle_lines = [line.strip() for line in tle_lines if line.strip() and not line.startswith('#')]

            if len(tle_lines) >= 2:
                # Find the start of the first TLE
                first_tle_start_index = -1
                for i in range(len(tle_lines)):
                    if tle_lines[i].startswith('1 '):
                        first_tle_start_index = i
                        break

                if first_tle_start_index != -1 and (first_tle_start_index + 1) < len(tle_lines) and tle_lines[first_tle_start_index+1].startswith('2 '):
                    line1 = tle_lines[first_tle_start_index]
                    line2 = tle_lines[first_tle_start_index + 1]

                    satellite = Satrec.twoline2rv(line1, line2, WGS72)
                    print(f"Successfully parsed the first TLE (Line 1: {line1}, Line 2: {line2})")
                    print(f"Satellite Name: {satellite.satnum} (using NORAD ID as name for simplicity)")
                    print(f"Orbital Inclination: {satellite.inclo:.4f} radians")
                    print(f"Mean Motion: {satellite.no_kozai:.4f} revolutions/day")
                else:
                    print("Could not find a complete first TLE (lines 1 and 2) in the downloaded file to parse.")
            else:
                print("The downloaded TLE file is empty or contains no valid TLE lines.")
                print("Not enough complete TLEs (expected at least two lines for one TLE) in the downloaded file to parse.")

        except FileNotFoundError:
            print(f"Error: Downloaded file '{output_filename}' not found for parsing.")
        except Exception as e:
            print(f"An error occurred during TLE parsing: {e}")
    else:
        print("Download failed, so cannot proceed with parsing.")