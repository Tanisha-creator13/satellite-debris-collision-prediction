import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- Configuration for directories ---
script_dir = os.path.dirname(os.path.abspath(__file__))
PROPAGATED_DATA_DIR = os.path.join(script_dir, '..', 'data', 'propagated')
PREPARED_DATA_DIR = os.path.join(script_dir, '..', 'data', 'prepared') 

os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
# --- End Configuration ---

def create_sequences(data, input_seq_len, output_seq_len, feature_columns):
    """
    Creates sequences for an LSTM/Transformer model from time series data.

    Args:
        data (pd.DataFrame): DataFrame containing time-series data for a single satellite.
                             Assumes data is sorted by TIMESTAMP_UTC.
        input_seq_len (int): Number of past time steps to use as input.
        output_seq_len (int): Number of future time steps to predict as output.
        feature_columns (list): List of column names to use as features.

    Returns:
        tuple: (inputs, targets) - NumPy arrays of shape (num_sequences, seq_len, num_features)
                                   and (num_sequences, output_seq_len, num_features)
    """
    inputs, targets = [], []
    num_samples = len(data)

    # Ensure data is numpy array for faster indexing
    features = data[feature_columns].values

    for i in range(num_samples - input_seq_len - output_seq_len + 1):
        # Input sequence: past 'input_seq_len' states
        input_sequence = features[i : i + input_seq_len]
        
        # Target sequence: next 'output_seq_len' states
        target_sequence = features[i + input_seq_len : i + input_seq_len + output_seq_len]
        
        inputs.append(input_sequence)
        targets.append(target_sequence)
        
    return np.array(inputs), np.array(targets)


if __name__ == "__main__":
    # --- Load the propagated states DataFrame ---
    today_str = datetime.utcnow().strftime('%Y%m%d')

    num_sats = 100 
    prop_hours = 1
    step_minutes = 5
    
    propagated_filename = (
        f'propagated_states_{today_str}_'
        f'first{num_sats}sats_{prop_hours}hr_{step_minutes}min_step.csv'
    )
    propagated_file_path = os.path.join(PROPAGATED_DATA_DIR, propagated_filename)

    print(f"Loading propagated states from: {propagated_file_path}")
    try:
        propagated_df = pd.read_csv(propagated_file_path, parse_dates=['TIMESTAMP_UTC'])
        print(f"Successfully loaded {len(propagated_df)} propagated states.")
    except FileNotFoundError:
        print(f"Error: Propagated states file not found at {propagated_file_path}. Please run orbit_propagator.py first.")
        exit()
    except Exception as e:
        print(f"An error occurred loading the propagated states file: {e}")
        exit()

    # --- Feature Engineering Settings ---
    feature_columns = ['X_KM', 'Y_KM', 'Z_KM', 'VX_KMS', 'VY_KMS', 'VZ_KMS']
    
    # Define sequence lengths
    INPUT_SEQUENCE_LENGTH = 5 # Number of previous time steps 
    OUTPUT_SEQUENCE_LENGTH = 1 # Number of future time steps to predict 

    print(f"\nPreparing sequences with Input Length: {INPUT_SEQUENCE_LENGTH}, Output Length: {OUTPUT_SEQUENCE_LENGTH}")

    # --- Data Scaling ---
    scaler = MinMaxScaler()
    
    # This ensures consistent scaling across all data (train, val, test)
    print("Fitting scaler on all feature data...")
    all_features_data = propagated_df[feature_columns].values
    scaler.fit(all_features_data)
    
    # Transform the features 
    propagated_df[feature_columns] = scaler.transform(all_features_data)
    print("Features scaled successfully.")
    
    # --- Create sequences for each satellite ---
    all_inputs, all_targets = [], []

    # Group data by NORAD_CAT_ID to create sequences per satellite
    unique_norad_ids = propagated_df['NORAD_CAT_ID'].unique()
    print(f"Creating sequences for {len(unique_norad_ids)} unique satellites.")

    for norad_id in unique_norad_ids:
        satellite_data = propagated_df[propagated_df['NORAD_CAT_ID'] == norad_id].sort_values(by='TIMESTAMP_UTC')
        
        if len(satellite_data) >= INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH:
            inputs, targets = create_sequences(satellite_data, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, feature_columns)
            all_inputs.append(inputs)
            all_targets.append(targets)
        else:
            print(f"Skipping NORAD_CAT_ID {norad_id} due to insufficient data for sequence creation (has {len(satellite_data)} points, needs at least {INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH}).")

    # Concatenate all sequences
    if all_inputs and all_targets:
        X = np.concatenate(all_inputs, axis=0) # Inputs
        y = np.concatenate(all_targets, axis=0) # Targets

        print(f"\nTotal input sequences (X) shape: {X.shape}")    
        print(f"Total target sequences (y) shape: {y.shape}")    
        
        # --- Save the prepared data (X and y) ---
        output_X_path = os.path.join(PREPARED_DATA_DIR, f'X_data_{today_str}_i{INPUT_SEQUENCE_LENGTH}_o{OUTPUT_SEQUENCE_LENGTH}.npy')
        output_y_path = os.path.join(PREPARED_DATA_DIR, f'y_data_{today_str}_i{INPUT_SEQUENCE_LENGTH}_o{OUTPUT_SEQUENCE_LENGTH}.npy')
        
        np.save(output_X_path, X)
        np.save(output_y_path, y)
        
        scaler_filename = f'scaler_{today_str}_i{INPUT_SEQUENCE_LENGTH}_o{OUTPUT_SEQUENCE_LENGTH}.pkl'
        scaler_path = os.path.join(PREPARED_DATA_DIR, scaler_filename)
        import joblib
        joblib.dump(scaler, scaler_path)

        print(f"\nPrepared input sequences saved to: {output_X_path}")
        print(f"Prepared target sequences saved to: {output_y_path}")
        print(f"Scaler saved to: {scaler_path}")
        
    else:
        print("No sequences were generated. Check data and sequence parameters.")