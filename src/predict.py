import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
from datetime import datetime
import joblib # For loading the scaler
import matplotlib.pyplot as plt 

#Configuration for directories
script_dir = os.path.dirname(os.path.abspath(__file__))
PREPARED_DATA_DIR = os.path.join(script_dir, '..', 'data', 'prepared')
MODELS_DIR = os.path.join(script_dir, '..', 'models')

#Define the LSTM Model
class OrbitPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(OrbitPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        
        return out


if __name__ == "__main__":
    # Load Model Parameters and Data Paths ---
    today_str = datetime.utcnow().strftime('%Y%m%d')

    input_seq_len = 5
    output_seq_len = 1 
    num_features = 6 

    X_path = os.path.join(PREPARED_DATA_DIR, f'X_data_{today_str}_i{input_seq_len}_o{output_seq_len}.npy')
    y_path = os.path.join(PREPARED_DATA_DIR, f'y_data_{today_str}_i{input_seq_len}_o{output_seq_len}.npy')
    scaler_path = os.path.join(PREPARED_DATA_DIR, f'scaler_{today_str}_i{input_seq_len}_o{output_seq_len}.pkl')
    
    model_filename = f'orbit_predictor_lstm_{today_str}_i{input_seq_len}_o{output_seq_len}.pth'
    model_path = os.path.join(MODELS_DIR, model_filename)

    print(f"Loading prepared data from: {PREPARED_DATA_DIR}")
    print(f"Loading scaler from: {scaler_path}")
    print(f"Loading model from: {model_path}")

    try:
        X_data = np.load(X_path)
        y_data = np.load(y_path)
        scaler = joblib.load(scaler_path)
        print(f"Successfully loaded data: X shape {X_data.shape}, y shape {y_data.shape}")
    except FileNotFoundError:
        print(f"Error: Data, scaler, or model files not found. Ensure train_model.py was run correctly and paths match.")
        exit()
    except Exception as e:
        print(f"An error occurred loading files: {e}")
        exit()

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(X_data).float()
    y_tensor = torch.from_numpy(y_data).float()
    y_tensor_squeezed = y_tensor.squeeze(1) # For target comparison, as model outputs 2D

    # Data Splitting
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15 

    total_samples = len(X_tensor)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    dataset = TensorDataset(X_tensor, y_tensor_squeezed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    
    # Model Initialization and Loading 
    input_size = num_features  
    hidden_size = 128         
    output_size = num_features 
    num_layers =3          

    model = OrbitPredictorLSTM(input_size, hidden_size, output_size, num_layers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nSuccessfully loaded trained model from: {model_path}")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        exit()
    
    model.eval() # evaluation mode

    #  Select a Sample for Prediction 
    sample_idx = 0 

    sample_input_scaled, actual_target_scaled = test_dataset[sample_idx]
    
    print(f"\n--- Making Prediction for Test Sample Index {sample_idx} ---")
    print(f"Input sequence (scaled, first 5 rows):\n{sample_input_scaled.cpu().numpy()[:5]}")
    print(f"Actual target (scaled):\n{actual_target_scaled.cpu().numpy()}")

    # batch dimension 
    sample_input_batch = sample_input_scaled.unsqueeze(0).to(device)

    # Make Prediction 
    with torch.no_grad():
        predicted_target_scaled = model(sample_input_batch)
    
    # Remove batch dimension and convert to numpy for inverse scaling
    predicted_target_scaled_np = predicted_target_scaled.squeeze(0).cpu().numpy()

    # Inverse Transform Predictions and Actual Targets
    predicted_target_unscaled = scaler.inverse_transform(predicted_target_scaled_np.reshape(1, -1))
    actual_target_unscaled = scaler.inverse_transform(actual_target_scaled.cpu().numpy().reshape(1, -1))
    
    print(f"\nPredicted target (unscaled, KM and KM/S):")
    print(f"X: {predicted_target_unscaled[0,0]:.4f} KM, Y: {predicted_target_unscaled[0,1]:.4f} KM, Z: {predicted_target_unscaled[0,2]:.4f} KM")
    print(f"Vx: {predicted_target_unscaled[0,3]:.4f} KM/S, Vy: {predicted_target_unscaled[0,4]:.4f} KM/S, Vz: {predicted_target_unscaled[0,5]:.4f} KM/S")

    print(f"\nActual target (unscaled, KM and KM/S):")
    print(f"X: {actual_target_unscaled[0,0]:.4f} KM, Y: {actual_target_unscaled[0,1]:.4f} KM, Z: {actual_target_unscaled[0,2]:.4f} KM")
    print(f"Vx: {actual_target_unscaled[0,3]:.4f} KM/S, Vy: {actual_target_unscaled[0,4]:.4f} KM/S, Vz: {actual_target_unscaled[0,5]:.4f} KM/S")

    # Visualize a comparison
    sample_input_unscaled = scaler.inverse_transform(sample_input_scaled.cpu().numpy())
    
    # Extract X and Y for plotting
    input_x = sample_input_unscaled[:, 0]
    input_y = sample_input_unscaled[:, 1]

    # Predicted and Actual future X, Y
    pred_x = predicted_target_unscaled[0, 0]
    pred_y = predicted_target_unscaled[0, 1]

    actual_x = actual_target_unscaled[0, 0]
    actual_y = actual_target_unscaled[0, 1]

    plt.figure(figsize=(10, 8))
    plt.plot(input_x, input_y, 'o-', label='Input History (X,Y)', color='blue')
    plt.plot(pred_x, pred_y, 'X', markersize=10, label='Predicted Next State (X,Y)', color='red')
    plt.plot(actual_x, actual_y, 'o', markersize=10, label='Actual Next State (X,Y)', color='green')

    plt.plot([input_x[-1], pred_x], [input_y[-1], pred_y], 'r--', alpha=0.7)
    plt.plot([input_x[-1], actual_x], [input_y[-1], actual_y], 'g--', alpha=0.7)

    plt.xlabel('X Coordinate (KM)')
    plt.ylabel('Y Coordinate (KM)')
    plt.title('Orbit Prediction: Actual vs. Predicted Next State (X-Y Plane)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # better spatial representation
    plt.show()