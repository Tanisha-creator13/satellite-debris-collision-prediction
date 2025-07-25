import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,random_split
import numpy as np
import os
from datetime import datetime
import joblib # for loading the scaler

# configuration for directories
script_dir=os.path.dirname(os.path.abspath(__file__))
PREPARED_DATA_DIR=os.path.join(script_dir,'..','data','prepared')
MODEL_DIR=os.path.join(script_dir,'..','models') # new directory for saving models

os.makedirs(MODEL_DIR, exist_ok=True)

# define lstm model
class OrbitPredictorLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(OrbitPredictorLSTM,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # fully connected layer to map LSTM output to desired output_size
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)

        #pass input through LSTM
        out,(hn,cn)=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:]) # Take output from the last time step of the sequence
        return out

if __name__ =="__main__":
    # load prepared data
    today_str=datetime.utcnow().strftime('%Y%m%d')
    input_seq_len=5
    output_seq_len=1
    num_features=6 
    X_path = os.path.join(PREPARED_DATA_DIR, f'X_data_{today_str}_i{input_seq_len}_o{output_seq_len}.npy')
    y_path = os.path.join(PREPARED_DATA_DIR, f'y_data_{today_str}_i{input_seq_len}_o{output_seq_len}.npy')
    scaler_path = os.path.join(PREPARED_DATA_DIR, f'scaler_{today_str}_i{input_seq_len}_o{output_seq_len}.pkl')
    print(f"Loading X data from: {X_path}")
    print(f"Loading y data from: {y_path}")
    print(f"Loading scaler from: {scaler_path}")
    try:
            X_data = np.load(X_path)
            y_data = np.load(y_path)
            scaler = joblib.load(scaler_path)
            print(f"Successfully loaded data: X shape {X_data.shape}, y shape {y_data.shape}")
    except FileNotFoundError:
            print(f"Error: Prepared data files or scaler not found. Please ensure feature_engineering.py was run correctly with matching parameters.")
            exit()
    except Exception as e:
            print(f"An error occurred loading prepared data: {e}")
            exit()

    # convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_data).float()
    y_tensor = torch.from_numpy(y_data).float()

    # For a sequence-to-one prediction, y_tensor needs to be (batch_size, num_features)
    y_tensor_squeezed = y_tensor.squeeze(1) # Shape becomes (num_sequences, num_features)

    # Data Splitting 
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15 

    total_samples = len(X_tensor)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size # Ensure all samples are used

    print(f"\nSplitting data: Train {train_size}, Val {val_size}, Test {test_size} samples.")
    # Create a TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor_squeezed)
    # Use random_split for splitting
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42) # For reproducibility
    )

    # --- DataLoaders ---
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Model Initialization 
    input_size = num_features  # Number of features per time step 
    hidden_size = 128        # Number of features in the hidden state of the LSTM
    output_size = num_features # Predicting the same 6 features
    num_layers = 3           # Number of stacked LSTM layers

    model = OrbitPredictorLSTM(input_size, hidden_size, output_size, num_layers)

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"\nUsing device: {device}")
    print(model)

    # Loss Function and Optimizer 
    criterion = nn.MSELoss() # common for regression tasks
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop 
    num_epochs = 200
    
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train() # training mode
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward and optimize
            optimizer.zero_grad() 
            loss.backward()       # Compute gradients
            optimizer.step()      # Update weights
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        #  Validation Loop 
        model.eval() # evaluation mode
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0: 
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    print("\nTraining complete.")

    # Evaluation on Test Set
    model.eval() # evaluation mode
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f'\nTest Loss: {avg_test_loss:.6f}')

    # Save the Trained Model 
    model_filename = f'orbit_predictor_lstm_{today_str}_i{input_seq_len}_o{output_seq_len}.pth'
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")