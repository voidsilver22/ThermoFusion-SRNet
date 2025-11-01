# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# --- Import our custom modules ---
from dataset import ThermalSRDataset
from model import CSRN, PhysicsInformedLoss

# --- 1. Dynamic Path Handling ---

# Get the absolute path of the directory where train.py lives (e.g., .../Case Study ML/Model)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the project root by going one level up (e.g., .../Case Study ML)
PROJECT_ROOT = os.path.dirname(MODEL_DIR)

# --- 2. Configuration ---

# Build robust paths from the project root
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "preprocessing", "processed_data")
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")

# Save checkpoints inside the 'Model' directory
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
PATCH_SIZE = 72 # Must be divisible by 3

# (Lambda weights for our loss function)
# Prioritizing Kinetic Loss (K) and Degradation Loss
LAMBDA_K = 1.0
LAMBDA_DEGRAD = 0.8
LAMBDA_SSIM = 0.1

# ... (the rest of your train.py file, starting with the main() function)
def main():
    # --- 1. Setup ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. DataLoaders ---
    print("Loading datasets...")
    train_dataset = ThermalSRDataset(
        processed_dir=TRAIN_DIR,
        patch_size=PATCH_SIZE,
        augment=True
    )
    val_dataset = ThermalSRDataset(
        processed_dir=VAL_DIR,
        patch_size=PATCH_SIZE,
        augment=False  # No augmentation for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4, # Adjust based on your CPU cores
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- 3. Model, Loss, Optimizer ---
    print("Initializing model...")
    model = CSRN().to(device)
    criterion = PhysicsInformedLoss(
        lambda_k=LAMBDA_K,
        lambda_ssim=LAMBDA_SSIM,
        lambda_degrad=LAMBDA_DEGRAD
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')

    # --- 4. Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_total = 0.0
        
        # TQDM for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch in train_pbar:
            lr_lst = batch["lr_lst"].to(device)
            hr_guide = batch["hr_guide"].to(device)
            hr_target = batch["hr_target"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_hr = model(lr_lst, hr_guide)
            
            # Calculate loss
            loss, loss_components = criterion(pred_hr, hr_target, lr_lst, device)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            
            # Update progress bar description
            train_pbar.set_postfix(
                Loss=f"{loss.item():.4f}",
                L_k=f"{loss_components['loss_k']:.4f}",
                L_deg=f"{loss_components['loss_degrad']:.4f}"
            )

        avg_train_loss = train_loss_total / len(train_loader)

        # --- 5. Validation Loop ---
        model.eval()
        val_loss_total = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                lr_lst = batch["lr_lst"].to(device)
                hr_guide = batch["hr_guide"].to(device)
                hr_target = batch["hr_target"].to(device)
                
                pred_hr = model(lr_lst, hr_guide)
                loss, _ = criterion(pred_hr, hr_target, lr_lst, device)
                val_loss_total += loss.item()
                
                val_pbar.set_postfix(Loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss_total / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")

        # --- 6. Save Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path} (Val Loss: {best_val_loss:.6f})")

    print("--- Training Complete ---")

if __name__ == "__main__":
    main()