import os
import torch.nn as nn
import torch.optim as optim
from hand_to_neuro_dataloaders import get_dataloaders
from hand_to_neuro_models import TransformerModel
from hand_to_neuro_evaluate import visualize_with_real_data, visualize_with_its_own_data
import torch
import psutil
from datetime import datetime
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
parser.add_argument('--latent_dim', type=int, default=None,
                    help='Latent dimension (optional)')
parser.add_argument('--model_type', type=str, default='transformer',
                    choices=['transformer', 'lstm'], help='Model type')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float,
                    default=0.001, help='Weight decay')
parser.add_argument('--custom_prefix', type=str, default='',
                    help='Custom prefix for model name')
args = parser.parse_args()

d_model = args.d_model
latent_dim = args.latent_dim if args.latent_dim is not None and args.latent_dim > 0 else None
model_type = args.model_type
lr = args.lr
weight_decay = args.weight_decay
custom_prefix = args.custom_prefix

n_fr_bins = 9
n_trials = 2000
n_epochs = 200

if custom_prefix != '':
    prefix = custom_prefix + "_"
else:
    prefix = ""
prefix += f"{model_type}_dm{d_model}"
if latent_dim is not None:
    prefix += f"_ld{latent_dim}"
prefix += f"_lr{lr}_wd{weight_decay}"
os.makedirs('model_data', exist_ok=True)

n_future_vel_bins = 20
n_fr_bins = 9
bin_size = 0.02


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


train_loader, test_loader, test_dataset, n_neurons, max_trial_length = get_dataloaders(
    n_trials=n_trials, n_future_vel_bins=n_future_vel_bins, n_fr_bins=n_fr_bins, bin_size=bin_size, verbose=True, batch_size=100)
print(n_neurons)


input_size = (n_neurons) + 2 * n_future_vel_bins
hidden_size = d_model
model = TransformerModel(input_size, hidden_size,
                         n_neurons, n_fr_bins, max_trial_length).to(device)


# Training parameters
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Lists to store metrics
train_losses = []
val_losses = []
test_accs = []

# Initialize start time
start_time = datetime.now()

# Training loop
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for batch_velocities, batch_spikes, batch_spikes_future in train_loader:
        batch_spikes = batch_spikes.to_padded_tensor(-100).to(device)
        batch_spikes_future = batch_spikes_future.to_padded_tensor(
            -100).to(device, dtype=torch.long)
        batch_velocities = batch_velocities.to_padded_tensor(-100).to(device)
        batch_size = batch_spikes.shape[0]
        n_context_bins = batch_spikes.shape[1]

        optimizer.zero_grad()
        outputs = model(batch_spikes, batch_velocities)

        loss = criterion(outputs.reshape(-1, n_fr_bins),
                         batch_spikes_future.reshape(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        test_acc = 0
        for batch_velocities, batch_spikes, batch_spikes_future in test_loader:
            batch_spikes = batch_spikes.to_padded_tensor(-100).to(device)
            batch_spikes_future = batch_spikes_future.to_padded_tensor(
                -100).to(device, dtype=torch.long)
            batch_velocities = batch_velocities.to_padded_tensor(
                -100).to(device)

            outputs = model(batch_spikes, batch_velocities)

            # Get predicted classes
            # Shape: (batch, n_context_bins-1, n_neurons)
            pred_classes = torch.argmax(outputs, dim=3)

            # Calculate accuracy
            acc = (pred_classes == batch_spikes_future).float().mean()
            test_acc += acc.item()
            val_loss += criterion(outputs.reshape(-1, n_fr_bins),
                                  batch_spikes_future.reshape(-1)).item()

        avg_val_loss = val_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader)

        val_losses.append(avg_val_loss)
        test_accs.append(avg_test_acc)

    # Get GPU memory usage
    gpu_mem_alloc = torch.cuda.memory_allocated(
        0) / 1024**2 if torch.cuda.is_available() else 0
    gpu_mem_cached = torch.cuda.memory_reserved(
        0) / 1024**2 if torch.cuda.is_available() else 0

    # Get CPU memory usage
    cpu_mem = psutil.Process().memory_info().rss / 1024**2

    # Calculate elapsed time and estimate remaining time
    current_time = datetime.now()
    elapsed_time = current_time - start_time
    time_per_epoch = elapsed_time / (epoch + 1)
    time_left = time_per_epoch * (n_epochs - epoch - 1)
    hours_left = int(time_left.total_seconds() // 3600)
    minutes_left = int((time_left.total_seconds() % 3600) // 60)
    seconds_left = int(time_left.total_seconds() % 60)

    if (epoch + 1) % 1 == 0:
        print(f"\n[{current_time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Test Acc: {avg_test_acc:.3f}")
        print(f"\tGPU Memory: {gpu_mem_alloc:.1f}MB (allocated) {gpu_mem_cached:.1f}MB (cached) | CPU Memory: {cpu_mem:.1f}MB | Estimated time remaining: {hours_left:02d}:{minutes_left:02d}:{seconds_left:02d}")

    # Save model checkpoint
    if (epoch + 1) % 40 == 0:
        # Save model checkpoint
        torch.save(model.state_dict(),
                   f'model_data/{prefix}_epoch{epoch+1}.pt')

        # Save losses and metrics to JSON
        metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'test_acc': avg_test_acc,
            'loss_store': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_accs': test_accs,
            },
            'memory': {
                'gpu_mem_alloc': gpu_mem_alloc,
                'gpu_mem_cached': gpu_mem_cached,
                'cpu_mem': cpu_mem
            },
            'time_remaining': f"{hours_left:02d}:{minutes_left:02d}:{seconds_left:02d}",
            'hyperparameters': {
                'n_epochs': n_epochs,
                'n_neurons': n_neurons,
                'n_fr_bins': n_fr_bins,
                'n_future_vel_bins': n_future_vel_bins,
                'bin_size': bin_size,
                'd_model': d_model,
                'latent_dim': latent_dim,
                'model_type': model_type,
                'lr': lr,
                'weight_decay': weight_decay,
                'device': str(device)
            }
        }
        with open(f'model_data/{prefix}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        visualize_with_real_data(model, test_loader, n_neurons, n_fr_bins,
                                 device, prefix+f"_epoch{epoch+1}", temperature=1.0)
        visualize_with_its_own_data(model, test_dataset, n_neurons,
                                    n_fr_bins, device, prefix+f"_epoch{epoch+1}", temperature=1.0)
