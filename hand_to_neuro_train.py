import os
import torch.nn as nn
import torch.optim as optim
from hand_to_neuro_dataloaders import get_dataloaders
from hand_to_neuro_models import TransformerModel
from hand_to_neuro_visualize import visualize_with_real_data, visualize_with_its_own_data
import torch

n_fr_bins = 9
d_model = 512
latent_dim = None
model_type = "transformer"  # transformer, lstm


n_trials = 200
n_epochs = 10
lr = 0.001
weight_decay = 0.001


prefix = f"{model_type}_dm{d_model}"
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
    n_trials=n_trials, n_future_vel_bins=n_future_vel_bins, n_fr_bins=n_fr_bins, bin_size=bin_size, verbose=True)
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

    if (epoch + 1) % 1 == 0:
        print(f"\nEpoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Test Acc: {avg_test_acc:.3f}")

    # Save model checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'test_acc': avg_test_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_accs': test_accs
    }
    torch.save(checkpoint, f'model_data/{prefix}_epoch{epoch+1}.pt')

    visualize_with_real_data(model, test_loader, n_neurons, n_fr_bins,
                             device, prefix+f"_epoch{epoch+1}", temperature=1.0)
    visualize_with_its_own_data(model, test_dataset, n_neurons,
                                n_fr_bins, device, prefix+f"_epoch{epoch+1}", temperature=1.0)
