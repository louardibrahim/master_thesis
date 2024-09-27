import torch
from torch.utils.data import DataLoader, random_split
from model import CNN_BiLSTM_Attention
from data_processing import load_and_preprocess_data
from train import train, validate
from utils import SaveBestModel, save_plots

# Hyperparameters
vocab_size = 35000  # Example vocab size, should match the actual size of your vocabulary
embedding_dim = 128
output_dim = 2  # Binary classification (suicide vs non-suicide)
learning_rate = 0.001
epochs = 10
batch_size = 32
dropout_rate = 0.2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
data_path = './suicide-watch.csv'  # Replace with the actual path to your dataset
dataset = load_and_preprocess_data(data_path)

# Split dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the CNN_BiLSTM_Attention model
model = CNN_BiLSTM_Attention(vocab_size, embedding_dim, output_dim, dropout=dropout_rate).to(device)

# Define the loss function (CrossEntropyLoss for classification) and optimizer (Adam)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the SaveBestModel utility for saving the best model based on validation loss
save_best_model = SaveBestModel()

# Lists to store accuracy and loss for plotting later
train_acc, valid_acc, train_loss, valid_loss = [], [], [], []

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}:")

    # Train the model
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, device)
    train_loss.append(train_epoch_loss)
    train_acc.append(train_epoch_acc)

    # Validate the model
    valid_epoch_loss, valid_epoch_acc = validate(model, test_loader, criterion, device)
    valid_loss.append(valid_epoch_loss)
    valid_acc.append(valid_epoch_acc)

    # Print the results for this epoch
    print(f"Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_acc:.2f}%")
    print(f"Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_epoch_acc:.2f}%")

    # Save the best model based on validation loss
    save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, './best_model.pth')

# Save accuracy and loss plots
save_plots(train_acc, valid_acc, train_loss, valid_loss, output_dir='./')

print("Training completed.")
