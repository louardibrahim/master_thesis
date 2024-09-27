import torch
import matplotlib.pyplot as plt

class SaveBestModel:
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, path):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, path)

def save_plots(train_acc, valid_acc, train_loss, valid_loss, output_dir='./'):
    epochs = range(1, len(train_acc) + 1)

    # Accuracy plot
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_acc, color='green', label='Train Accuracy')
    plt.plot(epochs, valid_acc, color='blue', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{output_dir}/accuracy.png')

    # Loss plot
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss, color='orange', label='Train Loss')
    plt.plot(epochs, valid_loss, color='red', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss.png')
