import argparse
import json
import time
import os
import questionary
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_batch
from datasets import EdgeDataset, EdgeDataBatch # * Important for wrapping PYG tensors to pytorch tensors
from DeepNeuralNetwork_class import DeepNeuralNetwork

# Accuracy plot threshold constat
THRESH = 0.55

def loss_curve_plot(test_loss_history, train_loss_history, model_name, epochs):
    # * Create a folder to save the plot
    plot_folder = f"./plots/models/{model_name}/"
    os.makedirs(plot_folder, exist_ok=True)

    # Generate x-axis values from 1 to number_epochs
    x = list(range(0, epochs))

    fig, axes = plt.subplots()
    axes.plot(x, train_loss_history, label='Train Loss', color='blue', marker='o')
    axes.plot(x, test_loss_history, label='Test Loss', color='red', marker='x')
    axes.set_title(f'Loss Curve of {model_name}')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.legend()
    plt.savefig(f'{plot_folder}loss_curve_{model_name}.png')
    print(f"Loss curve saved to {plot_folder}loss_curve_{model_name}.png")

def train(model, dataloader, optimizer, device):
    loss = 0
    
    model.train()
    
    print("...TRAINING INITIALIZED...")
    for data_i, data in enumerate(dataloader):
        data.to(device)
        
        outputs = model(data.x, data.edge_index, data.edge_attr)
        labels, outputs = data.y, outputs.squeeze(1)
        loss = F.binary_cross_entropy(outputs, labels, reduction="mean")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if data_i % 100 == 0:
            print(f"Iteration: {data_i}, Loss: {loss.item()}")

    print("...TRAINING COMPLETED...\n")

    return loss.item()

def test(model, dataloader, device):
    losses, accs = [], []
    
    model.eval()
    
    print("...TESTING INITIALIZED...")
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            
            output = model(data.x, data.edge_index, data.edge_attr)
            TP = torch.sum((data.y == 1).squeeze() & (output >= THRESH).squeeze()).item()
            TN = torch.sum((data.y == 0).squeeze() & (output <  THRESH).squeeze()).item()
            FP = torch.sum((data.y == 0).squeeze() & (output >= THRESH).squeeze()).item()
            FN = torch.sum((data.y == 1).squeeze() & (output <  THRESH).squeeze()).item()
            acc = (TP+TN)/(TP+TN+FP+FN)
            loss = F.binary_cross_entropy(output.squeeze(1), data.y, reduction="mean").item()
            
            accs.append(acc)
            losses.append(loss)

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accs)

    print(f"test loss:     {avg_loss:0.6f}", flush=True)
    print(f"test accuracy: {avg_acc:0.6f}", flush=True)
    print("...TESTING COMPLETED...\n")

    return avg_loss

def get_dataset_paths():
    # Define the paths to the test and train dataset folders
    test_dataset_options = './LineSegmentClassifier/datasets/test'
    train_dataset_options = './LineSegmentClassifier/datasets/train'

    # Get the list of files in the test and train dataset folders
    test_files = os.listdir(test_dataset_options)
    train_files = os.listdir(train_dataset_options)

    # Sort the files for better visibility
    test_files.sort()
    train_files.sort()

    # Create the list of choices for questionary select questions
    test_choices = [os.path.join(test_dataset_options, file) for file in test_files]
    train_choices = [os.path.join(train_dataset_options, file) for file in train_files]

    # Prompt the user to select test and train dataset paths
    test_path = questionary.select(
        'Select the test dataset file:',
        choices=test_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    train_path = questionary.select(
        'Select the train dataset file:',
        choices=train_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    return test_path, train_path

def export_json_summary(model, epoch, learning_rate, 
                        batch_size, train_loss_history, test_loss_history, 
                        start_time, current_time
                        ):
    model_filename = f'LSC_DNN_model_epoch{epoch}_{current_time}'
    summary_filename = f'LSC_DNN_summary_epoch{epoch}_{current_time}'
    end_time = time.time()
    elapsed_time = end_time - start_time

    # * Absolute paths with custom names and start timestamp
    MODEL_PATH = f'./LineSegmentClassifier/models/{model_filename}.pt'
    SUMMARY_PATH = f'./LineSegmentClassifier/models/summary/{summary_filename}.json'
            
    torch.save(model.state_dict(), MODEL_PATH)
            
    summary = {
        'model_name': model_filename + '.pt',
        'learning_rate': learning_rate,
        'number_epochs': epoch,
        'batch_size': batch_size,
        'start_time_s': start_time,
        'end_time_s': end_time,
        'elapsed_time_s': elapsed_time,
        'train_loss': train_loss_history,
        'test_loss': test_loss_history, 
    }

    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=4)
        f.write('\n')
        f.close()

    print(f"\nTraining loop completed at: {time.ctime(end_time)}\n")
    print("Total training time:", elapsed_time, "seconds\n")
    print(f"Model saved to {MODEL_PATH}")
    print(f'Summary saved to {SUMMARY_PATH}')

def main():

    # * Define hyperparameters
    LEARNING_RATE = 0.002
    NUMBER_EPOCHS = 100
    BATCH_SIZE = 10000

    # * list of summary values of loss and accuracy
    train_loss , test_loss = 0, 0
    train_loss_history, test_loss_history = [], []

    # * Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--number_epochs', type=int, default=NUMBER_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()

    # * Update hyperparameters
    LEARNING_RATE = args.learning_rate
    NUMBER_EPOCHS = args.number_epochs
    BATCH_SIZE = args.batch_size

    # * Get the dataset paths using inquirer
    TEST_DATASET_PATH, TRAIN_DATASET_PATH = get_dataset_paths()

    # * Load dataset
    train_dataset = torch.load(TRAIN_DATASET_PATH)
    test_dataset = torch.load(TEST_DATASET_PATH)

    # * Create data loaders
    train_loader = DataLoader(
            EdgeDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=lambda batch: EdgeDataBatch(batch)
        )
    test_loader = DataLoader(
            EdgeDataset(test_dataset), batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=lambda batch: EdgeDataBatch(batch)
        )

    # * Get if device has a CUDA GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # * Initialize the neural network in available GPU
    model = DeepNeuralNetwork().to(device)

    # * Initialize the DNN's optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # * Beginning timestamp
    start_time = time.time()
    current_time = time.strftime('%Y%m%d_%H%M%S')

    # * Training loop
    print(f"Training loop started at: {time.ctime(start_time)}.\n")
    for epoch in range(NUMBER_EPOCHS):
        print(f"BEGIN of EPOCH No. {epoch+1}")
        train_loss = train(model, train_loader, optimizer, device)
        test_loss = test(model, test_loader, device)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        if epoch % 5 == 0:
            export_json_summary(model, epoch, LEARNING_RATE, 
                                BATCH_SIZE, train_loss_history, test_loss_history,
                                start_time, current_time)

        print(f"END of EPOCH No. {epoch+1}\n")

if __name__ == '__main__':
    main()
