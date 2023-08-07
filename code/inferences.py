#import argparse
import csv
import os
import torch
import questionary
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from datasets import EdgeDataset, EdgeDataBatch  # * Important for wrapping PYG tensors to pytorch tensors
from DeepNeuralNetwork_class import DeepNeuralNetwork # * Class file for model

THRESH = 0.55
BATCH_SIZE = 1000

def read_csv_data(file_path, header_truths, header_scores):
    truths, scores = [], []

    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            truths.append(int(row[header_truths]))
            scores.append(float(row[header_scores]))

    return truths, scores

def export_csv_data(csv_row_data, model_name, dataset_type, model_type):
    # * Create a folder to save the plot
    plot_folder = f"./csv/models/{model_type}/{model_name}/"
    os.makedirs(plot_folder, exist_ok=True)

    file_path = f"{plot_folder}inferences_{dataset_type}_{model_name}.csv"
    file_exists = os.path.exists(file_path)
    
    print("...inference csv data file INITIALIZED...")
    if file_exists:
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['idx', 'truth', 'score'])
            for row in csv_row_data:
                csvwriter.writerow(row)
    else:
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['idx', 'truth', 'score'])
            for row in csv_row_data:
                csvwriter.writerow(row)

    print(f"...inference csv {model_name} data file FINALIZED...")

def inference(model, dataloader, device, dataset_type, model_name, model_type):
    total_correct, total_samples, row_index = 0, 0, 0
    truth_labels_history, predicted_scores_history, csv_row_data = [], [], []

    print("...INFERENCE INITIALIZED...")
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.edge_attr)
            predicted_scores = outputs.squeeze(1).cpu().numpy().flatten()
            predicted_scores_round = [round(score, 17) for score in predicted_scores] 
            predicted_scores_history.extend(predicted_scores_round)
            truth_labels = data.y.int().squeeze().cpu().numpy()
            truth_labels_history.extend(truth_labels)

            # * Calculating True Positives & True Negatives for accuracy rating via THRESH
            TP = torch.sum((data.y == 1).squeeze() & (outputs >= THRESH).squeeze()).item()
            TN = torch.sum((data.y == 0).squeeze() & (outputs <  THRESH).squeeze()).item()
            correct = TP + TN
            total_correct += correct
            total_samples += len(truth_labels)

            # * Append the data to the inference_data list for exporting to CSV
            print("Batch No.", batch_idx)
            batch_size = data.y.size(0)
            for i in range(batch_size):
                if i >= len(truth_labels_history) or i >= len(predicted_scores_round):
                # Skip if the index exceeds the available data in the lists
                    break
                row_index += 1
                truth = truth_labels[i].item()
                score = round(predicted_scores_round[i].item(), 17)
                csv_row_data.append([row_index, truth, score])
                print(f"idx: {row_index}\ttruth: {truth}\tscore: {score}")
            print("--------------------------------------------\n")

    overall_accuracy = total_correct / total_samples
    print(f"Overall Accuracy (THRESHOLD = {THRESH}): {overall_accuracy:0.3f}")
    print("...INFERENCE COMPLETED...\n")

    export_csv_data(csv_row_data, model_name, dataset_type, model_type)

    return truth_labels_history, predicted_scores_history

def get_model_path():
    # Define the path to the model folder
    models_folder_options = f'./LineSegmentClassifier/models'

    # Get the list of model files in the folder
    model_files = os.listdir(models_folder_options)

    # Filter .pt files
    model_files = [file for file in model_files if file.endswith('.pt')]

    # Sort the files for better visibility
    model_files.sort()

    # Create the list of choices for questionary select question
    model_choices = [os.path.join(models_folder_options, file) for file in model_files]

    # Prompt the user to select the model file
    model_file = questionary.select(
        'Select the model file:',
        choices=model_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    return model_file

def get_dataset_path(dataset_type):
    # Define the paths to the dataset folders
    dataset_options = f'./LineSegmentClassifier/datasets/{dataset_type}'

    # Get the list of files in the dataset folder
    dataset_files = os.listdir(dataset_options)

    # Sort the files for better visibility
    dataset_files.sort()

    # Create the list of choices for questionary select question
    dataset_choices = [os.path.join(dataset_options, file) for file in dataset_files]

    # Prompt the user to select dataset path
    dataset_path = questionary.select(
        f'Select the {dataset_type} dataset route:',
        choices=dataset_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    return dataset_path

def main():
    MODEL_PATH = get_model_path()
    DNN_DATASET_PATH = get_dataset_path('test')
    #GNN_CSV_PATH = './csv/models/GNN/ChangGNN_MDnodes_LSedges_modelChangNet_nhidden2_hiddensize200_lrStepLR0.005_epoch50_test_inferences.csv'
    #DNN_CSV_PATH = './csv/models/DNN/inferences_test_DNN.csv'
    
    # * Get model name
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    DNN_dataset = torch.load(DNN_DATASET_PATH)
    
    # * Create data loader
    DNN_loader = DataLoader(
        EdgeDataset(DNN_dataset), batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda batch: EdgeDataBatch(batch)
    )

    # * Get if device has a CUDA GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # * Initialize the neural network in available GPU
    model = DeepNeuralNetwork().to(device)

    # * Load the pre-trained model
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print("Generate TEST GNN INFERENCES for .csv file INITIALIZED")
    
    truths_DNN, scores_DNN = inference(model, DNN_loader, device, 'test', f'{model_name}', 'DNN')
    
    print("TEST GNN INFERENCES for .csv file FINALIZED")

if __name__ == '__main__':
    main()
