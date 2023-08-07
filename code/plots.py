import argparse
import csv
import questionary
import json
import os
import torch
import re
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from datasets import EdgeDataset, EdgeDataBatch  # * Important for wrapping PYG tensors to pytorch tensors
#from DeepNeuralNetwork_class import DeepNeuralNetwork # * Class file for model

plt.rcParams.update({"figure.facecolor":  (1,1,1,0)})

def find_threshold_for_tpr(tpr_array, fpr_array, threshold_array, tpr_threshold):
    # Convert the input arrays to numpy arrays
    tpr_array = np.array(tpr_array)
    threshold_array = np.array(threshold_array)

    # Check if the arrays have the same length
    if len(tpr_array) != len(threshold_array):
        raise ValueError("The 'tpr_array' and 'threshold_array' must have the same length.")

    # Find the first index where TPR is greater than or equal to tpr_threshold
    index = np.where(tpr_array > tpr_threshold)[0]

    # Get the corresponding threshold value and TPR value
    if index.size > 0:
        threshold = threshold_array[index[0]]
        tpr_value = tpr_array[index[0]]
        fpr_value = fpr_array[index[0]]
        return threshold, tpr_value, fpr_value
    else:
        raise ValueError("No threshold value found for the given TPR threshold.")

# * final name: Loss Curve
def loss_curve_plot(test_loss_history, train_loss_history, model_name, epochs):
    # * Create a folder to save the plot
    plot_folder = f"./plots/models/{model_name}/"
    os.makedirs(plot_folder, exist_ok=True)

    # Generate x-axis values from 1 to number_epochs
    x = list(range(0, epochs))

    fig, axes = plt.subplots(figsize=(12, 9))
    axes.set_title(f'Loss Curve Big DNN', fontsize=36) # * final name: Loss Curve
    axes.plot(x, train_loss_history, label='Train Loss', color='blue', marker='o')
    axes.plot(x, test_loss_history, label='Test Loss', color='red', marker='x')
    axes.set_xlabel('Epoch', fontsize=32)
    axes.set_ylabel('Loss', fontsize=32)
    axes.legend(fontsize=24)
    plt.savefig(f'{plot_folder}loss_curve_{model_name}.png')
    print(f"Loss curve saved to {plot_folder}loss_curve_{model_name}.png")

# * final name: ROC Curve
def roc_curve_plot(truth, scores, model_name, dataset_type):
    # * Create a folder to save the plot
    plot_folder = f"./plots/models/{model_name}/"
    os.makedirs(plot_folder, exist_ok=True)

    # * Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(truth, scores)
    roc_auc = auc(fpr, tpr)

    # * Plot ROC curve
    print("...ROC plot INITIALIZED...")
    fig, axes = plt.subplots(figsize=(12, 9))
    axes.set_title(f'ROC Curve of {model_name}', fontsize=36) # * final name: ROC Curve
    axes.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.6f)' % roc_auc)
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.0])
    axes.set_xlabel('False Positive Rate', fontsize=32)
    axes.set_ylabel('True Positive Rate', fontsize=32)
    axes.legend(loc="lower right", fontsize=24)  # Display the legend at the lower right corner
    plt.savefig(f'{plot_folder}/roc_curve_{dataset_type}_{model_name}.png')
    print(f"...ROC plot {model_name} FINALIZED...")

# * final name 1: ROC Curve GNN vs Big DNN
# * final name 2: ROC Curve GNN vs Small DNN
def roc_curve_plot(labels_gnn, scores_gnn, labels_dnn, scores_dnn, model_name):

    # * Calculate ROC curve for GNN
    fpr_gnn, tpr_gnn, thresholds_gnn = roc_curve(labels_gnn, scores_gnn)
    
    # * ... for DNN
    fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(labels_dnn, scores_dnn)

    threshold_dnn_95, tpr_value_dnn_95, fpr_value_dnn_95 = find_threshold_for_tpr(tpr_dnn, fpr_dnn, thresholds_dnn, tpr_threshold=0.95)
    threshold_dnn_99, tpr_value_dnn_99, fpr_value_dnn_99 = find_threshold_for_tpr(tpr_dnn, fpr_dnn, thresholds_dnn, tpr_threshold=0.99)
    threshold, tpr_value_gnn_95, fpr_value_gnn_95 = find_threshold_for_tpr(tpr_gnn, fpr_gnn, thresholds_gnn, tpr_threshold=0.95)
    threshold, tpr_value_gnn_99, fpr_value_gnn_99 = find_threshold_for_tpr(tpr_gnn, fpr_gnn, thresholds_gnn, tpr_threshold=0.99)

    # Plot ROC curve
    print("...ROC plot INITIALIZED...")
    fig, axes = plt.subplots(figsize=(12, 9))
    # Color scheme for scatter points and vertical lines
    colors = ['#1f77b4', '#147aff', '#1f4068', '#00d600']
    # * final name 1: ROC Curve GNN vs Big DNN
    # * final name 2: ROC Curve GNN vs Small DNN
    axes.set_title(f'ROC Curve of {model_name}', fontsize=36)
    axes.plot(fpr_gnn, tpr_gnn, color='darkorange', lw=2, label='GNN (area = %0.4f)' % auc(fpr_gnn, tpr_gnn), zorder=2)
    axes.plot(fpr_dnn, tpr_dnn, color='darkblue', lw=2, label='DNN (area = %0.4f)' % auc(fpr_dnn, tpr_dnn), zorder=2)
    axes.axvline(x=fpr_value_dnn_95, color=colors[1], linestyle='--', label=f'DNN TPR > 0.95', zorder=1)
    axes.axvline(x=fpr_value_dnn_99, color=colors[3], linestyle='--', label=f'DNN TPR > 0.99', zorder=1)
    axes.scatter(fpr_value_dnn_95, tpr_value_dnn_95, color=colors[1], marker='s', s=60, label=f'DNN > {threshold_dnn_95:.4f}', zorder=3)
    axes.scatter(fpr_value_dnn_99, tpr_value_dnn_99, color=colors[3], marker='s', s=60, label=f'DNN > {threshold_dnn_99:.4f}', zorder=3)
    #axes.scatter(fpr_value_gnn_95, tpr_value_gnn_95, color=colors[2], s=100, label='GNN > {threshold_gnn_95:.6f}')
    #axes.scatter(fpr_value_gnn_99, tpr_value_gnn_99, color=colors[3], s=100, label='GNN > {threshold_dnn_99:.6f}')
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.0])
    #axes.set_xscale('log')
    axes.set_xlabel('False Positive Rate', fontsize=32)
    axes.set_ylabel('True Positive Rate', fontsize=32)
    axes.legend(loc="lower right", fontsize=24)  # Display the legend at the lower right corner
    plt.savefig(f'./plots/models/roc_curve_{model_name}.png')
    print(f"...ROC plot {model_name} FINALIZED...")

# * final name 1: Predicted Scores for Big DNN
# * final name 2: Predicted Scores for Small DNN
def predicted_scores_histogram(scores, truths, model_name, dataset_type):
    # * Separate predicted scores for each class based on truth values
    scores_class_0 = np.array(scores)[np.where(np.array(truths) == 0)]
    scores_class_1 = np.array(scores)[np.where(np.array(truths) == 1)]

    # * Calculate the normalization factors
    num_class_0 = len(scores_class_0)
    num_class_1 = len(scores_class_1)
    norm_factor_0 = 1.0 / num_class_0
    norm_factor_1 = 1.0 / num_class_1

    # * Create a folder to save the plot
    plot_folder = f"./plots/models/{model_name}/"
    
    os.makedirs(plot_folder, exist_ok=True)

    # * Plot histograms for each class with explicit normalization
    print("...predicted scores plot INITIALIZED...")
    
    fig, axes = plt.subplots(figsize=(12,9))
        # * final name 1: Predicted Scores for Big DNN
        # * final name 2: Predicted Scores for Small DNN
    axes.set_title(f'Predicted Scores Big DNN', fontsize=36)
    axes.hist(scores_class_0, bins=100, color='blue', alpha=0.7, label='Fake', density=False, weights=np.full_like(scores_class_0, norm_factor_0))
    axes.hist(scores_class_1, bins=100, color='red', alpha=0.7, label='Real', density=False, weights=np.full_like(scores_class_1, norm_factor_1))
    axes.set_xlabel('Scores', fontsize=32)
    axes.set_ylabel('Count', fontsize=32)
    axes.legend(fontsize=24)
    plt.savefig(f'{plot_folder}predicted_scores_{dataset_type}_{model_name}.png')
    
    print("...predicted scores plot FINALIZED...")

def node_features_histograms(data, plot_name, dataset_name, dataset_type):
    name_node_features = [
        'MD_0_x', 
        'MD_0_y', 
        'MD_0_z', 
        'MD_1_x', 
        'MD_1_y', 
        'MD_1_z', 
        'MD_dphichange',
    ]

        # * Create a folder to save the plot
    plot_folder = f"./plots/dataset/{dataset_type}/{dataset_name}/node_features"
    os.makedirs(plot_folder, exist_ok=True)

    for i in range(7):
        fig, axes = plt.subplots()    
        axes.hist(data[:,i], bins=get_equidistant_bins(data[:,i]))
        axes.set_title(f'{plot_name} {name_node_features[i]} {dataset_name}')
        axes.set_xlabel(name_node_features[i])
        axes.set_ylabel('Count')
        plt.savefig(f'{plot_folder}/{plot_name}_{dataset_name}-{name_node_features[i]}.png')
        print(f"Plot {plot_name}-{name_node_features[i]} for {dataset_name} generated!\n")

def edge_features_histograms(data, plot_name, dataset_name, dataset_type):
    edge_features_name = [
        'LS_pt', 
        'LS_eta', 
        'LS_phi'
    ]

    # * Create a folder to save the plot
    plot_folder = f"./plots/dataset/{dataset_type}/{dataset_name}/edge_features"
    os.makedirs(plot_folder, exist_ok=True)

    for i in range(3):
        fig, axes = plt.subplots()
        if i == 0:
            LS_pt = data[:,0]
            #LS_pt[LS_pt > 2000] = 2000
            LS_pt[LS_pt > 2000] = 2000
            axes.hist(LS_pt, bins=get_equidistant_bins(LS_pt))    
        else:
            axes.hist(data[:,i], bins=get_equidistant_bins(data[:,i]))        
            axes.set_title(f'{plot_name} {edge_features_name[i]} {dataset_name}')
            axes.set_xlabel(edge_features_name[i])
            axes.set_ylabel('Count')
            plt.savefig(f'{plot_folder}/{plot_name}-{dataset_name}-{edge_features_name[i]}.png')
            print(f"Plot {plot_name}-{edge_features_name[i]} for {dataset_name} generated!\n")

def truth_label_histograms(data, plot_name, dataset_name, dataset_type):
    # * Create a folder to save the plot
    plot_folder = f"./plots/dataset/{dataset_type}/{dataset_name}/truth_label"
    os.makedirs(plot_folder, exist_ok=True)

    fig, axes = plt.subplots()
    axes.hist(data, bins=[0,1,2])
    axes.set_title(f'{plot_name} LS_isFake {dataset_name}')
    axes.set_xlabel('LS_isFake')
    axes.set_ylabel('Count')
    plt.savefig(f'{plot_folder}/{plot_name}-{dataset_name}-LS_isFake.png')
    print(f"Plot {plot_name}-LS_isFake for {dataset_name} generated!\n")

def generate_feature_plots(data, dataset_name, dataset_type):    
    node_features_histograms(data, 'node_features', dataset_name, dataset_type)
    edge_features_histograms(data, 'edge_features', dataset_name, dataset_type)
    truth_label_histograms(data, 'truth_label', dataset_name, dataset_type)

def get_merged_features(dataset_name, dataset):
    
    index, node_features, edge_features, truth_label = 0, 0, 0, 0

    if dataset_name == 'test':
        for sample in dataset:
            if index == 0:
                node_features = sample.x.clone()
                edge_features = sample.edge_attr.clone()
                truth_label = sample.y.clone()
            else:
                node_features = torch.cat((node_features, sample.x), dim=0)
                edge_features = torch.cat((edge_features, sample.edge_attr), dim=0)
                truth_label = torch.cat((truth_label, sample.y), dim=0)

            index = index + 1

    return node_features, edge_features, truth_label
     
def get_equidistant_bins(data):
    bins = np.linspace(
         torch.min(data).item(), 
         torch.max(data).item(), 
         11)
    
    return bins

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

def get_summary_path():
    # Define the paths to the test and train dataset folders
    summary_options = './LineSegmentClassifier/models/summary'

    # Get the list of files in the test and train dataset folders
    summary_files = os.listdir(summary_options)
    
    # Sort the files for better visibility
    summary_files.sort()
    
    # Create the list of choices for questionary select questions
    summary_choices = [os.path.join(summary_options, file) for file in summary_files]

    # Prompt the user to select test and train dataset paths
    summary_path = questionary.select(
        'Select the JSON summary file:',
        choices=summary_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    return summary_path

def get_csv_paths():
    # Define the paths to the test and train dataset folders
    GNN_csv_options = './csv/models/GNN'
    DNN_csv_options = './csv/models/DNN/LSC_DNN_model_20230722_145012'

    # Get the list of files in the test and train dataset folders
    GNN_files = os.listdir(GNN_csv_options)
    DNN_files = os.listdir(DNN_csv_options)

    # Sort the files for better visibility
    GNN_files.sort()
    DNN_files.sort()

    # Create the list of choices for questionary select questions
    GNN_choices = [os.path.join(GNN_csv_options, file) for file in GNN_files]
    DNN_choices = [os.path.join(DNN_csv_options, file) for file in DNN_files]

    # Prompt the user to select test and train dataset paths
    GNN_path = questionary.select(
        'Select the GNN test CSV file:',
        choices=GNN_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    DNN_path = questionary.select(
        'Select DNN test CSV file:',
        choices=DNN_choices,
        use_shortcuts=True,  # Allow using shortcuts for scrolling
    ).ask()

    return GNN_path, DNN_path

def read_csv_data(file_path, header_truths, header_scores):
    truths, scores = [], []

    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            truths.append(int(row[header_truths]))
            scores.append(float(row[header_scores]))

    return truths, scores

def main():

    # Ask user to select the plot type
    plot_choice = questionary.select(
        "Select the plot type:",
        choices=[
            "Plot dataset features from test/train datasets",
            "Plot loss curve from a model's JSON summary",
            "Plot inferences and predicted scores from a csv file",
        ]
    ).ask()

    if plot_choice == "Plot dataset features from test/train datasets":
        print('PLOTS INITIALIZED')
        TEST_DATASET_PATH, TRAIN_DATASET_PATH = get_dataset_paths()
        test_dataset_name = os.path.splitext(os.path.basename(TEST_DATASET_PATH))[0]
        train_dataset_name = os.path.splitext(os.path.basename(TRAIN_DATASET_PATH))[0]
        test_dataset = torch.load(TEST_DATASET_PATH)
        train_dataset = torch.load(TRAIN_DATASET_PATH)
        
        node_features, edge_features, truth_label = get_merged_features('test', test_dataset)
        generate_feature_plots(node_features, test_dataset_name, 'test')
        generate_feature_plots(edge_features, test_dataset_name, 'test') 
        generate_feature_plots(truth_label, test_dataset_name, 'test')

        node_features, edge_features, truth_label = get_merged_features('train', train_dataset)
        generate_feature_plots(node_features, train_dataset_name, 'train')
        generate_feature_plots(edge_features, train_dataset_name, 'train') 
        generate_feature_plots(truth_label, train_dataset_name, 'train')
        
        print('PLOTS FINALIZED')

    elif plot_choice == "Plot loss curve from a model's JSON summary":
        print('PLOTS INITIALIZED')
        #SUMMARY_PATH = get_summary_path()
        SUMMARY_PATH = './LineSegmentClassifier/models/summary/LSC_DNN_summary_20230722_145012.json'

        with open(SUMMARY_PATH, 'r') as file:
            data = json.load(file)

        model_name = os.path.splitext(os.path.basename(data['model_name']))[0]
        epochs = data['number_epochs']
        train_loss_history = data['train_loss']
        test_loss_history = data['test_loss']

        loss_curve_plot(test_loss_history, train_loss_history, model_name, epochs)

        print('PLOTS FINALIZED')

    elif plot_choice == "Plot inferences and predicted scores from a csv file":
        print('PLOTS INITIALIZED')
        GNN_CSV_PATH, DNN_CSV_PATH = get_csv_paths()
        print(DNN_CSV_PATH)
        model_name = os.path.splitext(os.path.basename(DNN_CSV_PATH))[0]
        filtered_text = re.search(r'inferences_test_(.+)', model_name)
        model_name = filtered_text.group(1)
        truths_GNN, scores_GNN = read_csv_data(GNN_CSV_PATH, 'label', 'score')
        truths_DNN, scores_DNN = read_csv_data(DNN_CSV_PATH, 'truth', 'score')
        roc_curve_plot(truths_GNN, scores_GNN, truths_DNN, scores_DNN, f'GNN vs Big DNN')
        predicted_scores_histogram(scores_DNN, truths_DNN, model_name, 'test')
        print('PLOTS FINALIZED')

if __name__ == '__main__':
    main()