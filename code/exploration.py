import io
import os
import pandas as pd
import matplotlib as mpl
import numpy as np
from sklearn.metrics import roc_curve

def find_threshold_for_tpr(tpr_array, threshold_array, tpr_threshold):
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
        return threshold, tpr_value
    else:
        raise ValueError("No threshold value found for the given TPR threshold.")

def get_csv_dataframes(dnn_csv_path, gnn_csv_path):
    # * Load the CSV files into pandas DataFrames
    gnn_df = pd.read_csv(gnn_csv_path)
    dnn_df = pd.read_csv(dnn_csv_path)

    # * Make gnn_df use a filtered version of it (Given the data of CSV_CSV_DATA_PATH)
    gnn_df = gnn_df[['LS_idx', 'label', 'score']]

    # * Rename gnn_df headers the same as the dnn_df columns
    gnn_df.columns = dnn_df.columns

    # * Re-assign 'idx' column to have values consistent to their respective dataframes
    gnn_df['idx'] = np.arange(0, len(gnn_df))
    dnn_df['idx'] = np.arange(0, len(dnn_df))

    return gnn_df, dnn_df

def get_filtered_label_dataframes(dnn_df, dnn_df_label0, dnn_df_label1, 
                                  gnn_df, gnn_df_label0, gnn_df_label1, x_thresh, y_thresh):
    dnn_all_df = dnn_df[dnn_df['score'] > x_thresh]
    gnn_all_df = gnn_df[gnn_df['score'] > y_thresh]
    dnn_fil_df_lbl0 = dnn_df_label0[dnn_df_label0['score'] > x_thresh]
    gnn_fil_df_lbl0 = gnn_df_label0[gnn_df_label0['score'] > y_thresh]
    dnn_fil_df_lbl1 = dnn_df_label1[dnn_df_label1['score'] > x_thresh]
    gnn_fil_df_lbl1 = gnn_df_label1[gnn_df_label1['score'] > y_thresh]

    return dnn_all_df, gnn_all_df, dnn_fil_df_lbl0, gnn_fil_df_lbl0, dnn_fil_df_lbl1, gnn_fil_df_lbl1

def get_roc_curve_values(dnn_df, gnn_df):
    fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(dnn_df['truth'], dnn_df['score'])
    fpr_gnn, tpr_gnn, thresholds_gnn = roc_curve(gnn_df['truth'], gnn_df['score'])

    return tpr_dnn, tpr_gnn, thresholds_dnn, thresholds_gnn

def get_tpr_sample_values(type, dnn_df, dnn_df_label0, dnn_df_label1, 
                          gnn_df, gnn_df_label0, gnn_df_label1, x_thresh, y_thresh):
    dnn_all_df, gnn_all_df, dnn_fil_df_lbl0, gnn_fil_df_lbl0, dnn_fil_df_lbl1, gnn_fil_df_lbl1 = get_filtered_label_dataframes(
        dnn_df, dnn_df_label0, dnn_df_label1, gnn_df,gnn_df_label0, gnn_df_label1, x_thresh, y_thresh)
    
    if type == 'all':
        idx_in_both = np.isin(dnn_all_df['idx'], gnn_all_df['idx'])
        number_dnn = len(dnn_all_df)
        number_gnn = len(gnn_all_df)
        number_dnngnn = np.sum(idx_in_both)

    if type == 'real':
        idx_in_both = np.isin(dnn_fil_df_lbl0['idx'], gnn_fil_df_lbl0['idx'])
        number_dnn = len(dnn_fil_df_lbl0)
        number_gnn = len(gnn_fil_df_lbl0)
        number_dnngnn = np.sum(idx_in_both)

    if type == 'fake':
        idx_in_both = np.isin(dnn_fil_df_lbl1['idx'], gnn_fil_df_lbl1['idx'])
        number_dnn = len(dnn_fil_df_lbl1)
        number_gnn = len(gnn_fil_df_lbl1)
        number_dnngnn = np.sum(idx_in_both)

    return number_dnn, number_gnn, number_dnngnn

def main():
    DNN_CSV_PATH = './csv/models/DNN/LSC_DNN_model_20230722_145012/inferences_test_LSC_DNN_model_20230722_145012.csv'
    GNN_CSV_PATH = './csv/models/GNN/ChangGNN_MDnodes_LSedges_modelChangNet_nhidden2_hiddensize200_lrStepLR0.005_epoch50_test_inferences.csv'
    TPR_VALUES = [0.95, 0.99]

    # * Load the CSV files into pandas DataFrames
    gnn_df, dnn_df = get_csv_dataframes(DNN_CSV_PATH, GNN_CSV_PATH)

    # * Get threshold values
    tpr_dnn, tpr_gnn, thresholds_dnn, thresholds_gnn = get_roc_curve_values(dnn_df, gnn_df)
    
    # * For isFake==0 ... (LS is NOT fake)
    dnn_df_label0 = dnn_df[dnn_df['truth'] == 0]
    gnn_df_label0 = gnn_df[gnn_df['truth'] == 0]

    # * For isFake==1 ... (LS IS fake)
    dnn_df_label1 = dnn_df[dnn_df['truth'] == 1]
    gnn_df_label1 = gnn_df[gnn_df['truth'] == 1]

    '''
        For any label
            TPR_val = 0.99, 0.95
            Find largest X s.t. DNN > X gives TPR = TPR_val
                         Y s.t. GNN > Y gives TPR = TPR_val
            
            DNN_X = # of DNN > X
            GNN_Y = # of GNN > Y
            # of DNN_X AND GNN_Y
    '''

    # * Filter the DataFrames based on the TPR threshold
    x_threshold_dnn_95, tpr_dnn_95 = find_threshold_for_tpr(tpr_dnn, thresholds_dnn, tpr_threshold=TPR_VALUES[0])
    y_threshold_gnn_95, tpr_gnn_95 = find_threshold_for_tpr(tpr_gnn, thresholds_gnn, tpr_threshold=TPR_VALUES[0])
    x_threshold_dnn_99, tpr_dnn_99 = find_threshold_for_tpr(tpr_dnn, thresholds_dnn, tpr_threshold=TPR_VALUES[1])
    y_threshold_gnn_99, tpr_gnn_99 = find_threshold_for_tpr(tpr_gnn, thresholds_gnn, tpr_threshold=TPR_VALUES[1])
    
        # * for TPR > 0.95
    # * Calculate the total number of both labels with DNN > X & GNN > Y
    number_dnn_all_95, number_gnn_all_95 , number_dnngnn_all_95 = get_tpr_sample_values(
        'all', dnn_df, dnn_df_label0, dnn_df_label1, 
        gnn_df, gnn_df_label0, gnn_df_label1, 
        x_threshold_dnn_95, y_threshold_gnn_95)
    
    # * Calculate the total number of isFake==0 with DNN > X & GNN > Y
    number_dnn_label0_95, number_gnn_label0_95, number_dnngnn_label0_95 = get_tpr_sample_values(
        'fake', dnn_df, dnn_df_label0, dnn_df_label1, 
        gnn_df, gnn_df_label0, gnn_df_label1, 
        x_threshold_dnn_95, y_threshold_gnn_95)
    
    # * Calculate the total number of isFake==1 with DNN > X & GNN > Y
    number_dnn_label1_95, number_gnn_label1_95, number_dnngnn_label1_95 = get_tpr_sample_values(
        'real', dnn_df, dnn_df_label0, dnn_df_label1, 
        gnn_df, gnn_df_label0, gnn_df_label1, 
        x_threshold_dnn_95, y_threshold_gnn_95)

        # * for TPR > 0.99
    # * Calculate the total numbet of both labels with DNN > X & GNN > Y
    number_dnn_all_99, number_gnn_all_99, number_dnngnn_all_99 = get_tpr_sample_values(
        'all', dnn_df, dnn_df_label0, dnn_df_label1, 
        gnn_df, gnn_df_label0, gnn_df_label1, 
        x_threshold_dnn_99, y_threshold_gnn_99)
    
    # * Calculate the total number of isFake==0 with DNN > X & GNN > Y
    number_dnn_label0_99, number_gnn_label0_99, number_dnngnn_label0_99 = get_tpr_sample_values(
        'fake', dnn_df, dnn_df_label0, dnn_df_label1, 
        gnn_df, gnn_df_label0, gnn_df_label1, 
        x_threshold_dnn_99, y_threshold_gnn_99)
    
    # * Calculate the total number of isFake==1 with DNN > X & GNN > Y
    number_dnn_label1_99, number_gnn_label1_99, number_dnngnn_label1_99 = get_tpr_sample_values(
        'real', dnn_df, dnn_df_label0, dnn_df_label1, 
        gnn_df, gnn_df_label0, gnn_df_label1, 
        x_threshold_dnn_99, y_threshold_gnn_99)
    
    print()

    print('\tThreshold value for TPR > ', TPR_VALUES[0])
    print(f"Threshold for TPR > {TPR_VALUES[0]} (DNN): {x_threshold_dnn_95}")
    print(f"Threshold for TPR > {TPR_VALUES[0]} (GNN): {y_threshold_gnn_95}")
    print(f'\nall:\tDNN: {number_dnn_all_95},\tGNN: {number_gnn_all_95},\tBoth: {number_dnngnn_all_95}')
    print(f'Real:\tDNN: {number_dnn_label0_95},\tGNN: {number_gnn_label0_95},\tBoth: {number_dnngnn_label0_95}')
    print(f'Fake:\tDNN: {number_dnn_label1_95},\tGNN: {number_gnn_label1_95},\tBoth: {number_dnngnn_label1_95}')
    
    print()

    print('\tThreshold value for TPR > ', TPR_VALUES[1])
    print(f"Threshold for TPR > {TPR_VALUES[1]} (DNN): {x_threshold_dnn_99}")
    print(f"Threshold for TPR > {TPR_VALUES[1]} (GN): {y_threshold_gnn_99}")
    print(f'\nall:\tDNN: {number_dnn_all_99},\tGNN: {number_gnn_all_99},\tBoth: {number_dnngnn_all_99}')
    print(f'Real:\tDNN: {number_dnn_label0_99},\tGNN: {number_gnn_label0_99},\tBoth: {number_dnngnn_label0_99}')
    print(f'Fake:\tDNN: {number_dnn_label1_99},\tGNN: {number_gnn_label1_99},\tBoth: {number_dnngnn_label1_99}')

    print()

if __name__ == '__main__':
    main()
