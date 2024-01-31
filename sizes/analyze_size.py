import argparse
import pandas as pd
import os
from itertools import combinations
from tqdm import tqdm
import ast

import nibabel as nib
import numpy as np

def get_val_gt_size_from_nifti(nifti_file):
    # Return the number of voxels for each class in the NIfTI file
    nifti_data = nib.load(nifti_file)
    data = nifti_data.get_fdata()

    return [np.sum(data == i) for i in range(5)]

def find_combination(val_gt_size):
    #  Return the combination of classes that has at least one row in the input file
    return tuple(i for i, count in enumerate(val_gt_size) if count > 0)



def calculate_average_for_column(df, column, combination):
    """ Calculate the average value for a specific column and combination of classes. """
    filtered_df = df[df['val_gt_size'].apply(lambda x: 
                      all((ast.literal_eval(x)[i] > 0 if i in combination else ast.literal_eval(x)[i] == 0) for i in range(5)))]
    

    if not filtered_df.empty:
        # Convert the string representation of list to actual list and then calculate the mean
        filtered_df[column] = filtered_df[column].apply(ast.literal_eval)
        return [round(sum(values) / len(values)) for values in zip(*filtered_df[column])]
        
    else:
        return 'No rows with classes {}'.format(combination)


def calculate_dumbpredwtags_average_if_no_combination(df, combination):

    class_averages = []
    for i in range(5):
        if i in combination:
            class_values = df['dumbpredwtags'].apply(lambda x: ast.literal_eval(x)[i])
            positive_values = [val for val in class_values if val > 0]
            if positive_values:
                class_average = round(sum(positive_values) / len(positive_values))
            else:
                class_average = 0
        else:
            class_average = 0
        class_averages.append(class_average)
    return class_averages


def create_new_file(nii_file, output_file, avg_dumbpredwtags, avg_dumbprednotags):
    """ Create a new file with the average values for dumbpredwtags and dumbprednotags. """
    new_row = {'NIfTI File': nii_file, 'dumbpredwtags': avg_dumbpredwtags, 'dumbprednotags': avg_dumbprednotags}
    df = pd.DataFrame([new_row])
    df.to_csv(output_file, sep=';', index=False)

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Create CSV files from NIfTI files with calculated averages.')
    parser.add_argument('-i', '--input', type=str, default='whs.csv', help='Input CSV file path')
    parser.add_argument('-g', '--gtdirs', nargs='+', help='Directories with NIfTI files')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file path')
    args = parser.parse_args()

    if not args.output:
        args.output = args.input.replace('.csv', '') + '_mr.csv'

    # Reading the input file
    df = pd.read_csv(args.input, delimiter=';')

    # Calculating averages for dumbpredwtags and dumbprednotags

    averages = {}
    for r in range(1, 6):
        for combination in combinations(range(5), r):
            avg_dumbpredwtags = calculate_average_for_column(df, 'dumbpredwtags', combination)
            avg_dumbprednotags = calculate_average_for_column(df, 'dumbprednotags', combination)
            
            # If the average is a list, then it means that there are rows with the given combination
            if isinstance(avg_dumbpredwtags, list) and isinstance(avg_dumbprednotags, list):
                averages[combination] = {
                    'dumbpredwtags': avg_dumbpredwtags,
                    'dumbprednotags': avg_dumbprednotags
                }
    _ = [print('Combination: {}, dumbpredwtags: {}, dumbprednotags: {}'.format(combination, averages[combination]['dumbpredwtags'], averages[combination]['dumbprednotags'])) for combination in averages]


    if args.gtdirs and args.output:
        # Get the average value for dumbprednotags for all the rows
        common_dumbprednotags = df['dumbprednotags'].iloc[0]
        all_data = []
        for gtdir in args.gtdirs:
            if os.path.exists(gtdir):
                for nii_file in tqdm(os.listdir(gtdir)):
                    if nii_file.endswith('.nii'):
                        val_gt_size = get_val_gt_size_from_nifti(os.path.join(gtdir, nii_file))
                        combination = find_combination(val_gt_size)
                        if combination in averages:
                            avg_wtags = averages[combination]['dumbpredwtags']
                        else:
                            # If the combination is not in the averages dictionary, then it means that there are no rows with that combination
                            avg_wtags = calculate_dumbpredwtags_average_if_no_combination(df, combination)
                        all_data.append({
                            'val_ids': nii_file,
                            'val_gt_size': val_gt_size,
                            'dumbpredwtags': avg_wtags,
                            'dumbprednotags': common_dumbprednotags
                        })

        # Create a new CSV file with the averages
        output_df = pd.DataFrame(all_data)
        output_df.to_csv(args.output, sep=';', index=False)

if __name__ == "__main__":
    main()
