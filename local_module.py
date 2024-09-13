# Import necessary modules
import os
import torch
from torch.utils.data import Dataset
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import re
import numpy as np
import joblib


hdf5plugin  # It was not called to be used, but I added it to avoid the error


# def dir_file_path(root_dir=None, labels=None):

#     # ! Need to be fixed in the case of file name, path, and labels may not in the same order

#     """
#     Finds all files containing 'data' with a .h5 extension in the specified directory and its subdirectories.

#     Args:
#         root_dir (str): The root directory to search.

#     Returns:
#         tuple: A tuple of two lists: (file_names, file_paths)
#     """
#     file_info = []
#     labels_paths = []

#     def find_files(root_dir):
#         for root, _, files in os.walk(root_dir):
#             for file in files:
#                 if file.endswith('.h5'):
#                     if 'siO2_PPG_fastshutter' and 'data' in file:
#                         file_path = os.path.join(root, file)
#                         file_info.append((file, file_path))
#                 if file.endswith('.joblib'):
#                     if 'siO2_PPG_fastshutter' and 'data' in file:
#                             file_path = os.path.join(root, file)
#                             file_info.append((file, file_path))

#             if 'Labels' in root:
#                 for file in files:
#                     if str(labels) in file and file.endswith('.joblib'):
#                         file_path = os.path.join(root, file)
#                         labels_paths.append(file_path)

#     if root_dir:
#         find_files(root_dir)
#     else:
#         find_files(os.getcwd())

#     file_info.sort()

#     file_names, file_paths = zip(*file_info)

#     if labels:
#         labels_paths.sort()
#         return file_names, file_paths, labels_paths
#     else:
#         return file_names, file_paths


def dir_file_path(root_dir=None):
    """
    Finds HDF5 and joblib files matching a pattern in the specified directory and its subdirectories.

    Args:
        root_dir (str, optional): The root directory to search. Defaults to the current working directory.
        labels (str, optional): A regular expression pattern to match in label file names.

    Returns:
        tuple: A tuple of three lists: (file_names, file_paths, label_paths)
    """

    file_paths = []

    def find_files(root_dir):
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.h5') or file.endswith('.joblib'):
                    # Use regular expression for more flexible matching
                    if re.search(r'siO2_PPG_fastshutter.*data', file):
                        file_path = os.path.join(root, file)
                        file_paths.append(file_path)

    if root_dir:
        find_files(root_dir)
    else:
        try:
            find_files(os.getcwd())
        except FileNotFoundError:
            print('Invalid root directory.')

    # Sort file paths directly
    file_paths.sort()
    return [os.path.basename(path) for path in file_paths], file_paths


class H5Dataset(Dataset):
    """
    A class for loading and accessing data from HDF5 files.

    Args:
        h5_file_path (str): Path to the HDF5 file.
        group (str): Group name within the HDF5 file containing the data.
        transform (callable, optional): A transformation function to apply to the data. Defaults to None.
        labels_path (str, optional): Path to a file containing labels for the data. Defaults to None.
    """

    def __init__(self, h5_file_path, group, transform=None, labels_path=None):
        """
        Initializes the H5Dataset class.

        Opens the HDF5 file and stores references to the data group and any provided transform or label path.
        """
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.group = group
        self.data = self.h5_file[group]
        self.transform = transform
        self.labels_path = labels_path

    def __len__(self):
        """
        Returns the length of the dataset (number of data points).

        Uses the length of the data group in the HDF5 file.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a data point and its corresponding label (if available) at a given index.

        Args:
            index (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the data point (torch.Tensor) and its label (optional)
        """
        data = torch.from_numpy(self.data[index])
        data = torch.as_tensor(data, dtype=torch.float)
        if self.transform:
            data = self.transform(data)

        if self.labels_path:
            try:
                labels = joblib.load(self.labels_path)
                labels = labels[index]
                return data, labels
            except FileNotFoundError:
                print(f'Labels file not found: {self.labels_path}')
                return data
        else:
            return data

    def shape(self):
        """
        Returns the shape of the data in the HDF5 file as a tuple of integers.
        """
        return self.data.shape


class MulTransform:
    """
    A transformation class that multiplies an input tensor by a mask factor.

    Args:
        h5_mask_file (str): Path to the HDF5 file containing the mask.
        mask_group (str): Group name within the HDF5 file where the mask is located.
    """

    def __init__(self, h5_mask_file, mask_group):
        """
        Initializes the MulTransform class.

        Loads the mask from the specified HDF5 file and group, and converts it to a PyTorch tensor.
        """
        self.mask = h5py.File(h5_mask_file, 'r')
        self.mask_group = self.mask[mask_group]
        self.factor = torch.from_numpy(self.mask_group[...])

    def __call__(self, input_tensor):
        """
        Applies the transformation to an input tensor.

        Multiplies the input tensor element-wise by the stored mask factor.

        Args:
            input_tensor (torch.Tensor): The input tensor to be transformed.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        return torch.mul(input_tensor, self.factor)


def sub_plot(paths, group, nrows, ncols, figsize=None, titles=None, xlabel=None, ylabel=None, hspace=None):
    """
    Creates a subplot figure and plots average value trends from multiple HDF5 files.

    Args:
        paths (list[str]): List of paths to HDF5 files containing the data.
        group (str): Group name within the HDF5 files where the data is located.
        nrows (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        figsize (tuple, optional): Desired figure size. Defaults to None.
        titles (list[str], optional): List of titles for each subplot. Defaults to None.
            The length of the list should match the number of paths.
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
        hspace (float, optional): Amount of vertical space between subplots. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The created figure with subplots.
    """

    if figsize:
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows, ncols)

    for path, ax in zip(paths, axs.ravel()):
        # Open and access data from HDF5 file
        f = h5py.File(path, 'r')
        dataset = f[group]
        data = torch.from_numpy(dataset[:])

        # Plot data with basic formatting
        ax.plot(data, '.', markersize=1)

        # Set title if titles exist and not None for current path
        if titles:
            try:
                title = titles[paths.index(path)]  # Access title based on path index
                if title:
                    ax.set_title(title)
            except IndexError:  # Handle cases where titles list is shorter than paths
                print(f'Warning: Not enough titles provided for path {path}')

        # Adjust spacing if hspace is provided
        if hspace:
            plt.subplots_adjust(hspace=hspace)

    fig.suptitle('Average value trends')


def rename(paths):
    """
    Renames files in a list of paths by removing everything up to the last '--'.

    Args:
        paths (list[str]): List of file paths to rename.
    """
    for path in paths:
        # Extract directory and filename
        the_dir = os.path.dirname(path)

        # Create new filename by removing everything before the last '--'
        newname = re.sub('.*--', '', os.path.basename(path))

        # Construct new path
        newpath = os.path.join(the_dir, newname)

        # Rename file if necessary, checking for conflicts
        if path != newpath:
            if os.path.exists(newpath):
                print('Warning: cannot rename into {}'.format(newpath))
            else:
                os.rename(path, newpath)


# def comparing(data1, data2):
#     data1, data2 = np.array(data1), np.array(data2)
#     diff = (data1 - data2)


def vector_distance(vector1, vector2):

    """
    Find the distance between 2 vectors on the vector space to find by using normalization
    Compare the difference of 2 vectors and return as a percentage of difference

    Return:
        Float: percantage of difference
    """

    euclidean_dist = np.linalg.norm(np.array(vector1) - np.array(vector2))
    normalized_euclidean_sim = 1 / (1 + euclidean_dist)
    percentage_of_diff = np.abs(normalized_euclidean_sim - 1)/1 * 100
    return percentage_of_diff
    # return normalized_euclidean_sim


def norm(input_tensor):
    """
    Normalizes an input tensor to the range [0, 1] using min-max scaling.

    Args:
        input_tensor (torch.Tensor): The input tensor to be normalized.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    max_val = torch.max(input_tensor)
    min_val = torch.min(input_tensor)
    return (input_tensor - min_val)/(max_val - min_val)
