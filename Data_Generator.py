import torch
from torch.utils.data import Dataset
import numpy as np

from read_data import load_rotor_data_from_csv


class TSPDataset(Dataset):
    """
    Random TSP dataset with one-dimensional cities and random solutions.
    """

    def __init__(self, data_size, seq_len, solver=None, solve=True):
        """
        Initialize the dataset with the given origin data.

        :param origin_data: List of dictionaries containing 'blade_mass_list' and 'feasible_permutation'.
        """

        self.data = self._process_data()
        self.seq_len = seq_len
        self.data_size = len(self.data['blade_mass_list'])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # Get the blade mass list (cities) and the feasible permutation (optimal path)
        # blade_mass_list = np.array(self.data['blade_mass_list'][idx])
        # feasible_permutation = np.array(self.data['feasible_permutation'][idx])

        # Convert them to torch tensors
        tensor_points = torch.from_numpy(self.data['blade_mass_list'][idx]).float()
        tensor_solution = torch.from_numpy(self.data['feasible_permutation'][idx]).long()

        sample = {'Points': tensor_points, 'Solution': tensor_solution}
        return sample

    # def __getitem__(self, idx):
    #     # Get the blade mass list (cities) and the feasible permutation (optimal path)
    #     blade_mass_list = np.array(self.data[idx]['blade_mass_list'])
    #     feasible_permutation = np.array(self.data[idx]['feasible_permutation'])
    #
    #     # Convert them to torch tensors
    #     tensor_points = torch.from_numpy(blade_mass_list).float()
    #     tensor_solution = torch.from_numpy(feasible_permutation).long()
    #
    #     # Return the Points as a list of blade mass lists and the Solution as a list of feasible permutations
    #     sample = {
    #         'Points': tensor_points,  # List of blade mass lists
    #         'Solution': tensor_solution  # List of feasible permutations
    #     }
    #     return sample

    def _process_data(self):
        """
        Process the input origin data into two lists:
        1. A list of all blade mass lists.
        2. A list of all feasible permutations.

        :param origin_data: List of dictionaries containing 'blade_mass_list' and 'feasible_permutation'.
        :return: A dictionary with two keys: 'blade_mass_list' and 'feasible_permutation'
        """
        csv_path = "rotor_data.csv"
        origin_data = load_rotor_data_from_csv(csv_path)
        blade_mass_list = []
        feasible_permutation = []

        for entry in origin_data:
            list1 = [[i] for i in entry['blade_mass_list']]
            # list2 = [[i] for i in entry['feasible_permutation']]
            blade_mass_list.append(np.array(list1))
            feasible_permutation.append(np.array(entry['feasible_permutation']))

        return {'blade_mass_list': blade_mass_list, 'feasible_permutation': feasible_permutation}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import itertools
# from tqdm import tqdm
#
#
# def random_solution(seq_len):
#     """
#     Generate a random solution for TSP.
#     Returns a random permutation of city indices.
#
#     :param seq_len: Number of cities in the TSP problem
#     :return: Random permutation of city indices
#     """
#     return np.random.permutation(seq_len)
#
#
# class TSPDataset(Dataset):
#     """
#     Random TSP dataset with one-dimensional cities and random solutions.
#     """
#
#     def __init__(self, data_size, seq_len, solver=None, solve=True):
#         self.data_size = data_size
#         self.seq_len = seq_len
#         self.solve = solve
#         self.solver = solver
#         self.data = self._generate_data()
#
#     def __len__(self):
#         return self.data_size
#
#     def __getitem__(self, idx):
#         tensor = torch.from_numpy(self.data['Points_List'][idx]).float()
#         solution = torch.from_numpy(self.data['Solutions'][idx]).long() if self.solve else None
#
#         sample = {'Points':tensor, 'Solution':solution}
#
#         return sample
#
#     def _generate_data(self):
#         """
#         :return: Set of points_list and their random solution orders.
#         """
#         points_list = []
#         solutions = []
#         data_iter = tqdm(range(self.data_size), unit='data')
#         for i, _ in enumerate(data_iter):
#             data_iter.set_description('Data points %i/%i' % (i+1, self.data_size))
#             points_list.append(np.random.random(self.seq_len))  # Random 1D city coordinates
#
#         solutions_iter = tqdm(points_list, unit='solve')
#         if self.solve:
#             for i, points in enumerate(solutions_iter):
#                 solutions_iter.set_description('Solved %i/%i' % (i + 1, len(points_list)))
#                 # Random solution instead of optimal solution
#                 solutions.append(random_solution(self.seq_len))  # Random solution
#         else:
#             solutions = None
#
#         return {'Points_List': points_list, 'Solutions': solutions}
