# Adapted from iphysresearch/S_Dbw_validity_index (https://github.com/iphysresearch/S_Dbw_validity_index)
import pdb
import numpy as np
import pandas as pd
from box import Box


class S_Dbw():
    def __init__(self, data, data_cluster, cluster_centroids_):
        """
        data --> raw data
        data_cluster --> The category that represents each piece of data(the number of category should begin 0)
        cluster_centroids_ --> the center_id of each cluster's center
        """
        self.data = data
        self.data_cluster = data_cluster
        self.cluster_centroids_ = cluster_centroids_

        # Temporary: accumulators for saving step-wise results to database for debugging
        self.acc = Box(stdev=[], density_list=[], Dense_bw=[], Scat=[])

        self.k = cluster_centroids_.shape[0]
        self.stdev = 0
        for i in range(self.k):
            if len(data[self.data_cluster == i]) > 1:
                std_matrix_i = np.std(data[self.data_cluster == i],axis=0)
                self.stdev += np.linalg.norm(std_matrix_i)
                self.acc.stdev.append(self.stdev)
        self.stdev = np.sqrt(self.stdev)/self.k
        self.acc.stdev.append(self.stdev)

    def density(self, density_list=[]):
        """
        compute the density of one or two cluster(depend on density_list)
        """
        density = 0
        if len(density_list) == 2:
            center_v = (self.cluster_centroids_[density_list[0]] + self.cluster_centroids_[density_list[1]])/2
        else:
            center_v = self.cluster_centroids_[density_list[0]]
        for i in density_list:
            temp = self.data[self.data_cluster == i]
            for j in temp:
                if np.linalg.norm(j - center_v) <= self.stdev:
                    density += 1
        return density

    def Dens_bw(self):
        density_list = []
        result = 0
        for i in range(self.k):
            density_list.append(self.density(density_list=[i]))
        self.acc.density_list = density_list
        for i in range(self.k):
            for j in range(self.k):
                if i == j:
                    continue
                denom = max(density_list[i], density_list[j])
                if denom != 0:  # FIXME this shouldn't be happening (right?)
                    result += self.density([i, j])/denom
                self.acc.Dense_bw.append(result)
        res = result/(self.k*(self.k-1))
        self.acc.Dense_bw.append(res)
        return res

    def Scat(self):
        sigma_s = np.var(self.data,axis=0)
        sigma_s_2norm = np.sqrt(np.dot(sigma_s.T,sigma_s))

        sum_sigma_2norm = 0
        for i in range(self.k):
            if len(self.data[self.data_cluster == i]) > 1:
                matrix_data_i = self.data[self.data_cluster == i]
                sigma_i = np.var(matrix_data_i, axis=0)
                sum_sigma_2norm += np.linalg.norm(sigma_i)
                self.acc.Scat.append(sum_sigma_2norm)
        res = sum_sigma_2norm / (sigma_s_2norm * self.k)
        self.acc.Scat.append(res)
        return res

    def S_Dbw_result(self):
        """
        compute the final result
        """
        try:
            return self.Dens_bw() + self.Scat()
        except Exception:
            return np.nan