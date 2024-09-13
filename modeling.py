from local_module import dir_file_path, norm, H5Dataset
from sklearn.mixture import GaussianMixture

names, paths = dir_file_path()

print(len(paths))


dataset = H5Dataset(paths[0], 'entry/data/data')
data = dataset[:].reshape(-1, 1)
data_norm = norm(data)

gm = GaussianMixture(n_components=2).fit(norm(data_norm))

print(gm.means_)