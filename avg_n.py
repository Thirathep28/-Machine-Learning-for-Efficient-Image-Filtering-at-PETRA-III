import torch
import torchvision
from torch.utils.data import DataLoader
from local_module import dir_file_path, H5Dataset, MulTransform
import h5py
import hdf5plugin

hdf5plugin  # It was not called to be used, but I added it to avoid the error

names, paths = dir_file_path()
print(len(paths))


transform = torchvision.transforms.Compose([MulTransform(
    '/home/phiankha/Desktop/Summer-Student-2024-Project/Preparation/data/masks/Det_029_mask.h5', 'blemish')])

for name, path in zip(names, paths):
    print(name)

    avgr = []

    dataset = H5Dataset(path, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=6, shuffle=False, num_workers=6)

    for images in dataloader:
        avg = torch.mean(images.float(), (1, 2))
        avgr.append(avg)

    avgr = torch.cat(avgr)
    print(type(avgr), avgr.size())

    hf = h5py.File(
        '/home/phiankha/Desktop/Summer-Student-2024-Project/avg-value/G24_5k_00006_alt/{}.h5'.format(name), 'w')
    hf.create_dataset('average-values', data=avgr)
    hf.close
