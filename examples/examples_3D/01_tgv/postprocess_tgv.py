import json
import os

import matplotlib.pyplot as plt
import numpy as np

from jaxfluids.post_process.post_process_utils import load_data, generate_png

paths = []
paths.append("./results/tgv-3/domain")

deformations_steady = []

save_path = "/home/dbezgin/Desktop"

for path in paths:
    files = []
    for file in os.listdir(path):
        if file.endswith("h5"):
            if "nan" in file:
                continue 
            files.append(file)
    no_files = len(files)

    velocity_array  = []
    vorticity_array = []
    for ii in range(no_files):
        quantities = ["vorticity", "velocity"]
        cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, start=ii, end=ii+1, N=1)
        X, Y    = np.meshgrid(cell_centers[0], cell_centers[1], indexing="ij")

        velocity      = data_dict["velocity"]
        vorticity     = data_dict["vorticity"]
        velocity_mag  = np.sqrt(velocity[:,0]**2 + velocity[:,1]**2 + velocity[:,2]**2)
        vorticity_mag = np.sqrt(vorticity[:,0]**2 + vorticity[:,1]**2 + vorticity[:,2]**2)

        velocity_array.append(velocity_mag[0,:,:,31:32])
        vorticity_array.append(vorticity_mag[0,:,:,31:32])

    velocity_array = np.stack(velocity_array, axis=0)
    vorticity_array = np.stack(vorticity_array, axis=0)
    print(velocity_array.shape)

    # MAKE PNGs
    data_to_plot = {
        "velocity": velocity_array, 
        "vorticity": vorticity_array, 
        }

    nrows_ncols = (1,2)
    path = "./results/images"
    generate_png(X, Y, data_to_plot, path, nrows_ncols=nrows_ncols, dpi=200)