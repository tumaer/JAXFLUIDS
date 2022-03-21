import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

# SETUP SIMULATION
input_reader = InputReader("movingsolids.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["schlieren", "mask_real", "temperature", "levelset"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (1,2)
plot_dict = {
    "schlieren"     : np.clip(data_dict["schlieren"] * data_dict["mask_real"], 1e0, 2e2) ,
    "temperature"   : data_dict["temperature"] * data_dict["mask_real"],
    }
create_contourplot(plot_dict, cell_centers, times, levelset=data_dict["levelset"], nrows_ncols=nrows_ncols, plane="xy", interval=100)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.transpose(plot_dict["schlieren"][-1,:,:,0]), origin="lower")
ax[1].imshow(np.transpose(plot_dict["temperature"][-1,:,:,0]), origin="lower")
plt.show()