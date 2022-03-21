import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

# SETUP SIMULATION
input_reader = InputReader("shockvortex.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "pressure", "mach_number"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (1,3)
plot_dict = {
    "density"   : data_dict["density"],
    "pressure"  : data_dict["pressure"],
    "mach_number": np.clip(data_dict["mach_number"], 0.0, 4.0),
    }
create_contourplot(plot_dict, cell_centers, times, nrows_ncols=nrows_ncols, plane="xy", interval=100)

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(np.transpose(plot_dict["density"][-1,:,:,0]), origin="lower")
ax[1].imshow(np.transpose(plot_dict["pressure"][-1,:,:,0]), origin="lower")
ax[2].imshow(np.transpose(plot_dict["mach_number"][-1,:,:,0]), origin="lower")
plt.show()