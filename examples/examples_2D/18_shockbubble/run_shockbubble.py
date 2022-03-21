import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

# SETUP SIMULATION
input_reader = InputReader("shockbubble.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["real_pressure", "schlieren", "mach_number"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (3,1)
plot_dict = {
    "schlieren"     : np.clip(data_dict["schlieren"], 1e-1, 5e1),
    "pressure"      : data_dict["real_pressure"],
    "mach_number"   : data_dict["mach_number"]
    }
create_contourplot(plot_dict, cell_centers, times, nrows_ncols=nrows_ncols, plane="xy", interval=100)

fig, ax = plt.subplots(nrows=3)
ax[0].imshow(np.transpose(plot_dict["schlieren"][-1,:,:,0]), origin="lower")
ax[1].imshow(np.transpose(plot_dict["pressure"][-1,:,:,0]), origin="lower")
ax[2].imshow(np.transpose(plot_dict["mach_number"][-1,:,:,0]), origin="lower")
plt.show()