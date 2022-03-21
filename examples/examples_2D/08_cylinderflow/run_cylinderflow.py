import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

# SETUP SIMULATION
input_reader = InputReader("cylinderflow.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["schlieren", "mach_number", "levelset", "mask_real"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (1,2)
plot_dict = {
    "mach_number": np.clip(data_dict["mach_number"] * data_dict["mask_real"], 0.0, 3.0),
    "schlieren"  : np.clip(data_dict["schlieren"] * data_dict["mask_real"], 1e0, 5e2)}
create_contourplot(plot_dict, cell_centers, times, nrows_ncols=nrows_ncols, plane="xy", interval=100)

fig, ax = plt.subplots()
ax.imshow(np.transpose(plot_dict["mach_number"][-1,:,:,0]), origin="lower")
plt.show()