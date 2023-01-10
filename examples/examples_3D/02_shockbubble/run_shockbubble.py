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
quantities = ["real_density", "real_pressure", "schlieren", "levelset"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
plot_dict = {
    "density"  : data_dict["real_density"],
    "schlieren": np.clip(data_dict["schlieren"], 1e-1, 5e2)
    }
nrows_ncols = (1,2)
create_contourplot(plot_dict, cell_centers, times, nrows_ncols=nrows_ncols, plane="xy", plane_value=0, interval=100)