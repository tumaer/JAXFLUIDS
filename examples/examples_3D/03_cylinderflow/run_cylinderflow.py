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
quantities = ["absolute_vorticity", "schlieren"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
plot_dict = {
    "schlieren": np.clip(data_dict["schlieren"], 1e0, 5e2),
    "absolute_vorticity": np.clip(data_dict["absolute_vorticity"], 1e0, 2e2)
}
nrows_ncols = (2, 1)
create_contourplot(plot_dict, cell_centers, times, nrows_ncols=nrows_ncols)
