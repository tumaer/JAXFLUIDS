import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

# SETUP SIMULATION
input_reader = InputReader("HIT_forced.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "pressure", "temperature", "velocityX", "velocityY", "velocityZ"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
TKE      = np.mean(0.5 * (data_dict["velocityX"]**2 + data_dict["velocityY"]**2 + data_dict["velocityZ"]**2), axis=(1,2,3))
dens_rms = np.std(data_dict["density"], axis=(1,2,3))

# PLOTTING
fig, ax = plt.subplots()
ax.plot(times, dens_rms / dens_rms[0], label=r"$\rho_{rms}$")
ax.plot(times, TKE / TKE[0], label=r"$TKE$")
ax.legend()
plt.show()