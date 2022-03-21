import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

# SETUP SIMULATION
input_reader = InputReader("sheardrop.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["real_pressure", "real_density", "real_velocity", "levelset"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (2,2)
plot_dict = {
    "density"   : data_dict["real_density"],
    "pressure"  : data_dict["real_pressure"],
    "velocityX" : data_dict["real_velocity"][:,0],
    "velocityY" : data_dict["real_velocity"][:,1],
    }
create_contourplot(plot_dict, cell_centers, times, nrows_ncols=nrows_ncols, plane="xy", interval=100)

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0,0].imshow(np.transpose(plot_dict["density"][-1,:,:,0]), origin="lower")
ax[0,1].imshow(np.transpose(plot_dict["pressure"][-1,:,:,0]), origin="lower")
ax[1,0].imshow(np.transpose(plot_dict["velocityX"][-1,:,:,0]), origin="lower")
ax[1,1].imshow(np.transpose(plot_dict["velocityY"][-1,:,:,0]), origin="lower")
plt.show()