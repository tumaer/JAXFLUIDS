import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_1D_animation

# SETUP SIMULATION
input_manager = InputManager("couette.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, \
forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocity", "pressure"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (1,3)
plot_dict = {
    "density": data_dict["density"],
    "velocityX": data_dict["velocity"][:,0],
    "pressure": data_dict["pressure"]
}
x,y,z = cell_centers

# CREATE ANIMATION
create_1D_animation(
    plot_dict,
    cell_centers,
    times,
    nrows_ncols=nrows_ncols,
    axis="y", axis_values=[0.0, 0.0],
    interval=100)

# CREATE FIGURE
fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(15,3))
ax[0].plot(y, plot_dict["density"][-1,0,:,0])
ax[1].plot(y, plot_dict["velocityX"][-1,0,:,0], label="JXF")
ax[1].plot(y, 0.1*y, color="black", linestyle="--", label="Exact")
ax[2].plot(y, plot_dict["pressure"][-1 ,0,:,0])
ax[1].legend()
ylabels = (r"$\rho$", r"$u$", r"$p$")
for i, axi in enumerate(ax):
    axi.set_xlabel(r"$x$")
    axi.set_ylabel(ylabels[i])
    axi.set_box_aspect(1.0)
plt.savefig("couette.png", dpi=200)
plt.show()
plt.close()
