import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

# SETUP SIMULATION
input_manager = InputManager("lid_driven_cavity.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, \
forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["velocity"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, step=5)

# PLOT
nrows_ncols = (1,2)
plot_dict = {
    "velocityX": data_dict["velocity"][:,0],
    "velocityY": data_dict["velocity"][:,1],
}
x,y,z = cell_centers

# CREATE ANIMATION
create_2D_animation(
    plot_dict,
    cell_centers, 
    times, 
    nrows_ncols=nrows_ncols,
    plane="xy", cmap="seismic",
    interval=100)

# CREATE FIGURE
velX = data_dict["velocity"][-1,0,:,:,0]
velY = data_dict["velocity"][-1,1,:,:,0]
vel_abs = np.sqrt(velX**2 + velY**2)

N = len(x)
u_wall = 0.5
reference_data = np.loadtxt("reference_data.txt")

fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(15,4))
ax[0].streamplot(x, y, velX.T, velY.T, color=vel_abs.T, arrowsize=0)
ax[1].plot(reference_data[:,0], reference_data[:,2], linestyle="None", marker=".", label="Reference")
ax[1].plot(y, velX[N//2,:] / u_wall, label="JXF")
ax[2].plot(reference_data[:,1], reference_data[:,3], linestyle="None", marker=".", label="Reference")
ax[2].plot(x, velY[:,N//2] / u_wall, label="JXF")
# create_2D_figure(plot_dict, cell_centers=cell_centers, plane="xy", plane_value=0.0)
ax[1].set_xlabel(r"$y$")
ax[1].set_ylabel(r"$u / u_w$")
ax[2].set_xlabel(r"$x$")
ax[2].set_ylabel(r"$v / u_w$")
ax[2].legend()
for axi in ax:
    axi.set_box_aspect(1.0)
plt.savefig("lid_driven_cavity.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()