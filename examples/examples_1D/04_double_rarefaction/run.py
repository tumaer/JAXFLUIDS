import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_1D_animation, create_1D_figure

# SETUP SIMULATION
input_manager = InputManager("double_rarefaction.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, \
forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocity", "pressure", "temperature"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
plot_dict = {
    "density": data_dict["density"], 
    "velocityX": data_dict["velocity"][:,0],
    "pressure": data_dict["pressure"],
    "temperature": data_dict["temperature"]
}
nrows_ncols = (1,4)

# CREATE ANIMATION
create_1D_animation(
    plot_dict, 
    cell_centers, 
    times, 
    nrows_ncols=nrows_ncols, 
    interval=200)

# CREATE FIGURE
create_1D_figure(
    plot_dict, 
    cell_centers=cell_centers, 
    nrows_ncols=nrows_ncols,
    axis="x", axis_values=(0,0),
    save_fig="double_rarefaction.png")