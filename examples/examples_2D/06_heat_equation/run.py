import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

# SETUP SIMULATION
input_manager = InputManager("heat_equation.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
simulation_buffers, time_control_variables, \
forcing_parameters = initialization_manager.initialization()
sim_manager.simulate(simulation_buffers, time_control_variables)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["temperature"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (1,1)

# CREATE ANIMATION
create_2D_animation(
    data_dict,
    cell_centers,
    times,
    nrows_ncols=nrows_ncols,
    interval=100)

# CREATE FIGURE
create_2D_figure(
    data_dict,
    cell_centers=cell_centers,
    nrows_ncols=nrows_ncols,
    plane="xy", plane_value=0.0,
    colorbars="vertical",
    cmap="plasma",
    save_fig="heat_equation.png")