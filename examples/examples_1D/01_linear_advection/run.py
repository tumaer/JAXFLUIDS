import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_1D_figure, create_1D_animation

# SETUP SIMULATION
input_manager = InputManager("linear_advection.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density"]
jxf_data = load_data(path, quantities)

# PLOT
nrows_ncols = (1,1)

data = jxf_data.data
cell_centers = jxf_data.cell_centers
times = jxf_data.times

# CREATE ANIMATION
create_1D_animation(
    data,
    cell_centers,
    times,
    nrows_ncols=nrows_ncols,
    interval=50)

# CREATE FIGURE
create_1D_figure(
    data,
    cell_centers=cell_centers,
    nrows_ncols=nrows_ncols,
    axis="x", axis_values=(0,0), 
    save_fig="linear_advection.png")