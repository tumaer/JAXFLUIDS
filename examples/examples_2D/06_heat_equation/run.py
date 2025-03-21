import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

# SETUP SIMULATION
input_manager = InputManager("heat_equation.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["temperature"]
jxf_data = load_data(path, quantities)

cell_centers = jxf_data.cell_centers
data = jxf_data.data
times = jxf_data.times

# PLOT
nrows_ncols = (1,1)

# CREATE ANIMATION
create_2D_animation(
    data,
    cell_centers,
    times,
    nrows_ncols=nrows_ncols,
    interval=100)

# CREATE FIGURE
create_2D_figure(
    data,
    cell_centers=cell_centers,
    nrows_ncols=nrows_ncols,
    plane="xy", plane_value=0.0,
    colorbars="vertical",
    cmap="plasma",
    save_fig="heat_equation.png")