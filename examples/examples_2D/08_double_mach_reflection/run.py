import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_figure, create_2D_animation

# SETUP SIMULATION
input_manager = InputManager("double_mach_reflection.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocity", "pressure", "mach_number"]
jxf_data = load_data(path, quantities)

cell_centers = jxf_data.cell_centers
data = jxf_data.data
times = jxf_data.times

# PLOT
nrows_ncols = (1,3)
plot_dict = {
    "density": data["density"], 
    "pressure": data["pressure"],
    "mach_number": data["mach_number"],
}

# CREATE ANIMATION
create_2D_animation(
    plot_dict,
    cell_centers,
    times,
    nrows_ncols=nrows_ncols,
    plane="xy", plane_value=0.0,
    interval=100)

# CREATE FIGURE
create_2D_figure(
    plot_dict,
    nrows_ncols,
    cell_centers=cell_centers,
    plane="xy", plane_value=0.0,
    save_fig="double_mach_reflection.png")