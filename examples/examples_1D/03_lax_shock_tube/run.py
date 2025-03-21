import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_1D_animation, create_1D_figure

# SETUP SIMULATION
input_manager = InputManager("lax.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocity", "pressure"]
jxf_data = load_data(path, quantities)

data = jxf_data.data
cell_centers = jxf_data.cell_centers
times = jxf_data.times

# PLOT
plot_dict = {
    "density": data["density"], 
    "velocityX": data["velocity"][:,0],
    "pressure": data["pressure"]
}
nrows_ncols = (1,3)

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
    save_fig="lax_shock_tube.png")