import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure

# SETUP SIMULATION
input_manager = InputManager("NACA.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = [
    "density", "schlieren", "mach_number", 
    "levelset", "volume_fraction", "pressure"
]
jxf_data = load_data(path, quantities)

cell_centers = jxf_data.cell_centers
data = jxf_data.data
times = jxf_data.times

mask_fluid = data["volume_fraction"] > 0.0
mask_solid = 1.0 - mask_fluid

# PLOT
nrows_ncols = (2,2)
plot_dict = {
    "density"       : np.ma.masked_where(mask_solid, data["density"]),
    "pressure"      : np.ma.masked_where(mask_solid, data["pressure"]),
    "mach_number"   : np.clip(np.ma.masked_where(mask_solid, data["mach_number"]), 0.0, 3.0),
    "schlieren"     : np.clip(np.ma.masked_where(mask_solid, data["schlieren"]), 1e0, 5e2)
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
    nrows_ncols=nrows_ncols,
    cell_centers=cell_centers, 
    plane="xy", plane_value=0.0,
    dpi=400, save_fig="NACA.png")
