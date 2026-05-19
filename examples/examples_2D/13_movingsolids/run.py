from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# SETUP SIMULATION
input_manager = InputManager("movingsolids.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

# # LOAD DATA
path = sim_manager.output_writer.save_path_case
# path = "./results/movingsolids"
quantities = ["levelset", "pressure", "density"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities, step=10)

# PLOT
density = data_dict["density"]
levelset = data_dict["levelset"]
nrows_ncols = (1,1)
plot_dict = {
    "density" : np.ma.masked_where(levelset < 0.0, density),
    }
save_path = os.path.join(path,"images")
os.makedirs(save_path,exist_ok=True)
create_2D_animation(
    plot_dict,
    cell_centers,
    times, 
    levelset=data_dict["levelset"],
    nrows_ncols=nrows_ncols,
    plane="xy",
    save_png=save_path
)
# create_2D_figure(plot_dict, cell_centers=cell_centers, plane="xy", plane_value=0.0)