import matplotlib.pyplot as plt
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot

# SETUP SIMULATION
input_reader = InputReader("poiseuille.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocityX", "pressure"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (1,len(quantities))
create_lineplot(data_dict, cell_centers, times, nrows_ncols=nrows_ncols, axis="y", values=[0.0, 0.0])

def poiseuille_analytical(y, dm, rho, h):
    return 6 * dm / rho / h**3 * y * (h - y)

exact_solution = poiseuille_analytical(
    cell_centers[1], 
    dm = input_reader.mass_flow_target, 
    rho = input_reader.initial_condition["rho"], 
    h = input_reader.domain_size["y"][1] - input_reader.domain_size["y"][0])

fig, ax = plt.subplots(ncols=3)
ax[0].plot(cell_centers[1], data_dict["density"][-1,0,:,0], color="red")
ax[1].plot(cell_centers[1], data_dict["velocityX"][-1,0,:,0], color="red")
ax[1].plot(cell_centers[1], exact_solution, color="black", linestyle="--")
ax[2].plot(cell_centers[1], data_dict["pressure"][-1,0,:,0], color="red")
plt.show()