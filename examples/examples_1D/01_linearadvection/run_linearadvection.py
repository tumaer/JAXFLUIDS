import matplotlib.pyplot as plt
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot

# SETUP SIMULATION
input_reader = InputReader("linearadvection.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (1,1)
create_lineplot(data_dict, cell_centers, times, nrows_ncols=nrows_ncols, interval=100)

fig, ax = plt.subplots()
ax.plot(cell_centers[0], data_dict["density"][-1,:,0,0])
ax.plot(cell_centers[0], data_dict["density"][0,:,0,0], color="black")
plt.show()