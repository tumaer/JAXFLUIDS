import matplotlib.pyplot as plt
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

# SETUP SIMULATION
input_reader = InputReader("tgv.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["velocity", "vorticity"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
nrows_ncols = (1,2)
create_contourplot(data_dict, cell_centers, times, nrows_ncols, plane="xy", plane_value=np.pi/2, interval=100)

data_ref = np.loadtxt("tgv_reference_data.txt")
TKE = 0.5 * np.mean(data_dict["velocityX"]**2 + data_dict["velocityY"]**2 + data_dict["velocityZ"]**2, axis=(1,2,3))

fig, ax = plt.subplots()
ax.plot(times[:-1], -(K[1:]-K[:-1])/(times[1:]-times[:-1]), label="present")
ax.plot(data_ref[:,0], data_ref[:,1], marker='o', color="k", linestyle="none", label="Brachet 1991")
ax.set_xlabel("time")
ax.set_ylabel("rate of energy dissipation")
plt.show()