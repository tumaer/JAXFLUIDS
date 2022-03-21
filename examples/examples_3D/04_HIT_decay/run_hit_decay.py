import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

# SETUP SIMULATION
input_reader = InputReader("HIT_decay.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "pressure", "temperature", "velocityX", "velocityY", "velocityZ"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)

# PLOT
data_ref_1 = np.loadtxt("./reference_data/spyropoulos_case1.txt", skiprows=3)
data_ref_2 = np.loadtxt("./reference_data/spyropoulos_case2.txt", skiprows=3)
data_ref_3 = np.loadtxt("./reference_data/spyropoulos_case3.txt", skiprows=3)

dens     = data_dict["density"]
dens_rms = np.std(dens, axis=(1,2,3))

fig, ax = plt.subplots()
ax.plot(times/0.85, dens_rms)
ax.plot(data_ref_1[:,0], data_ref_1[:,1], linestyle="None", marker="o", markersize=4, mfc="black", mec="black")
ax.plot(data_ref_2[:,0], data_ref_2[:,1], linestyle="None", marker="o", markersize=4, mfc="black", mec="black")
ax.plot(data_ref_3[:,0], data_ref_3[:,1], linestyle="None", marker="o", markersize=4, mfc="black", mec="black", label="DNS")
ax.set_ylim([0, 0.16])
ax.set_yticks([0, 0.05, 0.1, 0.15])
ax.text(0.7, 0.15, "Case 1", transform=ax.transAxes, fontsize=12,
    verticalalignment='top')
ax.text(0.7, 0.35, "Case 2", transform=ax.transAxes, fontsize=12,
        verticalalignment='top')
ax.text(0.7, 0.55, "Case 3", transform=ax.transAxes, fontsize=12,
        verticalalignment='top')
ax.set_xlabel(r"$t / \tau$")
ax.set_ylabel(r"$\rho_{rms}$")
ax.set_xlim([0, 5])
ax.set_box_aspect(1)
ax.legend()
plt.show()