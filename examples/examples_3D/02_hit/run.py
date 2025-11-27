import json
import matplotlib.pyplot as plt
import numpy as np

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# RUN JXF SIMULATIONS
results = {}
for mach_number in (0.2, 0.4, 0.6):

    case_setup = json.load(open("HIT_decay.json", "r"))
    case_setup["initial_condition"]["turbulent"]["parameters"]["ma_target"] = mach_number
    case_setup["material_properties"]["equation_of_state"]["specific_gas_constant"] = 1 / 1.4 / mach_number**2

    # SETUP SIMULATION
    input_manager = InputManager(case_setup, "numerical_setup.json")
    initialization_manager = InitializationManager(input_manager)
    sim_manager = SimulationManager(input_manager)

    # RUN SIMULATION
    jxf_buffers = initialization_manager.initialization()
    sim_manager.simulate(jxf_buffers)

    # LOAD DATA
    path = sim_manager.output_writer.save_path_domain
    quantities = ["density"]
    jxf_data = load_data(path, quantities)

    times = jxf_data.times
    density = jxf_data.data["density"]
    density_rms = np.std(density, axis=(1,2,3))

    results[f"ma{mach_number}"] = (times, density_rms)


# PLOT
# NOTE We load the reference data from Spyropoulos et al. 1996
data_ref = [
    np.loadtxt(f"./reference_data/spyropoulos_case{i+1}.txt", skiprows=3) for i in range(3)
]

# NOTE We non-dimensionalize time by the initial eddy turnover time
# tau_ref = 0.85, see Sec. 5.1.4 in Bezgin et al. 2023
tau_ref = 0.85

fig, ax = plt.subplots()
for i, (k, v) in enumerate(results.items()):
    label = "JXF" if i == len(results) - 1 else None
    times, density_rms = v
    ax.plot(times / tau_ref, density_rms, label=label)

ref_kwargs = {
    "linestyle": "None",
    "marker": "o",
    "markersize": 4,
    "mfc": "black",
    "mec": "black"
}
for i, data_ref_i in enumerate(data_ref):
    label = "DNS" if i == len(data_ref) - 1 else None
    ax.plot(data_ref_i[:,0], data_ref_i[:,1], label=label, **ref_kwargs)

text_kwargs = {"transform": ax.transAxes, "fontsize": 10, "verticalalignment": "top"}
for i, y_pos in enumerate((0.125, 0.3, 0.65)):
    ax.text(0.5, y_pos, f"Case {i+1}", **text_kwargs)

ax.set_ylim([0, 0.16])
ax.set_yticks([0, 0.05, 0.1, 0.15])
ax.set_xlabel(r"$t / \tau$")
ax.set_ylabel(r"$\rho_{rms}$")
ax.set_xlim([0, 5])
ax.set_box_aspect(1)
ax.legend()
plt.savefig("hit_plot.png", bbox_inches="tight", dpi=400)
plt.show()
