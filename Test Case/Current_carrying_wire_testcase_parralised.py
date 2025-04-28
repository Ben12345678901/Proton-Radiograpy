'''
Author - Ben fisher
email  - bf698@york.ac.uk
date   - 10/03/2025
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.interpolate import griddata  # For interpolating on uniform grids
from plasmapy.diagnostics.charged_particle_radiography import synthetic_radiography as cpr
from plasmapy.plasma.grids import CartesianGrid  # For creating PlasmaPy-compatible grids
from astropy import units as u  # For handling physical units
from joblib import Parallel, delayed  # For parallel processing
import time

# -----------------------------------------------------------
# 0. Define parallelisation parameters and global settings
# -----------------------------------------------------------
start_time_program = time.time()  # Start timing the entire program

# Switch to turn on/off all plotting. Set to False to disable plots.
enable_plotting = True

# Define number of CPU cores to use in various parallel steps.
n_cores = 4
n_jobs = n_cores            # Number of cores for parallel interpolation
n_chunks = n_cores          # Number of chunks for the proton simulation
n_theta_chunks = n_cores    # Number of chunks to build the full 3D magnetic field

# Total number of particles (protons) to simulate in the proton radiography step.
total_particles = int(1e6)

# -----------------------------------------------------------
# 1. Create the Test Field on an R-Z Slice
# -----------------------------------------------------------
parameter = 20  # Resolution parameter for the grid

# Define the number of grid points along the R and Z directions.
N_R = parameter
N_Z = parameter

# Create arrays of R and Z values. R starts at 0.1 to avoid division by zero.
R_vals = np.linspace(0.1, 1.0, N_R)
Z_vals = np.linspace(-1.0, 1.0, N_Z)

# Create a 2D meshgrid for R and Z. 'ij' indexing makes the first index correspond to R.
R_mesh, Z_mesh = np.meshgrid(R_vals, Z_vals, indexing='ij')

# For a current-carrying wire along the z-axis:
#   - Radial component (v_r) is zero,
#   - Azimuthal component (v_theta) is 1/R,
#   - Axial component (v_z) is zero.
v_r_slice = np.zeros_like(R_mesh)
v_theta_slice = 1.0 / R_mesh  # Compute 1/R (R is always >=0.1)
v_z_slice = np.zeros_like(R_mesh)

# Plot the R-Z slice of v_theta if plotting is enabled.
if enable_plotting:
    plt.figure(figsize=(8,6))
    contour = plt.contourf(R_mesh, Z_mesh, v_theta_slice, cmap='viridis')
    plt.xlabel("R")
    plt.ylabel("Z")
    plt.title("Azimuthal Component (v_theta) for a Current-Carrying Wire")
    plt.colorbar(contour, label="v_theta (1/R)")
    plt.show()

# -----------------------------------------------------------
# 2. Build the 3D Field by Rotating the R-Z Slice
# -----------------------------------------------------------
# Create an array of theta values from 0 to 2*pi.
theta = np.linspace(0, 2*np.pi, parameter)

# Split the theta array into chunks to parallelize replication.
theta_chunks = np.array_split(theta, n_theta_chunks)

def replicate_field(chunk):
    """
    Given a chunk of theta values, replicate the 2D field slices (v_r, v_theta, v_z)
    along the theta direction for the number of points in the chunk.
    """
    n_chunk = len(chunk)
    # Replicate along a new axis for the theta dimension.
    v_r_chunk = np.repeat(v_r_slice[:, np.newaxis, :], n_chunk, axis=1)
    v_theta_chunk = np.repeat(v_theta_slice[:, np.newaxis, :], n_chunk, axis=1)
    v_z_chunk = np.repeat(v_z_slice[:, np.newaxis, :], n_chunk, axis=1)
    return v_r_chunk, v_theta_chunk, v_z_chunk

# Process each theta chunk in parallel.
results = Parallel(n_jobs=n_theta_chunks)(
    delayed(replicate_field)(chunk) for chunk in theta_chunks
)

# Concatenate the results along the theta axis (axis=1) to build full 3D arrays.
v_r_3d = np.concatenate([res[0] for res in results], axis=1)
v_theta_3d = np.concatenate([res[1] for res in results], axis=1)
v_z_3d = np.concatenate([res[2] for res in results], axis=1)

# Create 3D meshgrids for cylindrical coordinates (R, theta, Z).
# Note that the theta values come from the full theta array.
R_grid, Theta_grid, Z_grid = np.meshgrid(R_vals, theta, Z_vals, indexing='ij')

'''
# Serial version (for reference):
# v_r_3d = np.repeat(v_r_slice[:, np.newaxis, :], len(theta), axis=1)
# v_theta_3d = np.repeat(v_theta_slice[:, np.newaxis, :], len(theta), axis=1)
# v_z_3d = np.repeat(v_z_slice[:, np.newaxis, :], len(theta), axis=1)
# R_grid, Theta_grid, Z_grid = np.meshgrid(R_vals, theta, Z_vals, indexing='ij')
'''

# -----------------------------------------------------------
# 3. Convert from Cylindrical to Cartesian Coordinates
# -----------------------------------------------------------
# Convert spatial coordinates: X = R*cos(theta), Y = R*sin(theta)
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)
Z_cart = Z_grid  # Z remains the same

# Convert vector components from cylindrical to Cartesian.
# For vectors: v_x = v_r*cos(theta) - v_theta*sin(theta)
#              v_y = v_r*sin(theta) + v_theta*cos(theta)
v_x = v_r_3d * np.cos(Theta_grid) - v_theta_3d * np.sin(Theta_grid)
v_y = v_r_3d * np.sin(Theta_grid) + v_theta_3d * np.cos(Theta_grid)
v_z_cart = v_z_3d  # remains zero

# Calculate the magnetic field magnitude at each point.
B_magnitude = np.sqrt(v_x**2 + v_y**2 + v_z_cart**2)

# Define the physical constant: permeability of free space (in SI units)
mu0 = 4 * np.pi * 1e-7  # N/A^2

# Calculate magnetic energy density using u_B = B^2 / (2*mu0)
energy_density = B_magnitude**2 / (2 * mu0)  # in Joules per cubic meter

# Determine grid spacings in cylindrical coordinates.
dR = R_vals[1] - R_vals[0]
dtheta = theta[1] - theta[0]
dZ = Z_vals[1] - Z_vals[0]

# Calculate the volume element in cylindrical coordinates: dV = R*dR*dtheta*dZ.
# Note: R_grid holds the R values at each grid point.
dV = R_grid * dR * dtheta * dZ

# Compute the total magnetic field energy by summing over all grid cells.
total_energy = np.sum(energy_density * dV)
print("Total magnetic field energy:", total_energy, "Joules")

# -----------------------------------------------------------
# 4. Visualize the 3D Field with a Quiver Plot (Nonuniform grid)
# -----------------------------------------------------------
if enable_plotting:
    step = 2  # Downsample the grid for clarity in plotting
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(
        X[::step, ::step, ::step],   # X positions
        Y[::step, ::step, ::step],   # Y positions
        Z_cart[::step, ::step, ::step],  # Z positions
        v_x[::step, ::step, ::step],  # X vector component
        v_y[::step, ::step, ::step],  # Y vector component
        v_z_cart[::step, ::step, ::step],  # Z vector component
        length=0.1, normalize=True
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Magnetic Field (Non-uniform Cartesian Grid)")
    plt.show()

# -----------------------------------------------------------
# 5. Create a Uniform Cartesian Grid and Interpolate the Field
# -----------------------------------------------------------
# Determine bounds from the Cartesian coordinates.
x_min, x_max = np.min(X), np.max(X)
y_min, y_max = np.min(Y), np.max(Y)
z_min, z_max = np.min(Z_cart), np.max(Z_cart)

# Define uniform grid axes (using the same 'parameter' resolution).
nx = parameter
ny = parameter
nz = parameter

x_uniform = np.linspace(x_min, x_max, nx)
y_uniform = np.linspace(y_min, y_max, ny)
z_uniform = np.linspace(z_min, z_max, nz)

# Create a uniform Cartesian meshgrid.
X_uniform, Y_uniform, Z_uniform = np.meshgrid(x_uniform, y_uniform, z_uniform, indexing='ij')

# Prepare original (nonuniform) grid points for interpolation.
points_original = np.column_stack((X.flatten(), Y.flatten(), Z_cart.flatten()))

def interpolate_component(v_field):
    """
    Interpolates a given magnetic field component (v_field) from the nonuniform grid
    onto the uniform grid (X_uniform, Y_uniform, Z_uniform) using linear interpolation.
    """
    return griddata(points_original, v_field.flatten(),
                    (X_uniform, Y_uniform, Z_uniform),
                    method='linear')

# Start timing the interpolation step.
start_time = time.time()

# Use parallel processing to interpolate v_x, v_y, and v_z simultaneously.
v_x_uniform, v_y_uniform, v_z_uniform = Parallel(n_jobs=n_jobs)(
    delayed(interpolate_component)(v) for v in (v_x, v_y, v_z_cart)
)

# End timing the interpolation.
end_time = time.time()
print(f"Interpolation took {end_time - start_time:.2f} seconds")

# Compute the magnitude of the interpolated magnetic field.
B_uniform = np.sqrt(v_x_uniform**2 + v_y_uniform**2 + v_z_uniform**2)

# Calculate the energy density on the uniform grid.
energy_density_cart = B_uniform**2 / (2 * mu0)

# Determine grid spacing in the uniform grid.
dx = x_uniform[1] - x_uniform[0]
dy = y_uniform[1] - y_uniform[0]
dz = z_uniform[1] - z_uniform[0]
dV_cart = dx * dy * dz

# Compute the total energy on the uniform grid (ignore NaNs from interpolation).
total_energy_cart = np.nansum(energy_density_cart * dV_cart)
print("Total magnetic field energy (Cartesian):", total_energy_cart, "Joules")

# Compute the difference between the cylindrical and Cartesian energy calculations.
energy_difference = (total_energy / total_energy_cart)
print("Energy difference between the cylindrical and cartesian grids after interpolation:", energy_difference, "%")

# -----------------------------------------------------------
# 6. Visualize the Uniform Cartesian Grid Field
# -----------------------------------------------------------
if enable_plotting:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    # Downsample the grid for clarity.
    s = (slice(None, None, 2), slice(None, None, 2), slice(None, None, 2))
    ax.quiver(
        X_uniform[s], Y_uniform[s], Z_uniform[s],
        v_x_uniform[s], v_y_uniform[s], v_z_uniform[s],
        length=0.1, normalize=True
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Magnetic Field (Uniform Cartesian Grid)")
    plt.show()

# Replace any NaN or infinite values with zeros to avoid errors later.
v_x_uniform = np.nan_to_num(v_x_uniform, nan=0.0, posinf=0.0, neginf=0.0)
v_y_uniform = np.nan_to_num(v_y_uniform, nan=0.0, posinf=0.0, neginf=0.0)
v_z_uniform = np.nan_to_num(v_z_uniform, nan=0.0, posinf=0.0, neginf=0.0)

# -----------------------------------------------------------
# 7. Mapping our Uniform Grid into PlasmaPy
# -----------------------------------------------------------
# Create a CartesianGrid with spatial dimensions scaled by 0.1 mm.
grid = CartesianGrid(X_uniform * 0.1 * u.mm, Y_uniform * 0.1 * u.mm, Z_uniform * 0.1 * u.mm)
# Add magnetic field quantities to the grid.
grid.add_quantities(B_x=v_x_uniform * u.T, B_y=v_y_uniform * u.T, B_z=v_z_uniform * u.T)

# Recalculate the energy density using the PlasmaPy grid.
mu0 = 4 * np.pi * 1e-7 * u.N / u.A**2  # permeability in SI with units
B_plasma = np.sqrt(grid["B_x"]**2 + grid["B_y"]**2 + grid["B_z"]**2)
energy_density_plasma = B_plasma**2 / (2 * mu0)
print(energy_density_plasma)

# Compare energy densities between the Cartesian and PlasmaPy grids.
diff = energy_density_cart / energy_density_plasma
print("Energy density difference", diff, "%")

# Compute the grid spacing in SI units (convert from 0.1 mm scale to meters).
dx = ((x_uniform[1] - x_uniform[0]) * 0.1 * u.mm).to(u.m)
dy = ((y_uniform[1] - y_uniform[0]) * 0.1 * u.mm).to(u.m)
dz = ((z_uniform[1] - z_uniform[0]) * 0.1 * u.mm).to(u.m)
dV_plasma = dx * dy * dz

# Calculate the total energy stored in the plasma grid.
total_energy_plasma = np.sum(energy_density_plasma) * dV_plasma
print("Total magnetic field energy (Plasma Grid):", total_energy_plasma.to(u.J), "Joules")

if enable_plotting:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(30, 30)  # Set a fixed view angle
    # Downsample for clarity.
    s = (slice(None, None, 2), slice(None, None, 2), slice(None, None, 2))
    ax.quiver(
        grid.pts0[s].to(u.mm).value,
        grid.pts1[s].to(u.mm).value,
        grid.pts2[s].to(u.mm).value,
        grid["B_x"][s],
        grid["B_y"][s],
        grid["B_z"][s],
        length=1e-1, normalize=True
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title("Magnetic Field to be probed")
    plt.show()

# -----------------------------------------------------------
# 8. Proton Radiography Simulation
# -----------------------------------------------------------
start_time_sim = time.time()  # Start timing the simulation

# Define source and detector positions for the radiography.
source = (0 * u.mm, 0 * u.mm, 10 * u.mm)
detector = (0 * u.mm, 0 * u.mm, -150 * u.mm)

# Determine particles per simulation chunk.
particles_per_chunk = total_particles // n_chunks

# Radiograph parameters: detector size and number of bins (resolution).
size = np.array([[-1, 1], [-1, 1]]) * 0.25 * u.cm
bins = [300, 300]

def run_simulation_chunk(chunk_id):
    """
    Runs a simulation for a chunk of particles.
    Creates a Tracker, generates particles, runs the simulation,
    and returns the radiograph intensity from that chunk.
    """
    sim_chunk = cpr.Tracker(grid, source, detector, verbose=False)
    sim_chunk.create_particles(
        particles_per_chunk,
        4 * u.MeV,
        max_theta=np.pi / 30 * u.rad,
        distribution='uniform'
    )



    sim_chunk.run()
    # Generate the synthetic radiograph for this chunk.
    # hax and vax (detector grid) are assumed to be identical across chunks.
    hax, vax, intensity_chunk = cpr.synthetic_radiograph(sim_chunk, size=size, bins=bins)
    return intensity_chunk

# Run simulation chunks in parallel.
intensity_chunks = Parallel(n_jobs=n_chunks)(
    delayed(run_simulation_chunk)(i) for i in range(n_chunks)
)

# Sum the intensity contributions from each chunk.
combined_intensity = np.sum(intensity_chunks, axis=0)

# Create a dummy simulation to obtain the detector coordinate grids (hax, vax).
dummy_tracker = cpr.Tracker(grid, source, detector, verbose=False)
dummy_tracker.create_particles(
    1,  # Only one particle is needed to generate the grid
    4 * u.MeV,
    max_theta=np.pi / 30 * u.rad,
    distribution='uniform'
)
dummy_tracker.run()
hax, vax, _ = cpr.synthetic_radiograph(dummy_tracker, size=size, bins=bins)

def plot_radiograph(hax, vax, intensity):
    """
    Plots the synthetic radiograph using pcolormesh.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    mesh = ax.pcolormesh(
        hax.to(u.cm).value,  # Convert horizontal axis to cm
        vax.to(u.cm).value,  # Convert vertical axis to cm
        np.log(intensity.T),         # Transpose so that orientation is correct
        cmap="inferno",      # Color map choice
        shading="auto"
    )
    cb = fig.colorbar(mesh)
    cb.ax.set_ylabel("Intensity")
    ax.set_aspect("equal")
    ax.set_xlabel("X (cm), Image plane")
    ax.set_ylabel("Y (cm), Image plane")
    ax.set_title("Synthetic Proton Radiograph (Parallel Simulation)")
    plt.savefig("output_radiograph.png")
    plt.show()

end_time_sim = time.time()   # End simulation timing
end_time_program = time.time() # End program timing

# Plot the combined radiograph.
plot_radiograph(hax, vax, combined_intensity)

# Print timing and resource usage information.
print(f"Total run time: {end_time_program - start_time_program:.2f} seconds")
print(f"Interpolation took: {end_time - start_time:.2f} seconds")
print(f"Simulation took: {end_time_sim - start_time_sim:.2f} seconds")
print(f"Number of chunks used in the simulation: {n_chunks}")
print(f"Cores used in interpolation: {n_jobs}")
