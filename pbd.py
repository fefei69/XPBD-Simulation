import time
import numpy as np
import pyvista as pv
import pdb
import os 
# Should only for 2 positions X1 and X2
def update_pos(position,L):
    dt = 0.1
    compliance = 1e-8/dt/dt
    diff = pos_diff(position)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    # computing lambda (for each constraint)
    delta_L = (-(norm - DISTANCE) - compliance*L)/(2+compliance)
    L_new = L + delta_L
    n = diff/norm
    delta_x1 = delta_L * MASS_INV * (-n)
    delta_x2 = delta_L * MASS_INV * (n)
    # print("diff",diff)
    # print("norm",norm)
    # pdb.set_trace()
    return delta_x1, delta_x2, L_new


# Should only for 2 positions X1 and X2
# X2 - X1
def pos_diff(position):
    diff_x = position[:,0][1:] - position[:,0][0:-1]
    diff_y = position[:,1][1:] - position[:,1][0:-1]
    diff_z = position[:,2][1:] - position[:,2][0:-1]
    diff = np.concatenate([diff_x.reshape(-1,1),diff_y.reshape(-1,1),diff_z.reshape(-1,1)],axis=-1)
    return diff
    
def generate_particles(num_particles, bounds):
    x = np.linspace(bounds[0], bounds[1], num_particles)
    y = 10 * np.ones(num_particles)
    z = 10 * np.ones(num_particles)
    return np.column_stack((x, y, z))

# spheres = pv.MultiBlock()   
ACC = np.array([0.,0.,-98])
# Setting each particle with same mass
MASS_INV = 1
new_lambda = 0
save = True
Velocity = 0
Fixed_point = True
# Generate particle positions
# bounds for x 
number_of_particles = 20
bounds = [0, 5000]
DISTANCE = bounds[1]/(number_of_particles-1)
particle_positions = generate_particles(number_of_particles, bounds)
if Fixed_point == True:
    V = Velocity*np.ones_like(particle_positions[1:])
    ACCELARATION = ACC*np.ones_like(particle_positions[1:])
else:
    V = Velocity*np.ones_like(particle_positions)
    ACCELARATION = ACC*np.ones_like(particle_positions)
# pdb.set_trace()
# print(particle_positions)
# x1_test,x2_test = update_pos(particle_positions)
# print(x1_test)
# pdb.set_trace()

# camera_focal_point = (-0.097, 0.168, -0.740)
camera_position =(0., -24413.195, -1000.)
camera_up = (0., 0., 1.)
num = len(os.listdir('videos'))
plotter = pv.Plotter()
spheres = pv.MultiBlock()  
pv.set_plot_theme('document')
plotter.set_background("black")  # You can change "black" to any color like "white", "blue", or an RGB tuple (r, g, b)
plotter.add_axes()
plotter.add_mesh(spheres, color='red', show_edges=True, lighting=True)
plotter.show(interactive_update=True)
plotter.camera_position = camera_position
plotter.camera.up = camera_up
# for pos in particle_positions:
#     sphere = pv.Sphere(radius=35, center=pos)
#     spheres.append(sphere)
# plotter.add_mesh(spheres, color='red')
# plotter.show(interactive_update=True)
# plotter.clear()
# plotter.camera.position = camera_position
# spheres = pv.MultiBlock()
if save == True:
    plotter.open_movie(f"videos/distance_constraint_PBD_test{num}.mp4")
    plotter.write_frame()

frames = 1000
dt = 0.1  # Time step for animation
solver_iter = 50
for frame in range(frames):
    start = time.time()
    # not updating the first particle
    old_particle_positions = particle_positions.copy()
    particle_positions[1:] = particle_positions[1:] + V * dt + ACCELARATION * (dt)**2   
    new_lambda = 0
    for j in range(particle_positions.shape[0]-1):
        # pdb.set_trace()
        # print(f"Distance{j}",np.linalg.norm(pos_diff(particle_positions[1:]), axis=-1, keepdims=True))
        for _ in range(solver_iter):
            x1,x2, new_lambda = update_pos(particle_positions[j:j+2],new_lambda)
            if j == 0:
                particle_positions[j] = particle_positions[j]
                particle_positions[j+1] = particle_positions[j+1] + x2[0]
            else:
                particle_positions[j] = particle_positions[j] + x1[0]
                particle_positions[j+1] = particle_positions[j+1] + x2[0]
    V = (1/dt)*(particle_positions[1:] - old_particle_positions[1:])
    # Create a new set of spheres for each frame
    spheres = pv.MultiBlock()
    # get camera parameter
    focal = plotter.camera.focal_point
    pos = plotter.camera.position
    up = plotter.camera.up
    print(f'Focal Point: ({focal[0]:.3f}, {focal[1]:.3f}, {focal[2]:.3f})')
    print(f'Camera Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')
    print(f'Camera Up: ({up[0]:.3f}, {up[1]:.3f}, {up[2]:.3f})')
    for pos in particle_positions:
        sphere = pv.Sphere(radius=35, center=pos)
        spheres.append(sphere)

    # Add the spheres to the plotter
    plotter.clear()
    plotter.add_mesh(spheres, color='red')
    # plotter.camera_position = 'xy'
    # Render the scene
    plotter.render()
    plotter.update()
    if save == True:
        plotter.write_frame()
    print(f'Frame Time: {time.time() - start} , Frame:{frame}')
plotter.close()