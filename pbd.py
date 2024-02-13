import time
import numpy as np
import pyvista as pv
import pdb
import os 
# Should only for 2 positions X1 and X2
def update_pos(position,L):
    dt = 0.01
    compliance = 1e-8/dt/dt
    diff = pos_diff(position)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    # computing lambda (for each constraint)
    delta_L = (-(norm - DISTANCE) - compliance*L)/(2+compliance)
    L_new = L - delta_L
    n = diff/norm
    delta_x1 = delta_L * MASS_INV * (n)
    delta_x2 = delta_L * MASS_INV * (-n)
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

spheres = pv.MultiBlock()   


ACC = np.array([0.,0.,-9.8])
# Setting each particle with same mass
MASS_INV = 1
new_lambda = 0
save = True
Velocity = 0
Fixed_point = True
# Generate particle positions
# bounds for x 
number_of_particles = 5
bounds = [0, 500]
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
camera_position =(500, 500, 0)
# camera_up = (-0.714, -0.288, -0.639)
num = len(os.listdir('videos'))
plotter = pv.Plotter()
pv.set_plot_theme('document')
plotter.add_axes()
plotter.add_mesh(spheres, color='red', show_edges=True, lighting=True)
plotter.show(interactive_update=True)
plotter.camera_position = 'xz'
if save == True:
    plotter.open_movie(f"videos/distance_constraint_PBD_test{num}.mp4")
    plotter.write_frame()
plotter.camera.position = camera_position

frames = 300 
dt = 0.01  # Time step for animation
solver_iter = 100
for frame in range(frames):
    start = time.time()
    # not updating the first particle
    particle_positions[1:] = particle_positions[1:] + V * frame * dt + ACCELARATION * (frame * dt)**2   
    old_particle_positions = particle_positions.copy()
    # print(f"it: {frame}, Before loop\n",particle_positions)
    for j in range(particle_positions.shape[0]-1):
        # pdb.set_trace()
        for _ in range(solver_iter):
            x1,x2, new_lambda = update_pos(particle_positions[j:j+2],new_lambda)
            # print(new_lambda)
            # print(x1)
            # pdb.set_trace()
            if j == 0:
                particle_positions[j] = particle_positions[j]
                particle_positions[j+1] = particle_positions[j+1] - x2[0]
            else:
                particle_positions[j] = particle_positions[j] - x1[0]
                particle_positions[j+1] = particle_positions[j+1] - x2[0]
    if frame == 200:
        pdb.set_trace()
    # print(particle_positions[1:])
    # print(old_particle_positions[1:])
    V = (1/dt/frames)*(particle_positions[1:] - old_particle_positions[1:])
    # print("Velocity",particle_positions[1:] - old_particle_positions[1:])
    print("Distance",np.linalg.norm(pos_diff(particle_positions[1:]), axis=-1, keepdims=True))
    
    # print(f"it:{frame} , after loop\n",particle_positions)
    # print(old_particle_positions)
    # pdb.set_trace()
    # Create a new set of spheres for each frame
    spheres = pv.MultiBlock()
    for pos in particle_positions:
        sphere = pv.Sphere(radius=5, center=pos)
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