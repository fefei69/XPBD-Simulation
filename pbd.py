import time
import numpy as np
import pyvista as pv
import pdb
import os 
# Should only for 2 positions X1 and X2
def update_pos(position):
    diff, norm = pos_diff(position)
    n = diff/norm
    x1 = (norm - DISTANCE)/2 * 1/2 * (-n)
    x2 = (norm - DISTANCE)/2 * 1/2 * (n)
    # pdb.set_trace()
    return x1, x2 

# Should only for 2 positions X1 and X2
def pos_diff(position):
    diff_x = position[:,0][1:] - position[:,0][0:-1]
    diff_y = position[:,1][1:] - position[:,1][0:-1]
    diff_z = position[:,2][1:] - position[:,2][0:-1]
    diff = np.concatenate([diff_x.reshape(-1,1),diff_y.reshape(-1,1),diff_z.reshape(-1,1)],axis=-1)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    # pdb.set_trace()
    # np.norm()
    return diff, norm
    
def generate_particles(num_particles, bounds):
    x = np.linspace(bounds[0], bounds[1], num_particles)
    y = 10 * np.ones(num_particles)
    z = 10 * np.ones(num_particles)
    return np.column_stack((x, y, z))

spheres = pv.MultiBlock()   

# bounds for x 
bounds = [0, 100]
MASS = 1
DISTANCE = 25
# Generate particle positions
number_of_particles = 5
particle_positions = generate_particles(number_of_particles, bounds)
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
plotter.open_movie(f"videos/distance_constraint_PBD{num}.mp4")

# plotter.camera.focal_point = camera_focal_point
plotter.camera.position = camera_position
# plotter.camera.up = camera_up
# plotter.camera_position = [(25, 25, 25), (5, 5, 5), (0, 0, 1)]
# Set up the animation
frames = 1000  # Number of animation frames
dt = 0.01  # Time step for animation
plotter.write_frame()
for frame in range(frames):
    start = time.time()
    # Update particle positions based on gravity
    # not updating the first particle
    # pdb.set_trace()
    particle_positions[1:, 2] -= 0.1 * frame * dt
    for j in range(particle_positions.shape[0]-1):
        # print(particle_positions[1:,2])
        # pdb.set_trace()
        for _ in range(10):
            x1,x2 = update_pos(particle_positions[j:j+2])
            # print(x1)
            # pdb.set_trace()
            if j == 0:
                particle_positions[j] = particle_positions[j]
                particle_positions[j+1] = particle_positions[j+1] - x2[0]
            else:
                particle_positions[j] = particle_positions[j] - x1[0]
                particle_positions[j+1] = particle_positions[j+1] - x2[0]
    # pdb.set_trace()
    # Create a new set of spheres for each frame
    spheres = pv.MultiBlock()
    for pos in particle_positions:
        sphere = pv.Sphere(radius=1, center=pos)
        spheres.append(sphere)

    # Add the spheres to the plotter
    plotter.clear()
    plotter.add_mesh(spheres, color='red')
    # plotter.camera_position = 'xy'
    # Render the scene
    plotter.render()
    plotter.update()
    plotter.write_frame()
    print(f'Frame Time: {time.time() - start}')
plotter.close()