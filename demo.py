import numpy as np
import matplotlib.pyplot as plt
from particle_system import ParticleSystem
from wcsph import WCSPHSolver
import matplotlib
matplotlib.use('Agg')


def main():
    #instanciate particle system (the main window for particles and grids interactions)
    ps = ParticleSystem((512, 512)) 

    #bottom.
    ps.add_cube(lower_corner=[1, 1],    #initial position
                cube_size=[0.8, 0.2],   #initial shape
                velocity=[0.0, 0.0],  #initial velocity
                density=1000.0,         #particle density
                color=(245/255, 144/255, 39/255, 0.7),         #visualisation color
                material=0)

    #left.
    ps.add_cube(lower_corner=[1, 1.2],    #initial position
            cube_size=[0.2, 0.6],   #initial shape
            velocity=[0.0, 0.0],  #initial velocity
            density=1000.0,         #particle density
            color=(245/255, 144/255, 39/255, 0.7),         #visualisation color
            material=0)

    #top.
    ps.add_cube(lower_corner=[1.2, 1.6],    #initial position
        cube_size=[0.38, 0.2],   #initial shape
        velocity=[0.0, 0.0],  #initial velocity
        density=1000.0,         #particle density
        color=(245/255, 144/255, 39/255, 0.7),         #visualisation color 
        material=0)

    #right.
    ps.add_cube(lower_corner=[1.6, 1.2],    #initial position
        cube_size=[0.2, 0.6],   #initial shape
        velocity=[0.0, 0.0],  #initial velocity
        density=1000.0,         #particle density
        color=(245/255, 144/255, 39/255, 0.7),         #visualisation color 
        material=0)

    #middle
    ps.add_cube(lower_corner=[1.2, 1.2],    #initial position
        cube_size=[0.38, 0.38],   #initial shape
        velocity=[0.0, 0.0],  #initial velocity
        density=1000.0,         #particle density
        color=(39/255, 245/255, 243/255, 0.7),         #visualisation color 
        material=1)


    wcsph_solver = WCSPHSolver(ps)      #import solver



    # Create subplots
    frames = 0
    fps = 60
    running_time = 500
    iteration = int(running_time/fps)
    iteration = 1

    while frames <= iteration:
        for i in range(10):
            wcsph_solver.step()
        particle_info = ps.dump()
        particle_position = particle_info['position']
        particle_pressure = particle_info['pressure']
        particle_density = particle_info['density']
        particle_color = particle_info['color']

        #save data
        # with open('./logfiles/position_{}.npy'.format(str(frames).zfill(6)), 'wb') as f:
        #     np.save(f, particle_info['position'])
        # f.close()
        # with open('./logfiles/density_{}.npy'.format(str(frames).zfill(6)), 'wb') as f:
        #     np.save(f, particle_info['density'])
        # f.close()
        # with open('./logfiles/pressure_{}.npy'.format(str(frames).zfill(6)), 'wb') as f:
        #     np.save(f, particle_info['pressure'])
        # f.close()
        # with open('./logfiles/color_{}.npy'.format(str(frames).zfill(6)), 'wb') as f:
        #     np.save(f, particle_info['color'])
        # f.close()
        if frames%1 ==0:
            fig = plt.figure(figsize=(10,10), dpi=70 )
            plt.scatter(particle_position[:, 0], particle_position[:, 1], c = particle_color, marker = '.')
            plt.xlim(0, 3)
            plt.ylim(0, 3)
            plt.savefig('./images/frame_{}.png'.format(str(frames).zfill(6)))
            plt.cla()
            plt.clf()
            plt.close()


        print(frames, '/', iteration)
        frames += 1

if __name__ == "__main__":
    import time
    start = time.time()
    print("hello")
    

    #profiling code
    import cProfile
    cProfile.run('main()', 'output.dat')

    import pstats
    from pstats import SortKey

    with open('output_time.txt', 'w') as f:
        p = pstats.Stats('Output.dat', stream = f)
        p.sort_stats('time').print_stats()

    with open('output_calls.txt', 'w') as f:
        p = pstats.Stats('output.dat', stream = f)
        p.sort_stats('calls').print_stats()



    end = time.time()
    print('time elapsed:', end - start)
    