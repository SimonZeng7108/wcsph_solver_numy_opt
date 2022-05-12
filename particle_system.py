import numpy as np
from functools import reduce

class ParticleSystem:
    def __init__(self, res):
        self.res = res                                          #resolution of the particle system
        self.dim = len(res)                                     #dimension of system; 2d

        #Material index for fluid and boundary
        self.material_boundary = 0                              #boundary material index
        self.material_fluid = 1                                 #fluid material index

        #Particle parameters
        self.particle_radius = 0.05                             #particle radius
        self.support_radius = self.particle_radius * 4.0        #support radius
        self.m_V = np.pi * self.particle_radius**self.dim       #Volume

        self.particle_num = 0                                   #initilise particle numbers

        #Grid properties
        self.grid_size = self.support_radius                    #neighbour search grid size
        self.grid_num = np.ceil(np.array(res) / self.grid_size).astype(int)         #neighbour search grid number
        self.grid_particles_num = np.zeros((self.grid_num), dtype = np.int8)             #initialise grid particle numbsers
        self.grid_particles = dict()                            #initialise the particles in a grid

        #Particle properties
        self.x = np.zeros((self.particle_num, self.dim), dtype = np.float64)                 #particles positions
        self.v = np.zeros((self.particle_num, self.dim), dtype = np.float64)                 #particles velocities
        self.density = np.zeros((self.particle_num, 1), dtype = np.float64)           #particles densities
        self.pressure = np.zeros((self.particle_num, 1), dtype = np.float64)          #particles pressures                                         
        self.material = np.zeros((self.particle_num, 1), dtype = np.int8)       #particles materials
        self.color = np.zeros((self.particle_num, 4), dtype = np.int8)          #particle colors

        self.particle_neighbors = dict()                            #list of each particle neighbor indices
        self.particle_neighbors_num = np.zeros((self.particle_num, 1), dtype = np.int8)          #list of each particle neighbor numbers


    ######Neighbor search######
    def pos_to_index(self, pos):
        "convert particle positions to grid indices in (i, j)"
        cell = np.floor((pos / self.grid_size)).astype(np.int8)
        return cell
    
    def is_valid_cell(self, cell):
        """check if the cell is a valid cell in the domain"""
        """when making particle arrays,
        make sure no particle lies on right and bottom edges, 
        e.g x[0] < res[0], y[1] < res[1]"""

        flag = True
        for d in range(self.dim):
            flag = flag and (0 <= cell[d] < self.grid_num[d])
        return flag

    def allocate_particles_to_grid(self):
        """match each particle to grid indices"""
        for p in range(self.particle_num):                          #loop over all particles
            cell = self.pos_to_index(self.x[p])                     #fetch position to grid indices in (i, j)
            if str(cell) not in self.grid_particles:
                self.grid_particles[str(cell)] = [p]
            else:
                self.grid_particles[str(cell)].append(p)                   #allocate the particle id to grid particles lists
            self.grid_particles_num[cell[0], cell[1]] += 1
            

    def search_neighbors(self):
        """main neighbor search algorithm"""
        for p_i in range(self.particle_num):
            
            center_cell = self.pos_to_index(self.x[p_i])

            for x_offset in range(-1, 2):
                for y_offset in range(-1, 2):
                    cell = np.zeros(2, dtype = np.int8)
                    cell[0] = center_cell[0] + x_offset
                    cell[1] = center_cell[1] + y_offset
                    if self.is_valid_cell(cell):
                        for j in range(self.grid_particles_num[cell[0], cell[1]]):
                            p_j = self.grid_particles[str(cell)][j]
                            x_diff = self.x[p_i] - self.x[p_j]
                            x_norm = (x_diff[0]**2 + x_diff[1]**2)**(1/2)
                            distance = x_norm
                            if p_i != p_j and distance < self.support_radius:
                                if str(p_i) not in self.particle_neighbors:
                                    self.particle_neighbors[str(p_i)] = [p_j]
                                else:
                                    self.particle_neighbors[str(p_i)].append(p_j)
                                self.particle_neighbors_num[p_i]+= 1

            if str(p_i) not in self.particle_neighbors:
                self.particle_neighbors[str(p_i)] = []

    def initialize_particle_system(self):
        """apply cell localisation and neighbour search"""
        self.grid_particles = dict()                                #initialise the particles in a grid
        self.grid_particles_num = np.zeros((self.grid_num), dtype = np.int8)             #initialise grid particle numbsers
        self.particle_neighbors = dict()                            #list of each particle neighbor indices
        self.particle_neighbors_num = np.zeros((self.particle_num, 1), dtype = np.int8)          #list of each particle neighbor numbers
        self.allocate_particles_to_grid()
        self.search_neighbors()



    ######Add particles######
    def add_particles(self,
                        new_particles_num,
                        new_particles_positions,
                        new_particles_velocity,
                        new_particle_density,
                        new_particle_pressure,
                        new_particles_material,
                        new_particles_color):

        for p in range(new_particles_num):
            self.v = np.append(self.v, new_particles_velocity[p].reshape((1, 2)), axis = 0)
            self.x = np.append(self.x, new_particles_positions[p].reshape((1, 2)), axis = 0)
            
            self.density = np.append(self.density, new_particle_density[p].reshape((1, 1)), axis = 0)
            self.pressure = np.append(self.pressure, new_particle_pressure[p].reshape((1, 1)), axis = 0)
            self.material = np.append(self.material, new_particles_material[p].reshape((1, 1)), axis = 0)
            self.color = np.append(self.color, new_particles_color[p].reshape((1, 4)), axis = 0)
        self.particle_num += new_particles_num
        self.particle_neighbors_num = np.zeros((self.particle_num, 1), dtype = np.int8)

    def add_cube(self,
                lower_corner,
                cube_size,
                material,
                color=0xFFFFFF,
                density=None,
                pressure=None,
                velocity=None):

        num_dim = []
        for i in range(2):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i],  self.particle_radius))

        
        num_new_particles = reduce(lambda x, y: x * y, [len(n) for n in num_dim])


        new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)
        new_positions = new_positions.reshape(-1,reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros((num_new_particles, 1)), material)
        color = np.full_like(np.zeros((num_new_particles, 4)), np.array(color).reshape(1, 4))
        density = np.full_like(np.zeros((num_new_particles, 1)), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros((num_new_particles, 1)), pressure if pressure is not None else 0.)

        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)


    def add_fluids(self,
                coord_file,
                material,
                color=0x68b2f7,
                density=None,
                pressure=None,
                velocity=None):

        with open(coord_file, 'rb') as f:
            a = np.load(f, allow_pickle=True)
        num_new_particles = a.shape[0]
        # new_positions = a / 20 - 5
        # new_positions[:, 0] -= 1
        new_positions = a / 15
        new_positions[:, 0] -= 11.65
        new_positions[:, 1] -= 11.65
        #print(new_positions)
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros((num_new_particles, 1)), material)
        color = np.full_like(np.zeros((num_new_particles, 1)), color)
        density = np.full_like(np.zeros((num_new_particles, 1)), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros((num_new_particles, 1)), pressure if pressure is not None else 0.)

        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)

    def add_solids(self,
            coord_file,
            material,
            color=0xecc89b,
            density=None,
            pressure=None,
            velocity=None):

        with open(coord_file, 'rb') as f:
                    a = np.load(f, allow_pickle=True)
        num_new_particles = a.shape[0]
        new_positions = a   / 15
        new_positions[:, 0] -= 11.5
        new_positions[:, 1] -= 11.5
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros((num_new_particles, 1)), material)
        color = np.full_like(np.zeros((num_new_particles, 1)), color)
        density = np.full_like(np.zeros((num_new_particles, 1)), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros((num_new_particles, 1)), pressure if pressure is not None else 0.)

        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)



    
    def dump(self):
        return {
            'position': self.x,
            'velocity': self.v,
            'material': self.material,
            'color': self.color,
            'pressure': self.pressure,
            'density': self.density
        }


if __name__ == "__main__":
    print('############test###########')
    res = (10, 10)
    grid_size = 2
    grid_num = np.ceil(np.array(res) / grid_size).astype(int)
    grid_particles_num = np.zeros((grid_num), dtype = np.int8)
    grid_particles_num[(1, 1)] = 1
    support_radius = 2
    ps = ParticleSystem(res)
    pos = np.array([1, 0.000001])
    cell = ps.pos_to_index(pos)
    iscell = ps.is_valid_cell(cell)


    x = np.array([
        [0.1, 0.1],
        [0.1, 0.2],
        [0.2, 0.3],
        [1.1, 2.3],
        [4.6, 7.6],
        [8.9, 9.1],
        [9.0, 9.3],
        [8.2, 9.2]
    ])
    particle_num = len(x)
    grid_particles_num = np.zeros((grid_num), dtype = np.int8)
    grid_particles = dict()
    def pos_to_index(pos):
        "convert particle positions to grid indices in (i, j)"
        cell = np.floor((pos / grid_size)).astype(np.int8)
        return cell

    def is_valid_cell(cell):
        flag = True
        for d in range(2):
            flag = flag and (0 <= cell[d] < grid_num[d])
        return flag


    def allocate_particles_to_grid():
        """match each particle to grid indices"""
        for p in range(particle_num):                          #loop over all particles
            cell = pos_to_index(x[p])                     #fetch position to grid indices in (i, j)
            if str(cell) not in grid_particles:
                grid_particles[str(cell)] = [p]
            else:
                grid_particles[str(cell)].append(p)                   #allocate the particle id to grid particles lists
            grid_particles_num[cell[0], cell[1]] += 1
    allocate_particles_to_grid()

    particle_neighbors = dict()
    particle_neighbors_num = np.zeros((particle_num, 1),dtype = np.int8)
    def search_neighbors():
        for p_i in range(particle_num):
            center_cell = pos_to_index(x[p_i])
            for x_offset in range(-1, 2):
                for y_offset in range(-1, 2):
                    cell[0] = center_cell[0] + x_offset
                    cell[1] = center_cell[1] + y_offset
                    if is_valid_cell(cell):
                        for j in range(grid_particles_num[cell[0], cell[1]]):
                            p_j = grid_particles[str(cell)][j]

                            distance = np.linalg.norm(x[p_i] - x[p_j])
                            if p_i != p_j and distance < support_radius:
                                if str(p_i) not in particle_neighbors:
                                    particle_neighbors[str(p_i)] = [p_j]
                                else:
                                    particle_neighbors[str(p_i)].append(p_j)
                                particle_neighbors_num[p_i]+= 1

            if str(p_i) not in particle_neighbors:
                particle_neighbors[str(p_i)] = []
    search_neighbors()
    print(particle_neighbors)
    print(particle_neighbors_num)


    #     ######Add particles######
    # particle_num = 0
    # particle_radius = 0.05
    # x = np.zeros((particle_num,  2), dtype = np.float64)                 #particles positions
    # v = np.zeros(( particle_num,  2), dtype = np.float64)                 #particles velocities
    # density = np.zeros(( particle_num, 1), dtype = np.float64)           #particles densities
    # pressure = np.zeros(( particle_num, 1), dtype = np.float64)          #particles pressures                                         
    # material = np.zeros(( particle_num, 1), dtype = np.int8)       #particles materials
    # color = np.zeros((particle_num, 1), dtype = np.int8)          #particle colors

    # def add_particle(p, x, v, density, pressure, material, color):
    #     x[p] = x
    #     v[p] = v
    #     density[p] = density
    #     pressure[p] = pressure
    #     material[p] = material
    #     color[p] = color


    # def add_particles(new_particles_num,
    #                     new_particles_positions,
    #                     new_particles_velocity,
    #                     new_particle_density,
    #                     new_particle_pressure,
    #                     new_particles_material,
    #                     new_particles_color):
    #     global particle_num, v, x, density, pressure, material, color

    #     for p in range(new_particles_num):
    #         v = np.append(v, new_particles_velocity[p].reshape((1, 2)), axis = 0)
    #         x = np.append(x, new_particles_positions[p].reshape((1, 2)), axis = 0)
            
    #         density = np.append(density, new_particle_density[p].reshape((1, 1)), axis = 0)
    #         pressure = np.append(pressure, new_particle_pressure[p].reshape((1, 1)), axis = 0)
    #         material = np.append(material, new_particles_material[p].reshape((1, 1)), axis = 0)
    #         color = np.append(color, new_particles_color[p].reshape((1, 1)), axis = 0)
    #     particle_num += new_particles_num

    # def add_cube(
    #             lower_corner,
    #             cube_size,
    #             material,
    #             color=0xFFFFFF,
    #             density=None,
    #             pressure=None,
    #             velocity=None):

    #     num_dim = []
    #     for i in range(2):
    #         num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i],  particle_radius))

    #     num_new_particles = reduce(lambda x, y: x * y, [len(n) for n in num_dim])

    #     new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)
    #     new_positions = new_positions.reshape(-1,reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
    #     print("new position shape ", new_positions.shape)
    #     if velocity is None:
    #         velocity = np.full_like(new_positions, 0)
    #     else:
    #         velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

    #     material = np.full_like(np.zeros((num_new_particles, 1)), material)
    #     color = np.full_like(np.zeros((num_new_particles, 1)), color)
    #     density = np.full_like(np.zeros((num_new_particles, 1)), density if density is not None else 1000.)
    #     pressure = np.full_like(np.zeros((num_new_particles, 1)), pressure if pressure is not None else 0.)

    #     add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)

    # add_cube(lower_corner=[3, 3],    #initial position
    #         cube_size=[2.0, 3.0],   #initial shape
    #         velocity=[0.0, 0.0],  #initial velocity
    #         density=1000.0,         #particle density
    #         color=0xecc89b,         #visualisation color
    #         material=0)

    # add_cube(lower_corner=[3, 3],    #initial position
    #         cube_size=[2.0, 3.0],   #initial shape
    #         velocity=[0.0, 0.0],  #initial velocity
    #         density=1000.0,         #particle density
    #         color=0xecc89b,         #visualisation color
            # material=0)


    # x = np.zeros((0,  2), dtype = np.float64) 
    # print(x.shape)
    # y = np.array([0, 0]).reshape((1, 2))  
    # print(y.shape)
    # x = np.append(x, y, axis=0)
    # print(x)


    # import numpy as np
    # from particle_system import ParticleSystem
    # from wcsph import WCSPHSolver

    # #instanciate particle system (the main window for particles and grids interactions)
    # ps = ParticleSystem((512, 512)) 

    # #add cude to the particle system
    # ps.add_cube(lower_corner=[4, 4],    #initial position
    #             cube_size=[0.2, 0.4],   #initial shape
    #             velocity=[0.0, -10.0],  #initial velocity
    #             density=1000.0,         #particle density
    #             color=0x68b2f7,         #visualisation color
    #             material=1)
    # wcsph_solver = WCSPHSolver(ps) 
    # wcsph_solver.step()

    # particle_info = ps.dump()
    # particle_position = particle_info['position']
    # print(particle_position)