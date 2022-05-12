import numpy as np
from sph_base import SPHBase
import time
import numba
from numba import int32, float32    # import the types
from numba.experimental import jitclass



class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.exponent = 7                               #gamma, rquired for state equation
        self.stiffness = 100.0                         #B, required for state equation

        self.d_velocity = np.zeros((self.ps.particle_num, 2))



    def compute_densities(self):
        for p_i in range(self.ps.particle_num):                                             #loop over all particles
            x_i = self.ps.x[p_i]                                                            #find the centre particle position
            self.ps.density[p_i] = 0.0                                                      #initialise particle density
            #print(self.ps.particle_neighbors_num[p_i])
            for j in range(int(self.ps.particle_neighbors_num[p_i])):                            #loop over all neighbor particles
                p_j = self.ps.particle_neighbors[str(p_i)][j]                                    #find neighboring particle
                x_j = self.ps.x[p_j]                                                        #find neighboring particle postion
                x_diff = x_i - x_j
                x_norm = (x_diff[0]**2 + x_diff[1]**2)**(1/2)
                self.ps.density[p_i] += self.mass * self.cubic_kernel(x_norm)  #accumulate neighboring particles densities to centre particle density
            self.ps.density[p_i] = self.ps.density[p_i]


    def compute_pressure_forces(self):
        for p_i in range(self.ps.particle_num):                
            self.ps.density[p_i] = np.maximum(self.ps.density[p_i], self.density_0)                                                 #cap minimum particle densities to reference density
            self.ps.pressure[p_i] = self.stiffness * ((self.ps.density[p_i] / self.density_0)**self.exponent - 1.0)             #calculate pressure based on tait equation
        for p_i in range(self.ps.particle_num):                                                                                 #loop over all particles
            x_i = self.ps.x[p_i]                                                                                                #find the centre particle position
            d_v = np.array([0.0, 0.0])                                                                                          #initialise particle pressure force
            for j in range(int(self.ps.particle_neighbors_num[p_i])):                                                                #loop over all neighbor particles
                p_j = self.ps.particle_neighbors[str(p_i)][j]                                                                        #find neighboring particle
                x_j = self.ps.x[p_j]                                                                                            #find neighboring particle postion
                d_v += self.pressure_force(p_i, p_j, x_i-x_j)                                                                   #accumulate Pressure forces
            self.d_velocity[p_i] += d_v 


    def compute_non_pressure_forces(self):
        for p_i in range(self.ps.particle_num):                                               #loop over all particles
            x_i = self.ps.x[p_i]                                                              #find the centre particle position
            d_v = np.array([0.0, 0.0])                                                        #initialise particle non pressure force
            # d_v[self.ps.dim-1] = self.g                                                     #add gravity in y direction
            for j in range(int(self.ps.particle_neighbors_num[p_i])):                              #loop over all neighbor particles
                p_j = self.ps.particle_neighbors[str(p_i)][j]                                       #find neighboring particle
                x_j = self.ps.x[p_j]                                                          #find neighboring particle postion
                d_v += self.viscosity_force(p_i, p_j, x_i - x_j)                              #accumulate non Pressure forces
            self.d_velocity[p_i] = d_v


    def boundary_condition(self):
        for p_i in range(self.ps.particle_num):
            if self.ps.material[p_i] != self.ps.material_fluid:
                if  self.ps.x[p_i][1] > 1.5:
                    self.ps.v[p_i][0] = 10
                    

    def advect(self):
        for p_i in range(self.ps.particle_num):
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt * self.d_velocity[p_i]                         #update velocity
                self.ps.x[p_i] += self.dt * self.ps.v[p_i]                               #update position

    def substep(self):
        # timer1 = time.time()
        self.boundary_condition()
        # timer2= time.time()
        self.compute_densities()
        # timer3= time.time()
        self.compute_non_pressure_forces()
        # timer4= time.time()
        self.advect()
        # timer5= time.time()
        self.compute_densities()
        # timer6= time.time()
        self.compute_pressure_forces()
        # timer7= time.time()
        self.advect()
        # timer8= time.time()

        # print('boundary_condition(): ', -(timer1 - timer2))
        # print('compute_densities(): ', -(timer2 - timer3))
        # print('compute_non_pressure_forces(): ', -(timer3 - timer4))
        # print('advect(): ', -(timer4 - timer5))
        # print('compute_densities(): ', -(timer5 - timer6))
        # print('compute_pressure_forces(): ', -(timer6 - timer7))
        # print('advect(): ', -(timer7 - timer8))

    @numba.jit() 
    def step(self):
        # start = time.time()
        self.ps.initialize_particle_system() #search neighbours
        end = time.time()
        # self.substep()                       #compute new positions and velocities
        # print('neighbour_search: ', -(start - end))