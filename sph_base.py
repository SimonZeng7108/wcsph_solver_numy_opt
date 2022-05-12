import numpy as np

class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system                   #instantiate particle system
        self.g = 0.0                                #gravitational force, set to zero in cavity case; requried for non-pressure force solver
        self.viscosity = 0.001                      #Dynamic viscosity of water @20 degrees, Ns/m2; requried for non-pressure force solver
        self.density_0 = 1000                       #Density of water; required for mass calculation and state equation for preesure
        self.mass = self.ps.m_V * self.density_0    #Particle mass; required for all approximations
        self.dt = 1e-4                              #Times step; required for time integration solver

    # def cubic_kernel(self, r_norm):
    #     "Wendland C2 kernel"
    #     h = self.ps.support_radius
    #     theta = 7/(4*np.pi*h**self.ps.dim)
    #     q = r_norm / h

    #     if 0 <= q <= 2:
    #         w = theta * (1-q/2)**4*(2*q+1)
    #     else:
    #         w = 0.0

    #     return w

    # def cubic_kernel_derivative(self, r):
    #     "derivative of Wendland C2 kernel"
    #     h = self.ps.support_radius
    #     theta = - 7 * 4 /(4*np.pi*h**self.ps.dim)
    #     r_norm = np.linalg.norm(r)
    #     q = r_norm / h
    #     grad_q = r / (r_norm * h)
    #     if 0 <= q <= 2:
    #         w = theta * (1 - q / 2) ** 3 * grad_q
    #     else:
    #         w = np.array([0.0, 0.0])


    #     return w

    def cubic_kernel(self, r_norm):
        "cubic B_spline kernel"
        h = self.ps.support_radius
        h = 0.12
        theta = 40/(7*np.pi*h**self.ps.dim)
        q = r_norm / h

        if 0 <= q <= 0.5:
            w = theta * (6 * (q**3 - q**2) + 1)
        elif 0.5 <= q <= 1:
            w = theta * (2 * (1 - q)**3)
        else:
            w = 0.0

        return w

    def cubic_kernel_derivative(self, r):
        "derivative of cubic B_spline kernel"
        h = self.ps.support_radius
        h = 0.12
        theta = 40 * 6 /(7*np.pi*h**self.ps.dim)
        r_norm = (r[0]**2 + r[1]**2)**(1/2)
        q = r_norm / h
        grad_q = r / (r_norm * h)
        if 0 <= q <= 0.5:
            w = theta *(3 * q**2 - 2 * q) * grad_q
        elif 0.5 <= q <= 1:
            w = - theta * (1 - q)**2 *  grad_q
        else:
            w = np.array([0.0, 0.0])


        return w


    def viscosity_force(self, p_i, p_j, r):
        "viscosity force in Predictor term"
        v_xy = np.dot((self.ps.v[p_i] - self.ps.v[p_j]), r)

        r_norm = (r[0]**2 + r[1]**2)**(1/2)
        viscosity_force = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass /  self.ps.density[p_j]) \
                            * v_xy / (r_norm**2+ 1e-10) * self.cubic_kernel_derivative(r)
        return viscosity_force

    def pressure_force(self, p_i, p_j, r):
        'pressure force in Corrector term'
        pressure_force = -self.mass* (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
        + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
        * self.cubic_kernel_derivative(r)
        return pressure_force

if __name__ == '__main__':
    class ParticleSystem:
        def __init__(self):
            self.dim = 2
            self.particle_radius = 0.05
            self.support_radius = self.particle_radius * 4.0
            self.m_V = np.pi * self.particle_radius**self.dim

            self.x = np.arange(0, 1, 0.001)
            self.y = np.ones_like(self.x) * 0.001
            self.r = np.ones((len(self.x), 2))
            self.r[:, 0] = self.x
            self.r[:, 1] = self.y

            self.density = np.ones((self.x.shape[0] , 1)) * 1000
            self.pressure = np.ones((self.x.shape[0] , 1)) * 1000
            np.random.seed(0)
            self.v = np.random.rand(self.x.shape[0] , 2)

    ps = ParticleSystem()
    sph_base = SPHBase(ps)

    p_i = 0
    p_j = 1
    r = ps.r
    for r in r:
        pressure_force = np.around(sph_base.pressure_force(p_i, p_j, r), 6)
        print(pressure_force)
    # for r in r:
