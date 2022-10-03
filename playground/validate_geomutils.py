import gym_auv.utils.geomutils as geom
import numpy as np

for i in range(4):
    psi = np.pi / 2 * i
    Rz = geom.Rz(psi)
    print(psi, "\n", Rz)
