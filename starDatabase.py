#radius(km),mass(M_sun)
import numpy as np
#obj = np.array([[7854,0.67]]) #
#obj = np.array([[6000,1],[4000,1.2],[1000,1.1],[1270,1.3],[1260,1.2],[1200,1.35]]) #,[7,1.2],[10,1.1],[12.7,1.3],[12.4,1.2],[12.7,1.356]
import numpy as np

tests = 1000



def make_objs(numTests=tests, start=1/500, end=2, radius_km=7000):
    masses = start + end - np.geomspace(start, end, numTests)  # solar masses
    masses = np.sort(masses)
    return np.column_stack([np.full(numTests, radius_km), masses])  # shape (numTests, 2)
    
obj = make_objs()
