from starDatabase import obj, tests
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import G, c, hbar, m_e, m_n
from astropy.constants import M_sun
import pdeSolver
from matplotlib.colors import LinearSegmentedColormap
#from pdeSolver import gammaNonRel,gammaRel,K_nonrel,K_rel,triggerPressure
from pdeSolver import solveWithSwitch,solveWithoutSwitch, find_eps_c,delete_last_line
###Problem at the moment is that the radius it calculates is differnt to the radius I give it. 
import shutil



def loadbar(i, total, prefix="", suffix="", decimals=1, fill='>'):
    pct = f'{500 * i / total:.{decimals}f}%'
    length = max(1, shutil.get_terminal_size().columns - len(f'{prefix} || {pct} {suffix}') - 1)
    n = length * i*5 // total
    print(f'\r{prefix} |{fill * n}{"-" * (length - n)}| {pct} {suffix}', end='\r' if i < total else '\n')
#plotting
#could maybe add some sort of circle that has the pressure so as the radius goes outwards, it changes colour (bit naff)




#Sean Rs Idea - make iterations more spaced out until the end where they get tighter - run faster

objectsLength = len(obj)

FMasses = []
FRadii = []
    
def SwitchedGraph():
    
    limit_mass = -np.inf      # largest finite mass seen across all objects
    limit_radius = None  
    objectsLength = len(obj)
    loadbar(i+1, objectsLength, suffix = "complete")
    
    realMass = obj[i][1]*M_sun.value #import the real mass and radius for comparison
    realRadius = obj[i][0]*1000
    #print(realMass)
    #print(realRadius)
    iterEps_C = find_eps_c(i, rel=False) # 
    if iterEps_C is None: # if theres no good value of epsilon
        print()
        print(f"Skipping object {i}: no valid epsilon_c")
        return None
        #continue
    r, m, radius_grid, states = solveWithSwitch(iterEps_C,allpoints=True)
    FMasses.append(m)
    FRadii.append(r)
    #print(f"modeled: {r/1000 :.2f} km , input: {realRadius/1000 :.2f} km , "
        #f"difference: {((r - realRadius)/1000):.2f} km")
    #mass_profile = states[0]
    #pressure_profile = states[1]
    mass_profile = np.asarray(states[0])
    pressure_profile = np.asarray(states[1])
    radius_grid = np.asarray(radius_grid)

    
    valid = np.where(np.isfinite(mass_profile))[0] # last index where the mass is still a real number
    if len(valid) > 0:
        j = valid[-1]
        if mass_profile[j] > limit_mass:
            limit_mass = mass_profile[j]
            limit_radius = radius_grid[j]
    
    return mass_profile/M_sun.value,radius_grid/1000,limit_mass


    #plt.show()


#plt.loglog(FMasses,FRadii)
loadbar(0, objectsLength, suffix = "complete")
for i in range(objectsLength):
    
    #delete_last_line() 
    try: 
        ms,rs,limit_mass = SwitchedGraph()
    except:
        break
    
    plt.plot(ms,rs)

#plt.axhline(y=limit_mass, color='r', linestyle='--',label=f'Limiting mass = {limit_mass:.3e} kg')





p_values = np.logspace(21,23, 60) # pascals

NRMs, NRRs = [], []
RMs,RRs = [], []
for p0 in p_values:
    NonRelR, NonRelM = solveWithoutSwitch(p0, rel=False)
    pRR, pRM = solveWithoutSwitch(p0, rel=True)
    NRRs.append(NonRelR)
    NRMs.append(NonRelM)
    RRs.append(pRR)
    RMs.append(pRM)

plt.plot(np.array(NRMs)/M_sun.value, np.array(NRRs)/1000)
plt.plot(np.array(RMs)/M_sun.value, np.array(RRs)/1000)
#plt.xscale('log')
#plt.yscale('log')


print(f"Limiting Mass: {limit_mass/M_sun.value} solar masses")
plt.title('Rel Rad Mass Relation')
#plt.legend()
plt.xlabel('Mass (Solar Masses)')
plt.ylabel('Radius (Km)') 
plt.xlim(0,1.46)
plt.ylim(0,20000)
plt.show()