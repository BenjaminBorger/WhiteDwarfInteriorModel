from starDatabase import obj
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import G, c, hbar, m_e, m_n
from astropy.constants import M_sun
import pdeSolver
from matplotlib.colors import LinearSegmentedColormap
#from pdeSolver import gammaNonRel,gammaRel,K_nonrel,K_rel,triggerPressure
from pdeSolver import solve, find_eps_c,delete_last_line
###Problem at the moment is that the radius it calculates is differnt to the radius I give it. 
import shutil

def loadbar(i, total, prefix="", suffix="", decimals=1, fill='>'):
    pct = f'{100 * i / total:.{decimals}f}%'
    length = max(1, shutil.get_terminal_size().columns - len(f'{prefix} || {pct} {suffix}') - 1)
    n = length * i*5 // total
    print(f'\r{prefix} |{fill * n}{"-" * (length - n)}| {pct} {suffix}', end='\r' if i < total else '\n')
#plotting
#could maybe add some sort of circle that has the pressure so as the radius goes outwards, it changes colour (bit naff)

limit_mass = -np.inf      # largest finite mass seen across all objects
limit_radius = None  
objectsLength = len(obj)


#Sean Rs Idea - make iterations more spaced out until the end where they get tighter - run faster


loadbar(0, objectsLength, suffix = "complete")
FMasses = []
FRadii = []
for i in range(objectsLength):
    #delete_last_line()
    loadbar(i+1, objectsLength, suffix = "complete")
    
    realMass = obj[i][1]*M_sun.value #import the real mass and radius for comparison
    realRadius = obj[i][0]*1000
    #print(realMass)
    #print(realRadius)
    iterEps_C = find_eps_c(i, rel=False) # 
    if iterEps_C is None: # if theres no good value of epsilon
        print(f"Skipping object {i}: no valid epsilon_c")
        break
        #continue
    r, m, radius_grid, states = solve(iterEps_C,allpoints=True)
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
    
    plt.plot(mass_profile/M_sun.value,radius_grid/1000)
    #plt.show()


#plt.loglog(FMasses,FRadii)



#plt.axhline(y=limit_mass, color='r', linestyle='--',label=f'Limiting mass = {limit_mass:.3e} kg')

print(f"Limiting Mass: {limit_mass/M_sun.value} solar masses")
plt.title('Rel Rad Mass Relation')
#plt.legend()
plt.xlabel('Mass (Solar Masses)')
plt.ylabel('Radius (Km)') 
plt.xlim(0,1.5)
plt.ylim(0,15000)
plt.show()


def plot_pressure():
    r = np.asarray(radius_grid, dtype=float).ravel()
    p = np.asarray(pressure_profile, dtype=float).ravel()

    # np.interp needs increasing x values
    order = np.argsort(r)
    r, p = r[order], p[order]


    print("radius range:", r.min(), r.max())
    print("pressure range:", p.min(), p.max())

    # Normalize radius to 0..1
    r_norm = (r - r.min()) / (r.max() - r.min())

    n = 500
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    P = np.interp(R, r_norm, p)
    P_masked = np.ma.masked_where(R > 1, P)


    cmap = LinearSegmentedColormap.from_list("pressure", ["blue","white"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    
    im = ax.imshow(
        P_masked,
        extent=(-1, 1, -1, 1),
        origin="lower",
        cmap=cmap,
        vmin=p.min(),
        vmax=p.max(),
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pressure")





