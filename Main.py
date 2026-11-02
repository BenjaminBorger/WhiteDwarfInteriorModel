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
import pandas as pd
import os



def loadbar(i, total, prefix="", suffix="", decimals=1, fill='>'):
    pct = f'{500 * i / total:.{decimals}f}%'
    length = max(1, shutil.get_terminal_size().columns - len(f'{prefix} || {pct} {suffix}') - 1)
    n = length * i*5 // total
    print(f'\r{prefix} |{fill * n}{"-" * (length - n)}| {pct} {suffix}', end='\r' if i < total else '\n')
#plotting
#could maybe add some sort of circle that has the pressure so as the radius goes outwards, it changes colour (bit naff)

#Sean Rs Idea - make iterations more spaced out until the end where they get tighter - run faster

objectsLength = len(obj)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CSV_FILE = os.path.join(
    CACHE_DIR,
    f"epsilon_values_{objectsLength}_iterations.csv"
)


def run_stars(use_cache=True):

    if use_cache and os.path.exists(CSV_FILE): 
        cache = pd.read_csv(CSV_FILE, index_col="Star")
    else:
        cache = pd.DataFrame(columns=["epsilon"]).rename_axis("Star")

    rows = []
    limit_mass = -np.inf
    loadbar(0, objectsLength, suffix="complete")

    for i in range(objectsLength):
        loadbar(i + 1, objectsLength, suffix="complete")

        # use the already tabulated value if there is one, otherwise compute it
        if i in cache.index:
            eps_c = cache.loc[i, "epsilon"]
        else:
            eps_c = find_eps_c(i, rel=False)

        if eps_c is None or np.isnan(eps_c):
            print(f"\nSkipping object {i}: no valid epsilon_c")
            break

        r, m, radius_grid, states = solveWithSwitch(eps_c, allpoints=True)
        limit_mass = max(limit_mass, m)
        plt.plot(np.asarray(states[0]) / M_sun.value,
                 np.asarray(radius_grid) / 1000)

        rows.append({
            "Star": i,
            "epsilon": eps_c,
            "RealMass_Msun": obj[i][1],
            "RealRadius_km": obj[i][0],
            "ModelMass_Msun": m / M_sun.value,
            "ModelRadius_km": r / 1000,
        })

    results = pd.DataFrame(rows).set_index("Star")
    results[["epsilon"]].to_csv(CSV_FILE)      #save after loop
    return results, limit_mass

results, limit_mass = run_stars()
print(results)

p_values = np.logspace(21,23, 60) # pascals

NRMs, NRRs = [], []
RMs,RRs = [], []
for p0 in p_values:
    NonRelR, NonRelM= solveWithoutSwitch(p0, rel=False)
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