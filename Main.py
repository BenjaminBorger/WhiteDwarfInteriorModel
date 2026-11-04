from helpers.starDatabase import obj, tests, endpoint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import G, c, hbar, m_e, m_n
from astropy.constants import M_sun
import helpers.pdeSolver as pdeSolver
from matplotlib.colors import LinearSegmentedColormap
#from pdeSolver import gammaNonRel,gammaRel,K_nonrel,K_rel,triggerPressure
from helpers.pdeSolver import solveWithSwitch,solveWithoutSwitch, find_eps_c,delete_last_line
###Problem at the moment is that the radius it calculates is differnt to the radius I give it. 
import shutil
import pandas as pd
import os
from matplotlib.colors import LogNorm

class WhiteDwarf:
    def __init__(self, index, rPoints, mPoints, pPoints):
        self.index = index
        self.rPoints = rPoints
        self.mPoints = mPoints
        self.pPoints = pPoints
        



def loadbar(i, total, prefix="", suffix="", decimals=1, fill='>'):
    pct = f'{100 * i / total:.{decimals}f}%'
    length = max(1, shutil.get_terminal_size().columns - len(f'{prefix} || {pct} {suffix}') - 1)
    n = length * i // total
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
    f"e_val_to_{endpoint}_w_{objectsLength}_iterations.csv"
)
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def run_stars(use_cache=True):
    cache = pd.DataFrame(columns=["epsilon"]).rename_axis("Star")

    if use_cache and os.path.exists(CSV_FILE):
        print(f'Cache found of epsilon values up to {objectsLength} for masses below {endpoint}')
        cache = pd.read_csv(CSV_FILE, index_col="Star")
        print("Cache read...")
    else:
        cache = pd.DataFrame(columns=["epsilon"]).rename_axis("Star")
        print(f'No cache found with these iterations. Creating new cache with {objectsLength} iterations')
        print('Solving...')
    
    
    
    rows = [] #administrative (constant things)
    starProfiles = [] #make these a collection of objects (stars)
    limit_mass = -np.inf
    loadbar(0, objectsLength, suffix="complete")

    for i in range(objectsLength): #each star in database
        loadbar(i + 1, objectsLength, suffix="complete")

        # use the already tabulated value if there is one, otherwise compute it
        if i in cache.index:
            eps_c = cache.loc[i, "epsilon"]
        else:
            eps_c = find_eps_c(i, rel=False)

        if eps_c is None or np.isnan(eps_c):
            break

        r, m, radius_grid, states = solveWithSwitch(eps_c, allpoints=True)
        limit_mass = max(limit_mass, m)


        ps = states[1]
        
        ###make it so that it shifts from one colour to another based on whether 
        # pressure is relatavistic at that point have an expression for pressure somewhere?
        
       # plt.plot(np.asarray(states[0]) / M_sun.value, np.asarray(radius_grid) / 1000)

        
        starProfiles.append(WhiteDwarf(i,radius_grid/1000,states[0]/M_sun,states[1]))
        

        rows.append({
            "Star": i,
            "epsilon": eps_c,
            "RealMass_Msun": obj[i][1],
            "ModelFinalMass_Msun": m / M_sun.value,
            "ModelFinalRadius_km": r / 1000,
        })

    results = pd.DataFrame(rows).set_index("Star")
    #pointsMRP = pd.DataFrame(starProfile)
    results[["epsilon"]].to_csv(CSV_FILE)      #save after loop
    return results, limit_mass, starProfiles  #pointsMRP

results, limit_mass, profiles = run_stars()


fig,ax = plt.subplots()



#######################################No Switch################################################
p_values = np.logspace(21,23, 60) # pascals
NRMs, NRRs = [], []
RMs,RRs = [], []
print()
print("Calculating rough approximation of polytropes...")
for p0 in p_values:
    NonRelR, NonRelM= solveWithoutSwitch(p0, rel=False)
    pRR, pRM = solveWithoutSwitch(p0, rel=True)
    NRRs.append(NonRelR)
    NRMs.append(NonRelM)
    RRs.append(pRR)
    RMs.append(pRM)

ax.plot(np.array(NRMs)/M_sun.value, np.array(NRRs)/1000)
ax.plot(np.array(RMs)/M_sun.value, np.array(RRs)/1000)
#plt.xscale('log')
#plt.yscale('log')
#######################################No Switch End ################################################
# one shared norm for every star
all_p = np.concatenate([np.asarray(p.pPoints) for p in profiles])
all_p = all_p[all_p > 0]
sorted_p = np.sort(all_p)
from matplotlib.colors import TwoSlopeNorm
from helpers.pdeSolver import triggerPressure   # whatever name/units you use
def rank_norm(p):
    # fraction of all points with pressure below p, in [0, 1]
    return np.searchsorted(sorted_p, p) / len(sorted_p)
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

#norm = Normalize(0,1) # center around center pressure
norm = TwoSlopeNorm(vcenter=rank_norm(triggerPressure), vmin=0, vmax=1) # centre around trigger pressure



#unpack starprofiles into seperate classes and plot each
for i in range(len(profiles)):
    radiiProfile = profiles[i].rPoints
    massProfile = profiles[i].mPoints
    pressureProfile = profiles[i].pPoints
    #print(massProfile)
    #print(radiiProfile)
    points = np.array([massProfile, radiiProfile]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    #norm = plt.Normalize(pressureProfile.min(), pressureProfile.max())
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    mid_p = 0.5 * (pressureProfile[:-1] + pressureProfile[1:])
    lc.set_array(rank_norm(mid_p))
    line = ax.add_collection(lc)


    #ax.plot(massProfile,radiiProfile)


print(f"Limiting Mass: {limit_mass/M_sun.value} solar masses")
ax.set_title('Rel Rad Mass Relation')
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(norm=norm, cmap='coolwarm')
fig.colorbar(ScalarMappable(norm=norm, cmap='coolwarm'), ax=ax, label='Pressure percentile')
#plt.legend()
ax.set_xlabel('Mass (Solar Masses)')
ax.set_ylabel('Radius (Km)') 
ax.set_xlim(0,1.46)
ax.set_ylim(0,20000)
plt.show()