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

class WhiteDwarf: # creates an object: white dwarf which lists will be stored contianing each value. needed for plotting later. 
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

objectsLength = len(obj) # length of iterations planned. 
import os  #for files
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # folder currently in

CACHE_DIR = os.path.join(BASE_DIR, "cache") #folder with cache 
os.makedirs(CACHE_DIR, exist_ok=True) #would make directory if didnt exis

CSV_FILE = os.path.join(
    CACHE_DIR,
    f"e_val_to_{endpoint}_w_{objectsLength}_iterations.csv"
)#links csv file with eps values to be created and read from cache directory
def is_float(s): # little function i dont really need
    try:
        float(s)
        return True
    except ValueError:
        return False

def run_stars(use_cache=True): # meat of the bloody matter
    cache = pd.DataFrame(columns=["epsilon"]).rename_axis("Star") # establish what is cached information - epsilon values
    print("Searching Cache...")
    if use_cache and os.path.exists(CSV_FILE): # if im allowed to use cache and there exists the csv file WITH THE SAME NAME! use that
        print(f'Cache found of epsilon values up to {objectsLength} for masses below {endpoint} solar masses.#')
        cache = pd.read_csv(CSV_FILE, index_col="Star")# read the file
        print("Reading Cache...")
    else: # if no cache with same name exists
        cache = pd.DataFrame(columns=["epsilon"]).rename_axis("Star") # create new dataframe 
        print(f'No cache found with these iterations. Creating new cache with {objectsLength} iterations')
        print('Solving...')
    
    
    
    rows = [] #administrative (constant things)
    starProfiles = [] #make these a collection of objects (stars)
    limit_mass = -np.inf # infinitely small limit mass so everything wil be bigger. Important to have this outside loop or it would reset
    loadbar(0, objectsLength, suffix="complete")

    for i in range(objectsLength): #each star in database
        loadbar(i + 1, objectsLength, suffix="complete")

        if i in cache.index: #if we have a cache index for this star
            eps_c = cache.loc[i, "epsilon"] # epsilon c equals the thing in that table
        else:
            eps_c = find_eps_c(i, rel=False) # find eps c if not in table (if no cache) this is the bit annoying ignore rel = False. been taken out

        if eps_c is None or np.isnan(eps_c): # as soon as we hit a wall and cant compute any more, stop the loop
            break #help help

        r, m, radius_grid, states = solveWithSwitch(eps_c, allpoints=True) # solve the equation for this star and return final radius r final mass m a grid of all the radii and the states [mass,pressure]
        limit_mass = max(limit_mass, m) # largest mass is the maximum between the largest mass up to now and the largest mass just returned


        ps = states[1] # why is this here
        
        ###make it so that it shifts from one colour to another based on whether 
        # pressure is relatavistic at that point have an expression for pressure somewhere?
        
       # plt.plot(np.asarray(states[0]) / M_sun.value, np.asarray(radius_grid) / 1000)

        
        starProfiles.append(WhiteDwarf(i,radius_grid/1000,states[0]/M_sun,states[1])) # add object to starProfiles with its index, all the raddi and corresponding masses and pressures
        

        rows.append({ # static quantaties in a dictionary
            "Star": i,
            "epsilon": eps_c,
            "RealMass_Msun": obj[i][1],
            "ModelFinalMass_Msun": m / M_sun.value,
            "ModelFinalRadius_km": r / 1000,
        })

    results = pd.DataFrame(rows).set_index("Star") # 
    #pointsMRP = pd.DataFrame(starProfile)
    results[["epsilon"]].to_csv(CSV_FILE)      #save epsilon file after loop
    return results, limit_mass, starProfiles  # return the static stuff, the largest mass and the list of white dwarf objects with all the points for plotting

results, limit_mass, profiles = run_stars() # call the function!


fig,ax = plt.subplots()#create subplots



#######################################No Switch################################################
p_values = np.logspace(21,23, 60) # pascals. test values of presure
NRMs, NRRs = [], [] # Non Rel Masses and Non Rel Radii
RMs,RRs = [], [] # Rel Masses and Rel Radii
print() # newline
print("Calculating rough approximation of polytropes...")
for p0 in p_values: # for each P value
    NonRelR, NonRelM= solveWithoutSwitch(p0, rel=False) # get the Rs and Ms with imputted p0
    pRR, pRM = solveWithoutSwitch(p0, rel=True)
    NRRs.append(NonRelR)
    NRMs.append(NonRelM)
    RRs.append(pRR)
    RMs.append(pRM)

ax.plot(np.array(NRMs)/M_sun.value, np.array(NRRs)/1000)#plot
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
cbar = fig.colorbar(sm, ax=ax, label='Pressure (Pa)')

ticks = np.linspace(0, 1, 6)                    
cbar.set_ticks(ticks)



cbar.set_ticklabels([f'{np.quantile(sorted_p, t):.1e}' for t in ticks])
#plt.legend()
ax.set_xlabel('Mass (Solar Masses)')
ax.set_ylabel('Radius (Km)') 
ax.set_xlim(0,1.46)
ax.set_ylim(0,20000)
plt.show()