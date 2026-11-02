import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import G, c, hbar, m_e, m_n
from astropy.constants import M_sun
import sys
from starDatabase import obj
from numba import njit


def delete_last_line(): # got from stackoverflow https://stackoverflow.com/questions/44565704/how-to-clear-only-last-one-line-in-python-output-console
    # cursor up one line
    sys.stdout.write('\x1b[1A')
    # delete last line
    sys.stdout.write('\x1b[2K')


mu_e     = 2    #for carbon
gammaRel = 4/3
gammaNonRel = 5/3
K_nonrel = hbar**2/(15*np.pi**2*m_e) * (3*np.pi**2/(mu_e*m_n*c**2))**(5/3)
K_rel    = hbar*c/(12*np.pi**2)      * (3*np.pi**2/(mu_e*m_n*c**2))**(4/3)
#triggerPressure = ((m_e**4)*(c**5))/(12*(np.pi**2)*(hbar**3))

eps_switch = np.e**((np.log(K_rel)-np.log(K_nonrel))/(gammaNonRel-gammaRel))
triggerPressure = K_rel*eps_switch**gammaRel

'''so electrons become relatavistic approx when kf = me c
which is when pressure = ((m_e**4)*(c**5))/(12*(pi**2)*(hbar**3))
This will be called the trigger pressure '''
@njit #speeds it up
def f(r,y,K,gamma):
        M,P = y
        eps = (max(P, 0)/K)**(1/gamma) #inversion of polytope
        dm = 4*np.pi*r**2*eps/c**2  #first equation
        dP = -(G/(c**2*r**2))*(eps + P)*(M + 4*np.pi*r**3*P/c**2) \
             / (1 - 2*G*M/(r*c**2)) # relatavistic pressure curve
        return [dm,dP]

def solveWithSwitch(eps_c,allpoints = False): # eps_c is given through findepsc
    #delete_last_line()
    #if not allpoints: print(f"Central value of Epsilon Guess = {eps_c}")
    #else: print(f"Central value of Epsilon Final = {eps_c}")
    
    p0 = K_rel*eps_c**gammaRel #initial pressure polytrope

    def surface(r, y,K,gamma): # define a surface where pressure basically equals zero and stop there
        return y[1] - (1e-60 * min(p0, triggerPressure))                 
    surface.terminal = True
    surface.direction = -1


    def switch(r,y,K,gamma):
        return y[1]-triggerPressure
    switch.terminal = True
    switch.direction = -1


    r0=1 # initial radius of 1 meter to stop a div by zero error
    m0 = 4*np.pi/3*r0**3*eps_c/c**2 # initial mass which will basiclaly equal zero
    
    
    if p0<=triggerPressure:
        p0 = K_nonrel*eps_c**gammaNonRel 
        totalStarisNonRel = solve_ivp(f,(r0,10e9),[m0,p0],args = (K_nonrel,gammaNonRel),events = surface, rtol = 1e-8,atol = [m0*1e-10, p0*1e-12],dense_output= allpoints) #simultaneously solve the equations
        R = totalStarisNonRel.t_events[0][0]
        M = totalStarisNonRel.y_events[0][0][0] # takes the last mass of star
        rs = totalStarisNonRel.t # these are all the values of radius 
        ys = totalStarisNonRel.y # all the fvlaues of mass
    else:

        relPart = solve_ivp(f,(r0,10e9),[m0,p0],args = (K_rel,gammaRel),events = [switch,surface], rtol = 1e-8,atol = [m0*1e-10, p0*1e-12],dense_output= allpoints) #simultaneously solve the equations
    
        r_event = relPart.t_events[0][0] # Takes the last radius of centre 
        y_event = relPart.y_events[0][0] # takes the last mass of centre
        
        
        nonRelPart = solve_ivp(f,(r_event,10e9),y_event,args = (K_nonrel,gammaNonRel),events = surface, rtol = 1e-8,atol = y_event*10e-12,dense_output= allpoints) #simultaneously solve the equations
        R = nonRelPart.t_events[0][0] # Takes the last radius of star
        M = nonRelPart.y_events[0][0][0] # takes the last mass of star

        rs = np.concatenate([relPart.t,nonRelPart.t]) # combines the two points from rel to non rel
        ys = np.concatenate([relPart.y,nonRelPart.y],axis=1)


    if allpoints == True: return (R,M, rs, ys) # return all points if needed (for plot)
    else: return (R,M) # just return last masses for comparisom


def find_eps_c(iter,rel): # i took this out. rel is boolean True or false if true: relatavistic and if not, not relatavistic
    '''This function makes a guess of epsilon
    assuming constant density and then takes iterations of the pde solver and hones in 
    on the correct value of epsilon for each star. After that a real measurement can be made.'''
    
    
    realRadius = obj[iter, 0]*1000 # radius of example star in meters 
    realMass = obj[iter, 1]*M_sun.value # mass of example star kg
    
    K = K_rel
    gamma = 4/3

    #print("Radius: ",float(realRadius), "     Mass: ", float(realMass)) #check that all is ok thusfar
    
    epsilonGuess = (realMass/((4/3)*np.pi*realRadius**3))*(c**2) # guess assuming const density
    epsilonSpace = np.logspace(np.log10(epsilonGuess)-5,np.log10(epsilonGuess)+5,30) # defines an area which is reasonable for eps to be in

    iteratedRs = np.full(len(epsilonSpace), np.nan) # creates an array of not a numbers the length of epsilon space
    iteratedMs = np.full(len(epsilonSpace), np.nan)
    
    intervals= []#these will be the intervals in epsilonspace within which contain a crossover of sign
    for i in range(len(epsilonSpace)): # loop for each value in epsilon space(will loop 40 times)
        
        R,M = solveWithSwitch(epsilonSpace[i], allpoints=False) # solve the PDE for this value of epsilon and only return the finishing mass and radius
        iteratedRs[i] = R # add these returned values to thir respective arrays
        iteratedMs[i] = M

        if i == 0 or np.isnan(iteratedMs[i]) or np.isnan(iteratedMs[i-1]): #or np.isnan(iteratedRs[i-1]) or np.isnan(iteratedRs[i]): # if this si the first iteration,the mass is not a number or the previous mass is not a number, skip
            continue
        '''if there is a sign change between the previous mass differnce and this mass difference
        the real mass must be somewhere in the middle of these (where returned m-real m = 0)
        this is the epsilon the hone in on'''
        if  np.sign(iteratedMs[i-1] - realMass) != np.sign(iteratedMs[i] - realMass): #and np.sign(iteratedRs[i-1] - realRadius) != np.sign(iteratedRs[i] - realRadius) :
                intervals.append([epsilonSpace[i-1], epsilonSpace[i]]) #if one of these sign changes is found, we must add this interval as one of interest

   
    eps_C_Real = 0
    goodEpsis = [] #these will have a list of candidates for the initial epsilon value

    goodEpsisR = []
    for interval in intervals: # go through each interval
        def massDiffFunc(epsilonVal): #simply solves the equations with this value of epsilon and creates a function with m-realMass
            
            r,m = solveWithSwitch(epsilonVal,allpoints=False)
            
            return m-realMass

        def radiusDiffFunc(epsilonVal,allpoints=False): # not sure whether to compare radius as well
            r,m = solveWithSwitch(e)
            return r-realRadius

        goodEpsiM = brentq(massDiffFunc,interval[0],interval[1]) # hones in on the values of epsilon which return a zero for our defined function 
        #may return multiple zeroes thats why i need the list
        goodEpsis.append(goodEpsiM) # when found, add to list
        #goodEpsiM = brentq(radiusDiffFunc,interval[0],interval[1])

    if len(goodEpsis) == 0: print("No roots Found for object ",iter) # if there are no roots, print this:problem

    bestE, bestErr = None, np.inf # makes our bestE none and our best error infinitely large
    for e in goodEpsis: # for each good espilon 
        '''for each good espsilon, check whether the error is better than the last one 
        if its the best, when the loop ends itll return this value'''
        r,m = solveWithSwitch(e,allpoints=False)
        if (np.abs(r/realRadius-1))<bestErr:
            bestE = e
            bestErr = np.abs(r/realRadius-1)

    return bestE
                