import random
import scipy as sp
#import diffrax

def sir_model():
    beta = 1**-3 # Infection rate
    r = 1**-1 # Constant per capita recovery rate

    S = 499 # Initial susceptible population
    I = 1    # Initial infected population
    R = 0    # Initial recovered population
    N = S + I + R  # Initial total population

    days = 160

    rate_of_new_infections = beta * S * I
    rate_of_recovery = r * I

    #differential equations for the S->I->R scheme
    dS = -beta * S * I
    dI = beta * S * I - r * I
    dR = r * I
    der = [dS, dI, dR]

    return der

def sir_simulation():
    list_of_values = sir_model()
    #for day in range(days):
