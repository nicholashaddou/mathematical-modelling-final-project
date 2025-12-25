import random
import scipy as sp
import diffrax

def sir_model():
    beta = 0 # Infection rate
    r = 0 # Constant per capita recovery rate

    S = 1000 # Initial susceptible population
    I = 1    # Initial infected population
    R = 0    # Initial recovered population
    N = S + I + R  # Initial total population

    days = 160

    rate_of_new_infections = beta * S * I
    rate_of_recovery = r * I

    dS = -beta * S * I
    dI = beta * S * I - r * I
    dR = r * I
    der = [dS, dI, dR]

    return der

def sir_simulation():
    list_of_values = sir_model()
    #for day in range(days):
