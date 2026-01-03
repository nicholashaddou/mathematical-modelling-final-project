import random
import scipy as sp
import salabim as sim
import numpy as np


class SIRSimulation(sim.Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initial conditions
        self.S = 50  # Initial susceptible population
        self.I = 1  # Initial infected population
        self.R = 0  # Initial recovered population
        self.N = self.S + self.I + self.R  # Total population

        # Parameters for the model
        self.beta = 0.02  # Infection rate
        self.r = 0.3  # Recovery rate
        self.m = 1e-4  # Host death rate
        self.v = 0.1  # Pathogen induced mortality rate

        # Time tracking
        self.sim_time = 0
        self.schedule_next_event()

    def reaction_rates_func(self):

        rate_S_I = self.beta * self.S * self.I  # Infection rate: S + I -> I
        rate_I_R = self.r * self.I  # Recovery rate: I -> R
        rate_I_Dead = self.v * self.I  # Death rate: I -> Dead
        return np.array([rate_S_I, rate_I_R, rate_I_Dead])

    def gillespie_step(self):

        # Calculate the propensities (reaction rates)
        propensities = self.reaction_rates_func()

        # Total propensity
        total_propensity = np.sum(propensities)

        if total_propensity == 0:
            return False

        time_step = np.random.exponential(1 / total_propensity)

        reaction_choice = np.random.choice([0, 1, 2], p=propensities / total_propensity)

        if reaction_choice == 0:  # S -> I (Infection)
            self.S -= 1
            self.I += 1
        elif reaction_choice == 1:  # I -> R (Recovery)
            self.I -= 1
            self.R += 1
        elif reaction_choice == 2:  # I -> Dead (Death due to disease)
            self.I -= 1

        self.sim_time += time_step
        return True

    def process(self):

        while self.gillespie_step():
            print(f"Time: {self.sim_time:.2f}, S: {self.S}, I: {self.I}, R: {self.R}")
            self.hold(0.1)  # Hold for a small time before the next event

    def schedule_next_event(self):
        self.activate()


def run_sir_simulation():
    env = sim.Environment()
    env.run()

run_sir_simulation()
