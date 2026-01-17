import numpy as np
import salabim as sim
import matplotlib.pyplot as plt

class SIRSimulation(sim.Component):
    def __init__(self, max_time=50, **kwargs):
        super().__init__(**kwargs)

        # Initial conditions
        self.S = 50
        self.I = 1
        self.R = 0
        self.N = self.S + self.I + self.R

        # Parameters
        self.beta = 0.02
        self.r = 0.3
        self.v = 0.1

        self.sim_time = 0
        self.max_time = max_time

        # Storage
        self.times = [0.0]
        self.S_hist = [self.S]
        self.I_hist = [self.I]
        self.R_hist = [self.R]

        self.activate()

    def reaction_rates(self):
        return np.array([
            self.beta * self.S * self.I,   # Infection
            self.r * self.I,               # Recovery
            self.v * self.I                # Death
        ])

    def gillespie_step(self):
        rates = self.reaction_rates()
        total_rate = np.sum(rates)

        if total_rate == 0 or self.sim_time > self.max_time:
            return False

        dt = np.random.exponential(1 / total_rate)
        reaction = np.random.choice(3, p=rates / total_rate)

        if reaction == 0 and self.S > 0:
            self.S -= 1
            self.I += 1
        elif reaction == 1 and self.I > 0:
            self.I -= 1
            self.R += 1
        elif reaction == 2 and self.I > 0:
            self.I -= 1

        self.sim_time += dt

        self.times.append(self.sim_time)
        self.S_hist.append(self.S)
        self.I_hist.append(self.I)
        self.R_hist.append(self.R)

        return True

    def process(self):
        while self.gillespie_step():
            self.hold(0)


def run_ensemble(n_sim=200, max_time=50, time_grid=None):
    if time_grid is None:
        time_grid = np.linspace(0, max_time, 200)

    S_all, I_all, R_all = [], [], []
    extinction_times = []

    for _ in range(n_sim):
        env = sim.Environment()
        sir = SIRSimulation(max_time=max_time)
        env.run()

        # Interpolate trajectories onto common time grid
        S_interp = np.interp(time_grid, sir.times, sir.S_hist)
        I_interp = np.interp(time_grid, sir.times, sir.I_hist)
        R_interp = np.interp(time_grid, sir.times, sir.R_hist)

        S_all.append(S_interp)
        I_all.append(I_interp)
        R_all.append(R_interp)

        # Extinction time
        extinct_indices = np.where(I_interp <= 0)[0]
        extinction_times.append(
            time_grid[extinct_indices[0]] if len(extinct_indices) > 0 else np.inf
        )

    return (
        time_grid,
        np.array(S_all),
        np.array(I_all),
        np.array(R_all),
        np.array(extinction_times),
    )


time, S_all, I_all, R_all, extinction_times = run_ensemble()

# Mean and quantiles
I_mean = I_all.mean(axis=0)
I_low = np.percentile(I_all, 10, axis=0)
I_high = np.percentile(I_all, 90, axis=0)

plt.figure()
plt.plot(time, I_mean, label="Mean infected")
plt.fill_between(time, I_low, I_high, alpha=0.3, label="10–90% quantile")
plt.xlabel("Time")
plt.ylabel("Infected individuals")
plt.title("Stochastic SIR dynamics")
plt.legend()
plt.show()

# Extinction probability over time
P_ext = [np.mean(extinction_times <= t) for t in time]

plt.figure()
plt.plot(time, P_ext)
plt.xlabel("Time")
plt.ylabel("Probability of extinction")
plt.title("Probability of epidemic extinction")
plt.show()
