import numpy as np
import pandas as pd

from numpy.typing import NDArray
from numpy.random import RandomState

from reg import NonparametricRegressor
from typing import Any


class LSMCPricer:

    def __init__(
        self,
        vol: float,
        spot: float,
        rate: float,
        strike: float,
        expiry: float,
        num_timesteps: int,
        num_paths: int,
        seeds: list[int],
        n_bins: int,
        antithetic: bool = True,
    ) -> None:

        # market and deal params
        self._horizon = expiry
        self._vol = vol
        self._rate = rate
        self._logspot = np.log(spot)
        self._strike = strike

        self._at = antithetic
        if antithetic:
            assert num_paths % 2 == 0, ValueError(
                "if using antithetic samping, must enter an even number of paths"
            )
            num_paths = num_paths // 2

        self._N = num_paths
        self._T = num_timesteps
        self._hor = expiry

        self._drift = self._rate - 0.5 * self._vol**2

        self._timedelta = expiry / num_timesteps

        self._model = NonparametricRegressor(n_bins=n_bins)

        self._results = pd.DataFrame(columns=["Seed", "Value", "Type"])
        self._data: list[Any] = []

        if len(seeds) == 0:
            raise ValueError("cannot provide empty list of seeds")

        if len(seeds) != len(set(seeds)):
            raise ValueError("repeated seed: check your input")

        self._seeds = seeds

    def price(self) -> None:
        """
        Run the least squares Monte Carlo scheme, once for each seed.
        """
        for seed in self._seeds:
            self._set_seed(seed)

            while self._tick > 0:
                self._sample_and_fit()

            self._data.append([seed, self._cashflows.mean(), "American"])

        self._results = pd.DataFrame(data=self._data, columns=["Seed", "Value", "Type"])

    def _sample_and_fit(self) -> None:
        """
        Sample the noise one step back using a Brownian bridge, update the
        spot and fit the current cashflows to determine the new cashflows.
        """

        frac = (self._tick - 1) / self._tick
        std = np.sqrt(frac * self._timedelta)

        ########################################################################
        # train the regression
        ########################################################################

        mean = frac * self._noise  # memo past noise

        self._noise = self._gen.normal(size=self._N, scale=std)

        if self._at:
            self._noise = np.concatenate([self._noise, -self._noise])

        self._noise += mean
        timenow = self._timedelta * (self._tick - 1)

        self._logspot_now = (
            self.vol * self._noise + self._drift * timenow + self._logspot
        )

        payoff_now_pv = np.exp(-self.rate * timenow) * np.maximum(
            self.strike - np.exp(self._logspot_now), 0
        )

        itm_mask = payoff_now_pv > 0

        if not any(itm_mask):
            self._tick -= 1
            return

        self._tick -= 1
        cashflows_masked = self._cashflows[itm_mask]
        x = self._logspot_now[itm_mask].reshape(-1, 1)
        self._model.fit(x, cashflows_masked)  # type: ignore

        # compare payoff now with with continuation value using test set
        cont_pv = self._model.predict(x)  # type: ignore
        itm_payoff_now_pv = payoff_now_pv[itm_mask]

        # update cashflows with new decision
        cashflows = self._cashflows
        itm_cashflows = cashflows[itm_mask]

        # use the decision function to update cashflows for the itm_paths
        param = self._decision(itm_payoff_now_pv, cont_pv)  # type: ignore
        cashflows[itm_mask] = param * itm_payoff_now_pv + (1 - param) * itm_cashflows
        self._cashflows = cashflows

    ### Properties of the sampler ##############################################
    @property
    def vol(self) -> float:
        return self._vol

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def results(self) -> pd.DataFrame:
        return self._results

    ### Setter of the sampler ##################################################

    def _set_seed(self, seed: int) -> None:
        """
        Set the seed of the pricer. Resets the time, restarts the final
        cashflows and recomputes the European.
        """
        self._gen = RandomState(seed=seed)
        self._tick = self._T

        # need to reset some things to base state

        # W(T) is self._noise
        self._noise: NDArray[np.float64] = np.sqrt(self._horizon) * self._gen.normal(
            size=self._N
        )

        if self._at:
            self._noise = np.concatenate([self._noise, -self._noise])

        # sample the spot at time_now = T (horizon)
        self._logspot_now = (
            self._logspot + self.vol * self._noise + self._drift * self._horizon
        )

        self._cashflows = np.exp(-self.rate * self._horizon) * np.maximum(
            0, self.strike - np.exp(self._logspot_now)
        )
        self._data.append([seed, self._cashflows.mean(), "European"])

    ############################################################################

    def _decision(
        self,
        x: NDArray[np.double],
        y: NDArray[np.double],
    ) -> NDArray[np.double]:
        """
        Determines how to modify cashflows. Instead of replacing y = Flow1
        with x = Flow2 when Flow1 > Flow2, replaces with p * Flow1 + (1-p) * Flow2
        where p is computed by this function.
        """

        return (x > y).astype(float)  # type: ignore
