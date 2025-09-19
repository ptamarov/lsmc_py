import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.interpolate import CubicSpline

from typing import Literal


class NonparametricRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_bins: int = 10,
        smoothing: Literal["spline", "linear"] | None = "spline",
        eps_scale: float = 1e-12,
    ):
        """
        eps_scale: relative scale to determine tiny spacing tolerance.
                   tol = eps_scale * max(1.0, data_range)
        """
        self.n_bins = int(n_bins)
        self.smoothing = smoothing
        self.eps_scale = eps_scale

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=True, y_numeric=True)
        if X.shape[1] != 1:
            raise ValueError("This estimator only supports 1D feature X")
        self.X_ = X.ravel()
        self.y_ = y

        #  bins by percentiles
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.asarray(np.percentile(self.X_, percentiles), dtype=float)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) - 1 < 1:
            self.constant_ = True
            self.constant_value_ = float(np.mean(self.y_))
            self.bin_edges_ = bin_edges
            self.bin_mids_ = np.array([np.mean(self.X_)])
            self.bin_means_ = np.array([self.constant_value_])
            self.spline_ = None
            return self

        self.bin_edges_ = bin_edges
        # compute midpoints and means
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_ids = np.digitize(self.X_, bin_edges[1:-1], right=True)
        bin_means = np.array(
            [
                self.y_[bin_ids == i].mean() if np.any(bin_ids == i) else np.nan
                for i in range(len(bin_mids))
            ]
        )
        # fill empty bins (interpolate over bin index)
        isnan = np.isnan(bin_means)
        if np.any(isnan):
            bin_means[isnan] = np.interp(
                np.flatnonzero(isnan), np.flatnonzero(~isnan), bin_means[~isnan]
            )

        # collapse duplicated or close midpoints.
        data_range = max(1.0, float(np.max(self.X_) - np.min(self.X_)))
        tol = float(self.eps_scale) * data_range

        order = np.argsort(bin_mids)
        bin_mids = bin_mids[order]
        bin_means = bin_means[order]

        unique_mids = []
        unique_means = []
        cur_mid = bin_mids[0]
        cur_vals = [bin_means[0]]
        for m, mm in zip(bin_mids[1:], bin_means[1:]):
            if abs(m - cur_mid) <= tol:
                cur_vals.append(mm)
            else:
                unique_mids.append(cur_mid)
                unique_means.append(float(np.mean(cur_vals)))
                cur_mid = m
                cur_vals = [mm]

        unique_mids.append(cur_mid)
        unique_means.append(float(np.mean(cur_vals)))

        self.bin_mids_ = np.asarray(unique_mids)
        self.bin_means_ = np.asarray(unique_means)
        self.constant_ = len(self.bin_mids_) == 1
        if self.constant_:
            self.constant_value_ = float(self.bin_means_[0])
            self.spline_ = None
            return self

        if self.smoothing == "spline":
            self.spline_ = CubicSpline(
                self.bin_mids_, self.bin_means_, bc_type="natural"
            )
        else:
            self.spline_ = None

        return self

    def predict(self, X):
        check_is_fitted(self, ["bin_mids_", "bin_means_"])
        X = check_array(X, ensure_2d=True)
        if X.shape[1] != 1:
            raise ValueError("This estimator only supports 1D feature X")
        x_arr = X.ravel()

        if self.constant_:
            return np.full_like(x_arr, fill_value=self.constant_value_, dtype=float)

        if self.smoothing is None:
            bin_ids = np.digitize(x_arr, self.bin_edges_[1:-1], right=True)
            bin_ids = np.clip(bin_ids, 0, len(self.bin_means_) - 1)
            return self.bin_means_[bin_ids]

        if self.smoothing == "spline":
            return self.spline_(x_arr)

        # linear smoothing:
        preds = np.empty_like(x_arr, dtype=float)
        mids = self.bin_mids_
        means = self.bin_means_
        n = len(mids)
        data_range = max(1.0, float(np.max(self.bin_mids_) - np.min(self.bin_mids_)))
        tol = float(self.eps_scale) * data_range

        inds = np.searchsorted(mids, x_arr, side="right") - 1
        inds = np.clip(inds, 0, n - 1)

        for k, (x, i) in enumerate(zip(x_arr, inds)):
            x0 = mids[i]
            y0 = means[i]
            if abs(x - x0) <= tol:
                preds[k] = y0
                continue

            if x > x0:
                j = i + 1
                while j < n and abs(mids[j] - x0) <= tol:
                    j += 1
                if j < n:
                    x1 = mids[j]
                    y1 = means[j]
                else:
                    preds[k] = y0
                    continue
            else:
                j = i - 1
                while j >= 0 and abs(mids[j] - x0) <= tol:
                    j -= 1
                if j >= 0:
                    x1 = mids[j]
                    y1 = means[j]
                else:
                    preds[k] = y0
                    continue

            dx = x1 - x0
            if abs(dx) <= tol:
                preds[k] = y0
            else:
                slope = (y1 - y0) / dx
                preds[k] = y0 + slope * (x - x0)

        return preds
