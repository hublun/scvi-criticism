from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse
from scvi.model.base import BaseModelClass


class ScviPPC:
    """
    Posterior predictive checks for comparing single-cell generative models
    """

    def __init__(self, n_samples: int = 50, raw_counts: np.ndarray = None):
        """
        TOfDO.
        """
        self.raw_counts = (
            scipy.sparse.coo_matrix(raw_counts)
            if isinstance(raw_counts, np.ndarray)
            else None
        )
        self.posterior_predictive_samples = {}
        self.n_samples = n_samples
        self.models = {}
        self.metrics = {}

    def store_scvi_posterior_samples(
        self, models_dict: Dict[str, BaseModelClass], batch_size=32
    ):
        """
        Samples from the Posterior objects and sets raw_counts if None.
        """
        self.models = models_dict
        self.batch_size = batch_size
        first_model = next(iter(models_dict.keys()))
        self.dataset = models_dict[first_model].adata

        for m, model in self.models.items():
            pp_counts = model.posterior_predictive_sample(
                model.adata, n_samples=self.n_samples, batch_size=self.batch_size
            )
            self.posterior_predictive_samples[m] = scipy.sparse.coo_matrix(pp_counts)

    def coefficient_of_variation(self, cell_wise: bool = True):
        """
        Calculate the coefficient of variation.

        Parameters:
            cell_wise: Calculate for each cell across genes if True, else do the reverse.
        """
        axis = 1 if cell_wise is True else 0
        identifier = "cv_cell" if cell_wise is True else "cv_gene"
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            cv = np.nanmean(
                np.std(samples, axis=axis).todense()
                / np.mean(samples, axis=axis).todense(),
                axis=-1,
            )

            df[m] = cv.ravel()

        df["Raw"] = (
            np.std(self.raw_counts, axis=axis).todense()
            / np.mean(self.raw_counts, axis=axis).todense()
        )
        df["Raw"] = np.nan_to_num(df["Raw"])

        self.metrics[identifier] = df

    def mean(self, cell_wise: bool = False):
        """Calculate the mean across cells in one gene (or vice versa). Reports the mean and std over samples per model

        Parameters:
            cell_wise: Calculate for each cell across genes if True, else do the reverse.
        """
        axis = 1 if cell_wise is True else 0
        identifier = "mean_cell" if cell_wise is True else "mean_gene"
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            item = np.mean(samples, axis=axis).todense()
            item_mean = np.nanmean(item, axis=-1)
            item_std = np.nanstd(item, axis=-1)
            # make all zeros have 0 cv
            df[m + "_mean"] = item_mean.ravel()
            df[m + "_std"] = item_std.ravel()

        df["Raw"] = np.mean(self.raw_counts, axis=axis).todense()
        df["Raw"] = np.nan_to_num(df["Raw"])

        self.metrics[identifier] = df

    def variance(self, cell_wise: bool = False):
        """Calculate the mean across cells in one gene (or vice versa). Reports the mean and std over samples per model

        Parameters:
            cell_wise: Calculate for each cell across genes if True, else do the reverse.
        """
        axis = 1 if cell_wise is True else 0
        identifier = "var_cell" if cell_wise is True else "var_gene"
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            item = np.var(samples, axis=axis).todense()
            item_mean = np.nanmean(item, axis=-1)
            item_std = np.nanstd(item, axis=-1)
            # make all zeros have 0 cv
            df[m + "_mean"] = item_mean.ravel()
            df[m + "_std"] = item_std.ravel()

        df["Raw"] = np.var(self.raw_counts, axis=axis).todense()
        df["Raw"] = np.nan_to_num(df["Raw"])

        self.metrics[identifier] = df

    def median_absolute_error(self, point_estimate="mean"):
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            if point_estimate == "mean":
                point_sample = np.mean(samples.todense(), axis=-1)
            else:
                point_sample = np.median(samples.todense(), axis=-1)
            # TODO This used to be point_sample[:, : self.dataset.nb_genes]. Why?
            # TODO mae_gene used to be np.median(...). Why?
            mae_gene = np.mean(
                np.abs(point_sample[:, :] - self.raw_counts.todense()[:, :])
            )

            df[m] = mae_gene

        self.metrics["mae"] = df

    def mean_squared_log_error(self):
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            point_sample = np.mean(samples, axis=-1)
            msle_gene = np.mean(
                np.square(
                    np.log(point_sample.todense()[:, :] + 1)
                    - np.log(self.raw_counts.todense()[:, :] + 1)
                )
            )
            df[m] = msle_gene

        self.metrics["msle"] = df

    def dropout_ratio(self):
        """Fraction of zeros in for a specific gene"""
        df = pd.DataFrame()
        for m, samples in self.posterior_predictive_samples.items():
            dr_mean = np.mean(np.mean(samples == 0, axis=0), axis=-1).todense()
            df[m + "_mean"] = dr_mean.ravel()

            dr_std = np.std(np.mean(samples == 0, axis=0), axis=-1).todense()
            df[m + "_std"] = dr_std.ravel()

        df["Raw"] = np.mean(self.raw_counts == 0, axis=0).todense()

        self.metrics["dropout_ratio"] = df

    # def gene_gene_correlation(
    #     self, gene_indices: List[int] = None, n_genes: int = 1000
    # ):
    #     """Computes posterior predictive gene-gene correlations

    #     Sets the "all gene-gene correlations" key in `self.metrics`

    #     Args:
    #         :param gene_indices: List of gene indices to use
    #         :param n_genes: If `gene_indices` is None, take this many random genes
    #     """
    #     if gene_indices is None:
    #         self.gene_set = np.random.choice(
    #             self.dataset.nb_genes, size=n_genes, replace=False
    #         )
    #     else:
    #         self.gene_set = gene_indices
    #         n_genes = len(gene_indices)

    #     model_corrs = {}
    #     for m, samples in tqdm(self.posterior_predictive_samples.items()):
    #         correlation_matrix = np.zeros((n_genes, n_genes))
    #         for i in range(self.n_samples):
    #             sample = StandardScaler().fit_transform(samples.todense()[:, :, i])
    #             gene_sample = sample[:, self.gene_set]
    #             correlation_matrix += np.matmul(gene_sample.T, gene_sample)
    #         correlation_matrix /= self.n_samples
    #         correlation_matrix /= self.raw_counts.shape[0] - 1
    #         model_corrs[m] = correlation_matrix.ravel()

    #     scaled_raw_counts = StandardScaler().fit_transform(self.raw_counts.todense())
    #     scaled_genes = scaled_raw_counts[:, self.gene_set]
    #     raw_count_corr = np.matmul(scaled_genes.T, scaled_genes)
    #     raw_count_corr /= self.raw_counts.shape[0] - 1
    #     model_corrs["Raw"] = raw_count_corr.ravel()

    #     model_corrs["gene_names1"] = (
    #         list(self.dataset.gene_names[self.gene_set]) * n_genes
    #     )
    #     model_corrs["gene_names2"] = np.repeat(
    #         self.dataset.gene_names[self.gene_set],
    #         len(self.dataset.gene_names[self.gene_set]),
    #     )

    #     df = pd.DataFrame.from_dict(model_corrs)
    #     self.metrics["all gene-gene correlations"] = df

    # def calibration_error(self, confidence_intervals: List[float] = None):
    #     """Calibration error as defined in

    #         http://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf

    #     Sets the "calibration" key in `self.metrics`

    #     Args:
    #         :param confidence_intervals: List of confidence interval widths to evaluate
    #     """
    #     if confidence_intervals is None:
    #         ps = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 82.5, 85, 87.5, 90, 92.5, 95, 97.5]
    #     else:
    #         ps = confidence_intervals
    #     reverse_ps = ps[::-1]
    #     model_cal = {}
    #     for m, samples in self.posterior_predictive_samples.items():
    #         percentiles = np.percentile(samples.todense(), ps, axis=2)
    #         reverse_percentiles = percentiles[::-1]
    #         cal_error_genes = 0
    #         for i, j, truth, reverse_truth in zip(
    #             percentiles, reverse_percentiles, ps, reverse_ps
    #         ):
    #             if truth > reverse_truth:
    #                 break
    #             true_width = (100 - truth * 2) / 100
    #             # For gene only model
    #             ci = np.logical_and(
    #                 self.raw_counts[:, : self.dataset.nb_genes] >= i,
    #                 self.raw_counts[:, : self.dataset.nb_genes] <= j,
    #             )
    #             pci_genes = np.mean(ci[:, : self.dataset.nb_genes])
    #             cal_error_genes += (pci_genes - true_width) ** 2
    #         model_cal[m] = {"genes": cal_error_genes}
    #     self.metrics["calibration"] = pd.DataFrame.from_dict(model_cal)
