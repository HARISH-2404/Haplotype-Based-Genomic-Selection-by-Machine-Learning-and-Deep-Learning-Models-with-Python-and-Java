import numpy as np
from .kernels import DosageKernel

class MultiKernelRidgeGS:
    '''
    Kernel-ridge genomic prediction with separate SNP and haplotype kernels.
    With adjusted genotype means as y, the kernel genetic deviation is a
    genomic prediction score. Treat it as GEBV only when phenotype adjustment,
    target definition and genetic model justify that interpretation.
    '''
    def __init__(self, alpha=1.0, snp_weight=0.5, hap_weight=0.5):
        if not 0 <= snp_weight <= 1 or not 0 <= hap_weight <= 1:
            raise ValueError("Kernel weights must be between 0 and 1.")
        if snp_weight + hap_weight == 0:
            raise ValueError("At least one kernel weight must be positive.")
        total = snp_weight + hap_weight
        self.snp_weight = snp_weight / total
        self.hap_weight = hap_weight / total
        self.alpha = alpha

    def fit(self, snp_train, hap_train, y):
        self.snp_kernel_ = DosageKernel().fit(snp_train)
        self.hap_kernel_ = DosageKernel().fit(hap_train)
        Ks = self.snp_kernel_.kernel(snp_train)
        Kh = self.hap_kernel_.kernel(hap_train)
        self.K_train_ = self.snp_weight * Ks + self.hap_weight * Kh
        self.y_mean_ = float(np.mean(y))
        yc = np.asarray(y, dtype=float) - self.y_mean_
        self.dual_coef_ = np.linalg.solve(
            self.K_train_ + self.alpha * np.eye(len(yc)), yc
        )
        self.train_ids_ = list(snp_train.index)
        return self

    def predict_components(self, snp_new, hap_new):
        Ks = self.snp_kernel_.kernel(snp_new, self._snp_train_)
        Kh = self.hap_kernel_.kernel(hap_new, self._hap_train_)
        snp_effect = self.snp_weight * (Ks @ self.dual_coef_)
        hap_effect = self.hap_weight * (Kh @ self.dual_coef_)
        total_deviation = snp_effect + hap_effect
        return {
            "snp_component": snp_effect,
            "haplotype_component": hap_effect,
            "genomic_deviation": total_deviation,
            "predicted_adjusted_value": self.y_mean_ + total_deviation
        }

    def fit_with_training_matrices(self, snp_train, hap_train, y):
        self._snp_train_ = snp_train.copy()
        self._hap_train_ = hap_train.copy()
        return self.fit(snp_train, hap_train, y)

    def predict(self, snp_new, hap_new):
        return self.predict_components(snp_new, hap_new)["predicted_adjusted_value"]
