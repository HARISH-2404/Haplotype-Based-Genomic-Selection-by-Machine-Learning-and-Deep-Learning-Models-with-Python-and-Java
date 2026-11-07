package org.hgs;

/**
 * RR-BLUP / GBLUP in dual (kernel) form with the ridge parameter chosen by efficient leave-one-out
 * cross-validation (the same criterion as scikit-learn's RidgeCV used in the Python pipeline).
 *
 *   Xc = X - colMeans(Xtrain),  K = Xc Xc' + c 11'   (c large: unpenalised intercept)
 *   c(lambda) = (K + lambda I)^-1 (y - ybar),   LOO residual_i = c_i / [(K + lambda I)^-1]_ii
 *   prediction = ybar + Xc_test Xc' c
 */
public class RidgeBlup {
    public double bestLambda;
    private double[][] Xc; private double[] mu, coef; private double ybar, icept;

    public static double[] lambdaGrid() {
        double[] g = new double[25];
        for (int i = 0; i < 25; i++) g[i] = Math.pow(10.0, 6.0 * i / 24.0);   // 1 ... 1e6 (log spaced)
        return g;
    }

    public void fit(double[][] X, double[] y) {
        int n = X.length, p = X[0].length;
        mu = new double[p];
        for (double[] r : X) for (int j = 0; j < p; j++) mu[j] += r[j];
        for (int j = 0; j < p; j++) mu[j] /= n;
        Xc = new double[n][p];
        for (int i = 0; i < n; i++) for (int j = 0; j < p; j++) Xc[i][j] = X[i][j] - mu[j];
        ybar = Utils.mean(y);
        double[] yc = new double[n];
        for (int i = 0; i < n; i++) yc[i] = y[i] - ybar;

        double[][] K = new double[n][n];
        for (int a = 0; a < n; a++)
            for (int b = a; b < n; b++) {
                double s = 0;
                for (int j = 0; j < p; j++) s += Xc[a][j] * Xc[b][j];
                K[a][b] = K[b][a] = s;
            }
        // Unpenalised-intercept trick: add a large constant to the kernel (K + c 11'). Without it the centred kernel is
        // singular and the leave-one-out score is spuriously optimal at tiny lambda.
        double md = 0;
        for (int a = 0; a < n; a++) md += K[a][a];
        icept = 1000.0 * md / n;
        for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) K[a][b] += icept;
        double bestScore = Double.POSITIVE_INFINITY;
        double[] bestC = null;
        for (double lam : lambdaGrid()) {
            double[][] A = new double[n][n];
            for (int a = 0; a < n; a++) { System.arraycopy(K[a], 0, A[a], 0, n); A[a][a] += lam; }
            double[][] Ainv = spdInverse(A);
            double[] c = new double[n];
            for (int a = 0; a < n; a++) { double s = 0; for (int b = 0; b < n; b++) s += Ainv[a][b] * yc[b]; c[a] = s; }
            double loo = 0;
            for (int a = 0; a < n; a++) { double r = c[a] / Ainv[a][a]; loo += r * r; }
            loo /= n;
            if (loo < bestScore) { bestScore = loo; bestLambda = lam; bestC = c; }
        }
        coef = bestC;
    }

    public double[] predict(double[][] Xte) {
        int p = mu.length, n = Xc.length;
        double[] out = new double[Xte.length];
        for (int t = 0; t < Xte.length; t++) {
            double[] z = new double[p];
            for (int j = 0; j < p; j++) z[j] = Xte[t][j] - mu[j];
            double s = 0;
            for (int a = 0; a < n; a++) {
                double k = 0;
                for (int j = 0; j < p; j++) k += z[j] * Xc[a][j];
                s += (k + icept) * coef[a];
            }
            out[t] = ybar + s;
        }
        return out;
    }

    /** Inverse of a symmetric positive-definite matrix via Cholesky (A = L L'). */
    static double[][] spdInverse(double[][] A) {
        int n = A.length;
        double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = A[i][j];
                for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
                if (i == j) L[i][i] = Math.sqrt(s); else L[i][j] = s / L[j][j];
            }
        }
        double[][] Linv = new double[n][n];                    // forward-substitute L Linv = I
        for (int c = 0; c < n; c++) {
            for (int i = c; i < n; i++) {
                double s = (i == c) ? 1.0 : 0.0;
                for (int k = c; k < i; k++) s -= L[i][k] * Linv[k][c];
                Linv[i][c] = s / L[i][i];
            }
        }
        double[][] inv = new double[n][n];                     // A^-1 = Linv' Linv
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++) {
                double s = 0;
                for (int k = j; k < n; k++) s += Linv[k][i] * Linv[k][j];
                inv[i][j] = inv[j][i] = s;
            }
        return inv;
    }
}
