package org.hgs;

import java.util.Random;

/**
 * Feed-forward neural network (deep learning) implemented from scratch:
 * ReLU hidden layers, linear output, squared-error loss, L2 weight decay, mini-batch Adam,
 * early stopping on an internal 15 % validation split. Inputs and response are standardised.
 */
public class Mlp {
    private final int[] hidden; private final double alpha, lr; private final int maxEpochs, patience, batch; private final long seed;
    private double[][][] W; private double[][] B;
    private double[] mu, sd; private double ym, ys;

    public int epochsRun;

    public Mlp(int[] hidden, double alpha, double lr, int maxEpochs, int patience, int batch, long seed) {
        this.hidden = hidden; this.alpha = alpha; this.lr = lr; this.maxEpochs = maxEpochs;
        this.patience = patience; this.batch = batch; this.seed = seed;
    }

    private void init(int in, Random rng) {
        int L = hidden.length + 1;
        W = new double[L][][]; B = new double[L][];
        int prev = in;
        for (int l = 0; l < L; l++) {
            int out = (l == L - 1) ? 1 : hidden[l];
            W[l] = new double[prev][out]; B[l] = new double[out];
            double s = Math.sqrt(2.0 / prev);
            for (int a = 0; a < prev; a++) for (int b = 0; b < out; b++) W[l][a][b] = rng.nextGaussian() * s;
            prev = out;
        }
    }

    /** Forward pass; returns activations of every layer (index 0 = input). */
    private double[][][] forward(double[][] X) {
        int L = W.length;
        double[][][] act = new double[L + 1][][];
        act[0] = X;
        for (int l = 0; l < L; l++) {
            int rows = act[l].length, in = W[l].length, out = W[l][0].length;
            double[][] z = new double[rows][out];
            for (int r = 0; r < rows; r++) {
                double[] xr = act[l][r], zr = z[r];
                for (int b = 0; b < out; b++) zr[b] = B[l][b];
                for (int a = 0; a < in; a++) {
                    double v = xr[a];
                    if (v == 0.0) continue;
                    double[] wa = W[l][a];
                    for (int b = 0; b < out; b++) zr[b] += v * wa[b];
                }
                if (l < L - 1) for (int b = 0; b < out; b++) if (zr[b] < 0) zr[b] = 0;
            }
            act[l + 1] = z;
        }
        return act;
    }

    private double[][] standardise(double[][] X) {
        double[][] Z = new double[X.length][X[0].length];
        for (int i = 0; i < X.length; i++) for (int j = 0; j < mu.length; j++) Z[i][j] = (X[i][j] - mu[j]) / sd[j];
        return Z;
    }

    public void fit(double[][] X, double[] y) {
        Random rng = new Random(seed);
        int n = X.length, p = X[0].length;
        mu = new double[p]; sd = new double[p];
        for (int j = 0; j < p; j++) {
            double m = 0; for (double[] r : X) m += r[j]; m /= n;
            double s = 0; for (double[] r : X) s += (r[j] - m) * (r[j] - m);
            mu[j] = m; sd[j] = Math.sqrt(s / n); if (sd[j] == 0) sd[j] = 1.0;
        }
        ym = Utils.mean(y); ys = Utils.sd(y); if (ys == 0) ys = 1.0;
        double[][] Z = standardise(X);
        double[] yz = new double[n];
        for (int i = 0; i < n; i++) yz[i] = (y[i] - ym) / ys;

        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) { int j = rng.nextInt(i + 1); int t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
        int nv = Math.max((int) (n * 0.15), 1);
        int[] va = java.util.Arrays.copyOfRange(perm, 0, nv), tr = java.util.Arrays.copyOfRange(perm, nv, n);
        double[][] Zva = Utils.subsetRows(Z, va); double[] yva = Utils.subset(yz, va);

        init(p, rng);
        int L = W.length;
        double[][][] mW = new double[L][][], vW = new double[L][][], bestW = new double[L][][];
        double[][] mB = new double[L][], vB = new double[L][], bestB = new double[L][];
        for (int l = 0; l < L; l++) {
            mW[l] = new double[W[l].length][W[l][0].length]; vW[l] = new double[W[l].length][W[l][0].length];
            mB[l] = new double[B[l].length]; vB[l] = new double[B[l].length];
        }
        double best = Double.POSITIVE_INFINITY; int wait = 0, step = 0;
        copyParams(bestW, bestB);
        for (int ep = 0; ep < maxEpochs; ep++) {
            int[] order = tr.clone();
            for (int i = order.length - 1; i > 0; i--) { int j = rng.nextInt(i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
            for (int s = 0; s < order.length; s += batch) {
                int e = Math.min(s + batch, order.length);
                int m = e - s;
                double[][] xb = new double[m][]; double[] yb = new double[m];
                for (int k = 0; k < m; k++) { xb[k] = Z[order[s + k]]; yb[k] = yz[order[s + k]]; }
                double[][][] act = forward(xb);
                double[][] delta = new double[m][1];
                for (int k = 0; k < m; k++) delta[k][0] = (act[L][k][0] - yb[k]) / m;
                step++;
                for (int l = L - 1; l >= 0; l--) {
                    int in = W[l].length, out = W[l][0].length;
                    double[][] gW = new double[in][out]; double[] gB = new double[out];
                    for (int k = 0; k < m; k++) {
                        double[] ak = act[l][k], dk = delta[k];
                        for (int b = 0; b < out; b++) gB[b] += dk[b];
                        for (int a = 0; a < in; a++) {
                            double v = ak[a]; if (v == 0.0) continue;
                            for (int b = 0; b < out; b++) gW[a][b] += v * dk[b];
                        }
                    }
                    double[][] prevDelta = null;
                    if (l > 0) {                                  // propagate before updating W[l]
                        prevDelta = new double[m][in];
                        for (int k = 0; k < m; k++)
                            for (int a = 0; a < in; a++) {
                                if (act[l][k][a] <= 0) continue;   // ReLU derivative
                                double s2 = 0;
                                for (int b = 0; b < out; b++) s2 += W[l][a][b] * delta[k][b];
                                prevDelta[k][a] = s2;
                            }
                    }
                    double c1 = 1 - Math.pow(0.9, step), c2 = 1 - Math.pow(0.999, step);
                    double wd = alpha / tr.length;
                    for (int a = 0; a < in; a++)
                        for (int b = 0; b < out; b++) {
                            double g = gW[a][b] + wd * W[l][a][b];
                            mW[l][a][b] = 0.9 * mW[l][a][b] + 0.1 * g; vW[l][a][b] = 0.999 * vW[l][a][b] + 0.001 * g * g;
                            W[l][a][b] -= lr * (mW[l][a][b] / c1) / (Math.sqrt(vW[l][a][b] / c2) + 1e-8);
                        }
                    for (int b = 0; b < out; b++) {
                        mB[l][b] = 0.9 * mB[l][b] + 0.1 * gB[b]; vB[l][b] = 0.999 * vB[l][b] + 0.001 * gB[b] * gB[b];
                        B[l][b] -= lr * (mB[l][b] / c1) / (Math.sqrt(vB[l][b] / c2) + 1e-8);
                    }
                    if (l > 0) delta = prevDelta;
                }
            }
            double[][][] a = forward(Zva);
            double vl = 0;
            for (int k = 0; k < nv; k++) { double d = a[L][k][0] - yva[k]; vl += d * d; }
            vl /= nv;
            epochsRun = ep + 1;
            if (vl < best - 1e-5) { best = vl; wait = 0; copyParams(bestW, bestB); } else if (++wait >= patience) break;
        }
        W = bestW; B = bestB;
    }

    private void copyParams(double[][][] w, double[][] b) {
        for (int l = 0; l < W.length; l++) {
            w[l] = new double[W[l].length][];
            for (int a = 0; a < W[l].length; a++) w[l][a] = W[l][a].clone();
            b[l] = B[l].clone();
        }
    }

    public double[] predict(double[][] X) {
        double[][][] act = forward(standardise(X));
        double[] out = new double[X.length];
        for (int i = 0; i < X.length; i++) out[i] = act[W.length][i][0] * ys + ym;
        return out;
    }
}
