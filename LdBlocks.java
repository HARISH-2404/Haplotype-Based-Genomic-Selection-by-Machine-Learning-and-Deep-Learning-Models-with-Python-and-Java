package org.hgs;

import java.util.*;

/** LD (r^2) calculation and haplotype-block definitions. */
public final class LdBlocks {
    private LdBlocks() {}

    /** r^2 matrix for the SNP columns listed in cols. */
    static double[][] r2(double[][] G, int[] cols) {
        int n = G.length, m = cols.length;
        double[][] z = new double[m][n];
        for (int a = 0; a < m; a++) {
            double mu = 0, s = 0;
            for (int i = 0; i < n; i++) mu += G[i][cols[a]];
            mu /= n;
            for (int i = 0; i < n; i++) { double v = G[i][cols[a]] - mu; s += v * v; }
            double sd = Math.sqrt(s / n); if (sd == 0) sd = 1.0;
            for (int i = 0; i < n; i++) z[a][i] = (G[i][cols[a]] - mu) / sd;
        }
        double[][] r2 = new double[m][m];
        for (int a = 0; a < m; a++)
            for (int b = a; b < m; b++) {
                double s = 0;
                for (int i = 0; i < n; i++) s += z[a][i] * z[b][i];
                double r = Math.max(-1, Math.min(1, s / n));
                r2[a][b] = r2[b][a] = r * r;
            }
        return r2;
    }

    /** Greedy LD blocks: add a SNP while its mean r^2 with the current block members >= threshold. */
    public static int[] detect(double[][] G, String[] chr, double threshold, int maxBlock) {
        int p = chr.length, next = 0;
        int[] block = new int[p];
        int start = 0;
        while (start < p) {
            int end = start;
            while (end < p && chr[end].equals(chr[start])) end++;
            int[] cols = new int[end - start];
            for (int k = 0; k < cols.length; k++) cols[k] = start + k;
            double[][] R2 = r2(G, cols);
            int i = 0, m = cols.length;
            while (i < m) {
                int j = i + 1;
                while (j < m && (j - i) < maxBlock) {
                    double s = 0;
                    for (int k = i; k < j; k++) s += R2[j][k];
                    if (s / (j - i) < threshold) break;
                    j++;
                }
                for (int k = i; k < j; k++) block[cols[k]] = next;
                next++;
                i = j;
            }
            start = end;
        }
        return block;
    }

    public static int[] fixedWindows(String[] chr, int window) {
        int p = chr.length, next = 0;
        int[] block = new int[p];
        int start = 0;
        while (start < p) {
            int end = start;
            while (end < p && chr[end].equals(chr[start])) end++;
            for (int s = start; s < end; s += window) {
                for (int k = s; k < Math.min(s + window, end); k++) block[k] = next;
                next++;
            }
            start = end;
        }
        return block;
    }

    public static int[] singleSnp(int p) {
        int[] b = new int[p];
        for (int i = 0; i < p; i++) b[i] = i;
        return b;
    }
}
