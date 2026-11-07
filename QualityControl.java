package org.hgs;

import java.util.*;

/** SNP quality control (call rate, MAF) and nearest-neighbour imputation for inbred lines. */
public final class QualityControl {
    private QualityControl() {}

    /** Returns indices of SNPs passing the filters. */
    public static int[] filterSnps(Dataset d, double minMaf, double maxMissing) {
        List<Integer> keep = new ArrayList<>();
        int n = d.nLines();
        for (int j = 0; j < d.nSnps(); j++) {
            int obs = 0; double s = 0;
            for (int i = 0; i < n; i++) if (!Double.isNaN(d.G[i][j])) { obs++; s += d.G[i][j]; }
            double miss = 1.0 - (double) obs / n;
            double f = obs == 0 ? 0 : s / obs / 2.0;
            if (miss <= maxMissing && Math.min(f, 1 - f) >= minMaf) keep.add(j);
        }
        return keep.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Copies the missing call from the genetically closest line (IBS distance on co-observed SNPs) that has data. */
    public static int imputeNearestNeighbour(double[][] G) {
        int n = G.length, p = G[0].length, filled = 0;
        double[][] dist = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                int both = 0; double mis = 0;
                for (int j = 0; j < p; j++) {
                    double x = G[a][j], y = G[b][j];
                    if (!Double.isNaN(x) && !Double.isNaN(y)) { both++; mis += Math.abs(x - y) / 2.0; }
                }
                dist[a][b] = dist[b][a] = mis / Math.max(both, 1);
            }
            dist[a][a] = Double.POSITIVE_INFINITY;
        }
        double[] major = new double[p];
        for (int j = 0; j < p; j++) {
            double s = 0; int c = 0;
            for (int i = 0; i < n; i++) if (!Double.isNaN(G[i][j])) { s += G[i][j]; c++; }
            major[j] = (c > 0 && s / c >= 1.0) ? 2.0 : 0.0;
        }
        double[][] orig = new double[n][];
        for (int i = 0; i < n; i++) orig[i] = G[i].clone();
        for (int i = 0; i < n; i++) {
            boolean any = false;
            for (double v : G[i]) if (Double.isNaN(v)) { any = true; break; }
            if (!any) continue;
            Integer[] order = new Integer[n];
            for (int k = 0; k < n; k++) order[k] = k;
            final int ii = i;
            Arrays.sort(order, Comparator.comparingDouble(k -> dist[ii][k]));
            for (int j = 0; j < p; j++) {
                if (!Double.isNaN(G[i][j])) continue;
                boolean done = false;
                for (int k : order) {
                    if (!Double.isNaN(orig[k][j])) { G[i][j] = orig[k][j]; done = true; break; }
                }
                if (!done) G[i][j] = major[j];
                filled++;
            }
        }
        return filled;
    }
}
