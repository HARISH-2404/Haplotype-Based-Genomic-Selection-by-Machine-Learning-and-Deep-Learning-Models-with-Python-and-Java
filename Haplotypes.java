package org.hgs;

import java.util.*;

/** Haplotype-allele identification inside blocks and one-hot feature construction. */
public final class Haplotypes {
    public final int[][] hapIds;   // lines x blocks
    public final int[] nAlleles;   // per block

    private Haplotypes(int[][] h, int[] n) { hapIds = h; nAlleles = n; }

    /** G is a 0/2 dosage matrix of inbred lines. Haplotypes rarer than minFreq are merged into the nearest common one. */
    public static Haplotypes build(double[][] G, int[] blockId, double minFreq) {
        int n = G.length;
        int nb = Arrays.stream(blockId).max().getAsInt() + 1;
        List<List<Integer>> cols = new ArrayList<>();
        for (int b = 0; b < nb; b++) cols.add(new ArrayList<>());
        for (int j = 0; j < blockId.length; j++) cols.get(blockId[j]).add(j);

        int[][] hap = new int[n][nb];
        int[] nAll = new int[nb];
        for (int b = 0; b < nb; b++) {
            List<Integer> cj = cols.get(b);
            String[] key = new String[n];
            for (int i = 0; i < n; i++) {
                StringBuilder sb = new StringBuilder();
                for (int j : cj) sb.append(G[i][j] >= 1.0 ? '1' : '0');
                key[i] = sb.toString();
            }
            TreeMap<String, Integer> count = new TreeMap<>();
            for (String k : key) count.merge(k, 1, Integer::sum);
            List<String> uniq = new ArrayList<>(count.keySet());          // lexicographic (same as numpy.unique)
            List<String> common = new ArrayList<>();
            for (String u : uniq) if ((double) count.get(u) / n >= minFreq) common.add(u);
            if (common.isEmpty()) {
                String best = uniq.get(0);
                for (String u : uniq) if (count.get(u) > count.get(best)) best = u;
                common.add(best);
            }
            Map<String, String> remap = new HashMap<>();
            for (String u : uniq) {
                if (common.contains(u)) { remap.put(u, u); continue; }
                String nearest = common.get(0); int bd = Integer.MAX_VALUE;
                for (String c : common) {
                    int d = 0;
                    for (int t = 0; t < u.length(); t++) if (u.charAt(t) != c.charAt(t)) d++;
                    if (d < bd) { bd = d; nearest = c; }
                }
                remap.put(u, nearest);
            }
            TreeSet<String> finalSet = new TreeSet<>(remap.values());
            Map<String, Integer> label = new HashMap<>();
            int k = 0;
            for (String s : finalSet) label.put(s, k++);
            for (int i = 0; i < n; i++) hap[i][b] = label.get(remap.get(key[i]));
            nAll[b] = finalSet.size();
        }
        return new Haplotypes(hap, nAll);
    }

    /** One-hot haplotype-allele design matrix (monomorphic blocks dropped). */
    public double[][] oneHot() {
        int n = hapIds.length, total = 0;
        for (int a : nAlleles) if (a >= 2) total += a;
        double[][] X = new double[n][total];
        int c = 0;
        for (int b = 0; b < nAlleles.length; b++) {
            if (nAlleles[b] < 2) continue;
            for (int i = 0; i < n; i++) X[i][c + hapIds[i][b]] = 1.0;
            c += nAlleles[b];
        }
        return X;
    }
}
