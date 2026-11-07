package org.hgs;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Small numeric + CSV helpers (no external libraries). */
public final class Utils {
    private Utils() {}

    public static List<String[]> readCsv(Path p) throws IOException {
        List<String[]> rows = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(p)) {
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.isEmpty()) rows.add(line.split(",", -1));
            }
        }
        return rows;
    }

    public static double mean(double[] a) {
        double s = 0;
        for (double v : a) s += v;
        return s / a.length;
    }

    public static double sd(double[] a) {
        double m = mean(a), s = 0;
        for (double v : a) s += (v - m) * (v - m);
        return Math.sqrt(s / a.length);
    }

    public static double pearson(double[] a, double[] b) {
        double ma = mean(a), mb = mean(b), sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < a.length; i++) {
            double da = a[i] - ma, db = b[i] - mb;
            sab += da * db; saa += da * da; sbb += db * db;
        }
        return (saa == 0 || sbb == 0) ? Double.NaN : sab / Math.sqrt(saa * sbb);
    }

    public static double rmse(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
        return Math.sqrt(s / a.length);
    }

    public static double[] subset(double[] v, int[] idx) {
        double[] o = new double[idx.length];
        for (int i = 0; i < idx.length; i++) o[i] = v[idx[i]];
        return o;
    }

    public static double[][] subsetRows(double[][] m, int[] idx) {
        double[][] o = new double[idx.length][];
        for (int i = 0; i < idx.length; i++) o[i] = m[idx[i]];
        return o;
    }
}
