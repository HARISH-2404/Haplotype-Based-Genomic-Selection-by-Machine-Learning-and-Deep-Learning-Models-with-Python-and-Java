package org.hgs;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/** Container for genotypes (0/2 dosage, NaN = missing), SNP map, families, phenotypes and true breeding values. */
public class Dataset {
    public String[] lineIds, family, snpIds, chr;
    public int[] pos;
    public double[][] G;                                   // lines x SNPs
    public final Map<String, double[]> pheno = new LinkedHashMap<>();
    public final Map<String, double[]> tbv = new LinkedHashMap<>();

    public int nLines() { return lineIds.length; }
    public int nSnps() { return snpIds.length; }

    public static Dataset load(Path rawDir, Path truthDir) throws IOException {
        Dataset d = new Dataset();
        List<String[]> g = Utils.readCsv(rawDir.resolve("genotypes_snp.csv"));
        String[] header = g.get(0);
        int n = g.size() - 1, p = header.length - 1;
        d.lineIds = new String[n]; d.G = new double[n][p];
        d.snpIds = Arrays.copyOfRange(header, 1, header.length);
        for (int i = 0; i < n; i++) {
            String[] r = g.get(i + 1);
            d.lineIds[i] = r[0];
            for (int j = 0; j < p; j++) d.G[i][j] = r[j + 1].isEmpty() ? Double.NaN : Double.parseDouble(r[j + 1]);
        }
        Map<String, Integer> idx = new HashMap<>();
        for (int i = 0; i < n; i++) idx.put(d.lineIds[i], i);

        // SNP map (same order as the genotype columns)
        List<String[]> si = Utils.readCsv(rawDir.resolve("snp_info.csv"));
        d.chr = new String[p]; d.pos = new int[p];
        for (int j = 0; j < p; j++) { d.chr[j] = si.get(j + 1)[1]; d.pos[j] = Integer.parseInt(si.get(j + 1)[2]); }

        // families
        d.family = new String[n];
        List<String[]> li = Utils.readCsv(rawDir.resolve("lines.csv"));
        for (int r = 1; r < li.size(); r++) d.family[idx.get(li.get(r)[0])] = li.get(r)[1];

        readTraits(rawDir.resolve("phenotypes.csv"), idx, d.pheno, n);
        readTraits(truthDir.resolve("true_breeding_values.csv"), idx, d.tbv, n);
        return d;
    }

    private static void readTraits(Path f, Map<String, Integer> idx, Map<String, double[]> out, int n) throws IOException {
        List<String[]> rows = Utils.readCsv(f);
        String[] h = rows.get(0);
        for (int c = 1; c < h.length; c++) out.put(h[c], new double[n]);
        for (int r = 1; r < rows.size(); r++) {
            int i = idx.get(rows.get(r)[0]);
            for (int c = 1; c < h.length; c++) out.get(h[c])[i] = Double.parseDouble(rows.get(r)[c]);
        }
    }

    /** Keep only the given SNP columns. */
    public void keepSnps(int[] cols) {
        double[][] ng = new double[nLines()][cols.length];
        String[] ns = new String[cols.length], nc = new String[cols.length]; int[] np = new int[cols.length];
        for (int j = 0; j < cols.length; j++) {
            ns[j] = snpIds[cols[j]]; nc[j] = chr[cols[j]]; np[j] = pos[cols[j]];
            for (int i = 0; i < nLines(); i++) ng[i][j] = G[i][cols[j]];
        }
        G = ng; snpIds = ns; chr = nc; pos = np;
    }
}
