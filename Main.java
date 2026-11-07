package org.hgs;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.*;
import java.util.*;

/**
 * Java implementation of the haplotype-based genomic selection pipeline:
 *   QC + imputation -> LD haplotype blocks -> haplotype one-hot features -> RR-BLUP / MLP -> cross-validation.
 * Uses exactly the same CV fold file as the Python pipeline (data/processed/cv_folds.csv).
 *
 * Usage: java -cp out org.hgs.Main [--root <repo root>] [--schemes random_rep1,family_out] [--no-mlp]
 */
public class Main {
    static final double MIN_MAF = 0.05, MAX_MISS = 0.10, LD_R2 = 0.10, MIN_HAP_FREQ = 0.03;
    static final int LD_MAX_BLOCK = 20, WINDOW = 5;

    public static void main(String[] args) throws IOException {
        Path root = Paths.get("..").toAbsolutePath().normalize();
        String schemesArg = "random_rep1,family_out";
        boolean mlp = true;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--root")) root = Paths.get(args[++i]);
            else if (args[i].equals("--schemes")) schemesArg = args[++i];
            else if (args[i].equals("--no-mlp")) mlp = false;
        }
        Path tab = root.resolve("results/tables");
        Files.createDirectories(tab);

        // ---------- load + QC + imputation ----------
        long t0 = System.nanoTime();
        Dataset d = Dataset.load(root.resolve("data/raw"), root.resolve("data/truth"));
        int snpsIn = d.nSnps();
        int[] keep = QualityControl.filterSnps(d, MIN_MAF, MAX_MISS);
        d.keepSnps(keep);
        int filled = QualityControl.imputeNearestNeighbour(d.G);
        System.out.printf("QC: %d -> %d SNPs, %d genotypes imputed%n", snpsIn, d.nSnps(), filled);

        // ---------- haplotype blocks + features ----------
        int[] ldBlk = LdBlocks.detect(d.G, d.chr, LD_R2, LD_MAX_BLOCK);
        int[] fxBlk = LdBlocks.fixedWindows(d.chr, WINDOW);
        Haplotypes hLd = Haplotypes.build(d.G, ldBlk, MIN_HAP_FREQ);
        Haplotypes hFx = Haplotypes.build(d.G, fxBlk, MIN_HAP_FREQ);
        LinkedHashMap<String, double[][]> feats = new LinkedHashMap<>();
        feats.put("SNP", d.G);
        feats.put("HAP_LD", hLd.oneHot());
        feats.put("HAP_FIXED", hFx.oneHot());
        double secPrep = (System.nanoTime() - t0) / 1e9;
        try (PrintWriter w = new PrintWriter(tab.resolve("java_feature_summary.csv").toFile())) {
            w.println("features,n_lines,n_columns,n_blocks");
            w.printf("SNP,%d,%d,%d%n", d.nLines(), d.nSnps(), d.nSnps());
            w.printf("HAP_LD,%d,%d,%d%n", d.nLines(), feats.get("HAP_LD")[0].length, hLd.nAlleles.length);
            w.printf("HAP_FIXED,%d,%d,%d%n", d.nLines(), feats.get("HAP_FIXED")[0].length, hFx.nAlleles.length);
        }
        System.out.printf("Features: SNP=%d, HAP_LD=%d (%d blocks), HAP_FIXED=%d (%d blocks); prep %.1fs%n", d.nSnps(),
                feats.get("HAP_LD")[0].length, hLd.nAlleles.length, feats.get("HAP_FIXED")[0].length, hFx.nAlleles.length, secPrep);

        // ---------- CV folds ----------
        List<String[]> fr = Utils.readCsv(root.resolve("data/processed/cv_folds.csv"));
        String[] fh = fr.get(0);
        Map<String, Integer> rowOfLine = new HashMap<>();
        for (int i = 0; i < d.nLines(); i++) rowOfLine.put(d.lineIds[i], i);
        Map<String, int[]> foldOf = new LinkedHashMap<>();
        for (int c = 2; c < fh.length; c++) foldOf.put(fh[c], new int[d.nLines()]);
        for (int r = 1; r < fr.size(); r++)
            for (int c = 2; c < fh.length; c++) foldOf.get(fh[c])[rowOfLine.get(fr.get(r)[0])] = Integer.parseInt(fr.get(r)[c]);

        // ---------- models x features x traits x folds ----------
        try (PrintWriter w = new PrintWriter(tab.resolve("cv_results_java_folds.csv").toFile())) {
            w.println("trait,features,model,family,scheme,repeat,fold,r_phenotype,r_tbv,rmse,fit_seconds,detail");
            for (String trait : d.pheno.keySet()) {
                double[] y = d.pheno.get(trait), tb = d.tbv.get(trait);
                for (Map.Entry<String, double[][]> fe : feats.entrySet()) {
                    for (String model : new String[]{"RR-BLUP", "MLP"}) {
                        if (model.equals("MLP") && !mlp) continue;
                        for (String sch : schemesArg.split(",")) {
                            int[] fold = foldOf.get(sch);
                            int nf = Arrays.stream(fold).max().getAsInt() + 1;
                            for (int f = 0; f < nf; f++) {
                                List<Integer> trL = new ArrayList<>(), teL = new ArrayList<>();
                                for (int i = 0; i < fold.length; i++) (fold[i] == f ? teL : trL).add(i);
                                int[] tr = trL.stream().mapToInt(Integer::intValue).toArray();
                                int[] te = teL.stream().mapToInt(Integer::intValue).toArray();
                                double[][] Xtr = Utils.subsetRows(fe.getValue(), tr), Xte = Utils.subsetRows(fe.getValue(), te);
                                double[] ytr = Utils.subset(y, tr);
                                long s = System.nanoTime();
                                double[] pred; String detail;
                                if (model.equals("RR-BLUP")) {
                                    RidgeBlup m = new RidgeBlup(); m.fit(Xtr, ytr); pred = m.predict(Xte);
                                    detail = "lambda=" + String.format("%.1f", m.bestLambda);
                                } else {
                                    Mlp m = new Mlp(new int[]{64, 32}, 1.0, 1e-3, 150, 15, 32, 7L + f);
                                    m.fit(Xtr, ytr); pred = m.predict(Xte); detail = "epochs=" + m.epochsRun;
                                }
                                double sec = (System.nanoTime() - s) / 1e9;
                                String scheme = sch.equals("family_out") ? "family_out" : "random_kfold";
                                int rep = sch.startsWith("random_rep") ? Integer.parseInt(sch.substring(10)) : 1;
                                w.printf(Locale.ROOT, "%s,%s,%s,%s,%s,%d,%d,%.5f,%.5f,%.5f,%.3f,%s%n", trait, fe.getKey(), model,
                                        model.equals("RR-BLUP") ? "linear" : "DL", scheme, rep, f,
                                        Utils.pearson(pred, Utils.subset(y, te)), Utils.pearson(pred, Utils.subset(tb, te)),
                                        Utils.rmse(pred, Utils.subset(y, te)), sec, detail);
                                w.flush();
                            }
                            System.out.printf("%s %-9s %-8s %s done%n", trait, fe.getKey(), model, sch);
                        }
                    }
                }
            }
        }
        System.out.println("Java pipeline finished.");
    }
}
