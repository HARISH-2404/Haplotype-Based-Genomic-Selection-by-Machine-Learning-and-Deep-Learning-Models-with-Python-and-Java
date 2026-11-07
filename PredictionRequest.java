package org.cotton.gs;

import java.util.List;

public record PredictionRequest(
    String trait,
    String featureSet,
    String model,
    List<String> featureNames,
    List<String> genotypes,
    List<List<Double>> matrix
) {}
