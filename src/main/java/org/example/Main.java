package org.example;

import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        final String DJL_PATH = "djl://ai.djl.huggingface.pytorch/" + args[0];
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelUrls(DJL_PATH)
                .optEngine("PyTorch")
                .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                .optProgress(new ProgressBar())
                .build();
        ZooModel<String, float[]> model = criteria.loadModel();
        Predictor<String, float[]> predictor = model.newPredictor();
        float[] result = predictor.predict("This is a test sentence to get an embedding from");
        System.out.println(Arrays.toString(result));
    }
}