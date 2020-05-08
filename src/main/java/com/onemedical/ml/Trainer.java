package com.onemedical.ml;

import java.util.List;

import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.documentiterator.SimpleLabelAwareIterator;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Trainer {

  private static final Logger log = LoggerFactory.getLogger(Trainer.class);

  private static final int NumEpochs = 5;
  private static final int Seed = 49;
  private static final double LearningRate = 0.01;
  private static final int NumNeurons = 100;
  private static final int Batch = 50;

  private final TrainingDataService trainingDataService;
  private final TrainingListener[] listeners;
  private final LabelsSource roleLabels;
  private final TfidfVectorizer vectorizer;

  public Trainer(
    TrainingDataService trainingDataService,
    TrainingListener[] listeners,
    LabelsSource roleLabels,
    TfidfVectorizer vectorizer
  ) {
    this.trainingDataService = trainingDataService;
    this.listeners = listeners;
    this.roleLabels = roleLabels;
    this.vectorizer = vectorizer;
  }

  public MultiLayerNetwork trainRoleClassifier() {

    List<LabelledDocument> trainingData = trainingDataService.getTrainingData();
    LabelAwareIterator delegate = new SimpleLabelAwareIterator(trainingData);
    DataSetIterator iterator = new TfidfVectorizorDataSetIterator(delegate, roleLabels, vectorizer, Batch);

    int numFeatures = iterator.inputColumns();
    int numLabels = iterator.totalOutcomes();

    log.info("--hyperparameters--");
    log.info("epochs: {}", NumEpochs);
    log.info("seed: {}", Seed);
    log.info("batch size: {}", Batch);
    log.info("learning rate: {}", LearningRate);
    log.info("intput layer num features: {}", numFeatures);
    log.info("hidden layer num neurons: {}", NumNeurons);
    log.info("output layer num labels: {}", numLabels);

    MultiLayerConfiguration conf = getMultiLayerConfiguration(numFeatures, numLabels);
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    if (listeners != null) {
      model.setListeners(listeners);
    }

    log.info("training neural net");
    for (int i = 1; i <= NumEpochs; i++) {
      log.info("epoch {}", i);
      model.fit(iterator);
    }
    log.info("done training neural net");

    return model;
  }
  
  private MultiLayerConfiguration getMultiLayerConfiguration(int numFeatures, int numLabels) {

    MultiLayerConfiguration conf = new NeuralNetConfiguration
      .Builder()
      .seed(Seed)
      .weightInit(WeightInit.XAVIER)
      .updater(new Sgd(LearningRate))
      .l2(1e-4)
      .list()
      .layer(new DenseLayer
        .Builder()
        .nIn(numFeatures)
        .nOut(NumNeurons)
        .activation(Activation.RELU)
        .build()
      )
      .layer(new OutputLayer
        .Builder()
        .nIn(NumNeurons)
        .nOut(numLabels)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
        .build()
      )
      .build();

    return conf;
  }
}
