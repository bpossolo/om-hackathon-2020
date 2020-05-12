package com.onemedical.ml;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
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
import org.nd4j.evaluation.classification.Evaluation;
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
  private static final int NumNeurons = 200;
  private static final int Batch = 4;

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
    LabelAwareIterator trainingDelegate = new SimpleLabelAwareIterator(trainingData);
    DataSetIterator trainingIterator = new TfidfVectorizorDataSetIterator(trainingDelegate, roleLabels, vectorizer, Batch);
    
    List<LabelledDocument> validationData = trainingDataService.getValidationData();
    LabelAwareIterator validationDelegate = new SimpleLabelAwareIterator(validationData);
    DataSetIterator validationIterator = new TfidfVectorizorDataSetIterator(validationDelegate, roleLabels, vectorizer, Batch);

    List<LabelledDocument> testData = trainingDataService.getValidationData();
    LabelAwareIterator testDelegate = new SimpleLabelAwareIterator(testData);
    DataSetIterator testIterator = new TfidfVectorizorDataSetIterator(testDelegate, roleLabels, vectorizer, Batch);

    int numFeatures = trainingIterator.inputColumns();
    int numLabels = trainingIterator.totalOutcomes();

    log.info("--hyperparameters--");
    log.info("max epochs: {}", NumEpochs);
    log.info("seed: {}", Seed);
    log.info("mini batch size: {}", Batch);
    log.info("learning rate: {}", LearningRate);
    log.info("intput layer num features: {}", numFeatures);
    log.info("hidden layer num neurons: {}", NumNeurons);
    log.info("output layer num labels: {}", numLabels);

    MultiLayerConfiguration networkConf = getNetworkConfiguration(numFeatures, numLabels);
    MultiLayerNetwork network = new MultiLayerNetwork(networkConf);
    network.init();
    if (listeners != null) {
      network.setListeners(listeners);
    }

    EarlyStoppingConfiguration<MultiLayerNetwork> stoppingConf = getStoppingConfiguration(validationIterator);
    EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(stoppingConf, network, trainingIterator);
    log.info("training neural net");
    Instant start = Instant.now();
    EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
    Instant end = Instant.now();
    long duration = Duration.between(start, end).toMinutes();
    log.info("done training neural net in {} minutes", duration);

    MultiLayerNetwork best = result.getBestModel();

    log.info("evaluating best net");
    Evaluation eval = best.evaluate(testIterator, roleLabels.getLabels());
    log.info(eval.stats());

    return best;
  }
  
  private MultiLayerConfiguration getNetworkConfiguration(int numFeatures, int numLabels) {

    MultiLayerConfiguration conf = new NeuralNetConfiguration
      .Builder()
      .seed(Seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
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

  private EarlyStoppingConfiguration<MultiLayerNetwork> getStoppingConfiguration(DataSetIterator iterator) {
    return new EarlyStoppingConfiguration
      .Builder<MultiLayerNetwork>()
      .epochTerminationConditions(new MaxEpochsTerminationCondition(NumEpochs))
      .evaluateEveryNEpochs(1)
      .scoreCalculator(new DataSetLossCalculator(iterator, true))
      .modelSaver(new InMemoryModelSaver<>())
      .build();
  }
}
