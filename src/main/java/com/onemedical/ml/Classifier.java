package com.onemedical.ml;

import java.nio.file.Path;

import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.BertWordPiecePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CompositePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Classifier {

  private static final int NumEpochs = 5;
  private static final int MinWordFrequency = 20;
  private static final int Seed = 49;
  private static final double LearningRate = 0.01;
  private static final int NumNeurons = 100;
  private static final int Batch = 50;

  private final Path dataDir;
  private final TrainingListener[] listeners;
  private TfidfVectorizer vectorizer;

  public Classifier(Path dataDir, TrainingListener[] listeners) {
    this.dataDir = dataDir;
    this.listeners = listeners;
  }

  public LabelsSource getRoleLabels() {
    LabelsSource labels = new LabelsSource();
    labels.storeLabel("admin");
    labels.storeLabel("tech");
    labels.storeLabel("medical");
    return labels;
  }

  public MultiLayerNetwork getRoleNetwork() {

    LabelsSource labelsSource = getRoleLabels();

    initVectorizer();

    LabelAwareIterator delegate = null;
    DataSetIterator iterator = new TfidfVectorizorDataSetIterator(delegate, labelsSource, vectorizer, Batch);

    int numFeatures = iterator.inputColumns();
    int numLabels = iterator.totalOutcomes();

    MultiLayerConfiguration conf = getMultiLayerConfiguration(numFeatures, numLabels);
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    if (listeners != null) {
      model.setListeners(listeners);
    }

    for (int i = 0; i < NumEpochs; i++) {
      model.fit(iterator);
    }

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

  private void initVectorizer() {
    TokenPreProcess preprocessor = new CompositePreProcessor(
      new BertWordPiecePreProcessor(true, true, null),
      new EndingPreProcessor()
    );
    TokenizerFactory factory = new DefaultTokenizerFactory();
    factory.setTokenPreProcessor(preprocessor);

    LabelAwareIterator iterator = null;

    vectorizer = new TfidfVectorizer
      .Builder()
      .setMinWordFrequency(MinWordFrequency)
      .setStopWords(StopWords.getStopWords())
      .setIterator(iterator)
      .setTokenizerFactory(factory)
      .build();

    vectorizer.fit();
  }
}
