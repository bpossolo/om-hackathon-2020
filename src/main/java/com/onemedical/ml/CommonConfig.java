package com.onemedical.ml;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.List;

import org.deeplearning4j.bagofwords.vectorizer.BaseTextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.documentiterator.SimpleLabelAwareIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.BertWordPiecePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CompositePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

@Configuration
public class CommonConfig {

  private static final Logger log = LoggerFactory.getLogger(CommonConfig.class);

  @Bean("role-labels")
  public LabelsSource getRoleLabels() {
    LabelsSource labels = new LabelsSource();
    labels.storeLabel("admin");
    labels.storeLabel("tech");
    labels.storeLabel("medical");
    return labels;
  }

  @Bean
  public TrainingDataService getTrainingDataService(Environment env) {
    String dir = env.getProperty("data-dir");
    TrainingDataService service = new TrainingDataService(Path.of(dir));
    service.init();
    return service;
  }

  @Bean
  public TfidfVectorizer getVectorizer(
    TrainingDataService trainingDataService,
    @Qualifier("role-labels") LabelsSource roleLabels
  ) throws Exception {
    log.info("initializing tf-idf vectorizer");
    TokenPreProcess preprocessor = new CompositePreProcessor(
      new BertWordPiecePreProcessor(true, true, null),
      new EndingPreProcessor()
    );
    TokenizerFactory factory = new DefaultTokenizerFactory();
    factory.setTokenPreProcessor(preprocessor);

    List<LabelledDocument> docs = trainingDataService.getAllData();
    LabelAwareIterator iterator = new SimpleLabelAwareIterator(docs);

    TfidfVectorizer vectorizer = new TfidfVectorizer
      .Builder()
      .setMinWordFrequency(20)
      .setStopWords(StopWords.getStopWords())
      .setIterator(iterator)
      .setTokenizerFactory(factory)
      .build();

    Field field = BaseTextVectorizer.class.getDeclaredField("labelsSource");
    field.setAccessible(true);
    field.set(vectorizer, roleLabels);

    vectorizer.fit();
    log.info("done initializing tf-idf vectorizer");
    return vectorizer;
  }

}
