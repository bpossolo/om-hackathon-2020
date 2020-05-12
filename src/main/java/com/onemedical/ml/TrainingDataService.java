package com.onemedical.ml;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrainingDataService {

  private static final Logger log = LoggerFactory.getLogger(TrainingDataService.class);

  /**
   * directory with all the training data:
   * dir
   *   |- admin
   *   |   \- doc1.txt
   *   |   \- doc2.txt
   *   |- tech
   *   |   \- doc3.txt
   *   |- medical
   *       \- doc4.txt
   */
  private final Path dir;
  private List<LabelledDocument> training;
  private List<LabelledDocument> validation;
  private List<LabelledDocument> test;
  
  public TrainingDataService(Path dir) {
    this.dir = dir;
  }

  public void init() {
    log.info("training data dir: {}", dir.toString());
    log.info("loading training data into memory");
    LabelAwareIterator iterator = new FileLabelAwareIterator
      .Builder()
      .addSourceFolder(dir.toFile())
      .build();
    List<LabelledDocument> docs = new ArrayList<>();
    iterator.forEachRemaining(docs::add);
    
    Collections.shuffle(docs);

    // 80% for training
    // 10% for validation
    // 10% for test
    int index1 = (int) (docs.size() * 0.8);
    int index2 = (int) (docs.size() * 0.1) + index1;

    training = docs.subList(0, index1);
    validation = docs.subList(index1, index2);
    test = docs.subList(index2, docs.size());

    log.info("done loading training data");
    log.info("training docs: {}", training.size());
    log.info("validation docs: {}", validation.size());
    log.info("test docs: {}", test.size());
    log.info("total docs: {}", docs.size());
  }

  public List<LabelledDocument> getAllData() {
    int total = training.size() + validation.size() + test.size();
    List<LabelledDocument> all = new ArrayList<>(total);
    all.addAll(training);
    all.addAll(validation);
    all.addAll(test);
    return all;
  }

  public List<LabelledDocument> getTrainingData() {
    return training;
  }

  public List<LabelledDocument> getValidationData() {
    return validation;
  }

  public List<LabelledDocument> getTestData() {
    return test;
  }

}
