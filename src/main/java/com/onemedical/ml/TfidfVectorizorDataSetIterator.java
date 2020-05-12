package com.onemedical.ml;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class TfidfVectorizorDataSetIterator implements DataSetIterator {
  
  private static final long serialVersionUID = 1L;
  
  private final LabelAwareIterator delegate;
  private final LabelsSource labelsSource;
  private final TfidfVectorizer vectorizer;
  private final int batch;

  public TfidfVectorizorDataSetIterator(
    LabelAwareIterator delegate,
    LabelsSource labelsSource,
    TfidfVectorizer vectorizer,
    int batch
    ) {
    this.delegate = delegate;
    this.labelsSource = labelsSource;
    this.vectorizer = vectorizer;
    this.batch = batch;
  }

  @Override
  public boolean asyncSupported() {
    return true;
  }

  @Override
  public List<String> getLabels() {
    return labelsSource.getLabels();
  }

  @Override
  public boolean hasNext() {
    return delegate.hasNext();
  }

  @Override
  public DataSet next() {
    return next(batch);
  }

  @Override
  public DataSet next(int num) {
    List<DataSet> datasets = new ArrayList<>(num);
    for (int i = 0; i < num && delegate.hasNext(); i++) {
      LabelledDocument doc = delegate.next();
      String text = doc.getContent();
      String label = doc.getLabels().get(0);
      datasets.add(vectorizer.vectorize(text, label));
    }
    return DataSet.merge(datasets);
  }

  @Override
  public void setPreProcessor(DataSetPreProcessor preProcessor) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void reset() {
    delegate.reset();
  }

  @Override
  public DataSetPreProcessor getPreProcessor() {
    return null;
  }

  @Override
  public int inputColumns() {
    return vectorizer.getVocabCache().numWords();
  }

  @Override
  public int totalOutcomes() {
    return labelsSource.getLabels().size();
  }

  @Override
  public boolean resetSupported() {
    return true;
  }

  @Override
  public int batch() {
    return batch;
  }
  
}