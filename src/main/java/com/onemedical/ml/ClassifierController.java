package com.onemedical.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class ClassifierController {

  private static final Logger log = LoggerFactory.getLogger(ClassifierController.class);

  private MultiLayerNetwork roleClassifier;
  private TfidfVectorizer vectorizer;

  public ClassifierController(
    @Qualifier("role-classifier") MultiLayerNetwork roleClassifier,
    TfidfVectorizer vectorizer
  ) {
    this.roleClassifier = roleClassifier;
    this.vectorizer = vectorizer;
  }

  @PostMapping("/classification/message")
  public MessageClassification getClassification(@RequestBody Message message) {
    log.info("message subject: [{}], body: [{}]", message.subject, message.body);

    String text = serialize(message);
    INDArray vector = vectorizer.transform(text);
    INDArray roleProbabilities = roleClassifier.output(vector);
    List<String> roleLabels = null;

    MessageClassification classification = new MessageClassification();
    classification.roles = getLabelledProbabilities(roleLabels, roleProbabilities);
    
    return classification;
  }

  private LabelProbability[] getLabelledProbabilities(List<String> labels, INDArray probabilities) {
    return IntStream
      .range(0, labels.size())
      .mapToObj(i -> {
        String label = labels.get(i);
        double probability = probabilities.getDouble(i);
        return new LabelProbability(label, probability);
      })
      .toArray(LabelProbability[]::new);
  }

  private String serialize(Message message) {
    List<String> parts = new ArrayList<>(2);
    if (message.subject != null) {
      parts.add(message.subject);
    }
    if (message.body != null) {
      parts.add(message.body);
    }
    String text = parts.stream().collect(Collectors.joining(" "));
    return text;
  }
}
