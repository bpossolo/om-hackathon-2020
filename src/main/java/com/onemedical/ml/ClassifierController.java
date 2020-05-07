package com.onemedical.ml;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class ClassifierController {

  private static final Logger log = LoggerFactory.getLogger(ClassifierController.class);

  @PostMapping("/classification/message")
  public MessageClassification getClassification(@RequestBody Message message) {
    log.info("message subject: [{}], body: [{}]", message.subject, message.body);

    LabelProbability admin = new LabelProbability("admin", 0.1);
    LabelProbability medical = new LabelProbability("medical", 0.8);
    LabelProbability tech = new LabelProbability("tech", 0.1);

    LabelProbability[] roles = new LabelProbability[] { admin, medical, tech };

    MessageClassification classification = new MessageClassification();
    classification.roles = roles;
    return classification;
  }
}
