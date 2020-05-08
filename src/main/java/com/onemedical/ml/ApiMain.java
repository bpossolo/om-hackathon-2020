package com.onemedical.ml;

import java.io.IOException;
import java.io.InputStream;

import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

@SpringBootApplication
public class ApiMain {

  public static void main(String[] args) {
    SpringApplication.run(ApiMain.class, args);
  }

  @Bean("role-classifier")
  public MultiLayerNetwork getRoleClassifier(Environment env) throws IOException {
    String dir = env.getProperty("model-dir");
    Resource resource = new FileSystemResource(dir);
    try (InputStream is = resource.getInputStream()) {
      MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(is);
      return model;
    }
  }

  @Bean
  public TfidfVectorizer getVectorizer() {
    return null;
  }
}
