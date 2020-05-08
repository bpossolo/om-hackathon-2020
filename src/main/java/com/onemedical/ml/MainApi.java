package com.onemedical.ml;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.env.Environment;

@SpringBootApplication
public class MainApi {

  public static void main(String[] args) {
    SpringApplication app = new SpringApplication(MainApi.class, CommonConfig.class);
    app.run(args);
  }

  @Bean("role-classifier")
  public MultiLayerNetwork getRoleClassifier(Environment env) throws IOException {
    String file = env.getProperty("role-model");
    return ModelSerializer.restoreMultiLayerNetwork(new File(file));
  }
}
