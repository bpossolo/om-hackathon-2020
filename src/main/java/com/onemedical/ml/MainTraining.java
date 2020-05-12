package com.onemedical.ml;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.WebApplicationType;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

@Configuration
public class MainTraining {

  private static final Logger log = LoggerFactory.getLogger(MainTraining.class);
  
  public static void main(String[] args) throws Exception {
    SpringApplication app = new SpringApplication(MainTraining.class, CommonConfig.class);
    app.setWebApplicationType(WebApplicationType.NONE);
    ApplicationContext ctx = app.run(args);
    Environment env = ctx.getEnvironment();

    StatsStorage statsStorage = ctx.getBean(StatsStorage.class);
    UIServer server = UIServer.getInstance();
    server.attach(statsStorage);
    
    Trainer trainer = ctx.getBean("role-trainer", Trainer.class);
    MultiLayerNetwork roleClassifier = trainer.trainRoleClassifier();

    String output = env.getProperty("output-role-model");
    log.info("serializing model to disk: {}", output);
    ModelSerializer.writeModel(roleClassifier, output, false);

    log.info("all done!");
  }

  @Bean("role-trainer")
  public Trainer getRoleTrainer(
    TrainingDataService trainingDataService,
    TrainingListener[] listeners,
    @Qualifier("role-labels") LabelsSource roleLabels,
    TfidfVectorizer vectorizer
  ) {
    return new Trainer(trainingDataService, listeners, roleLabels, vectorizer);
  }

  @Bean
  public StatsStorage getStatsStorage() {
    return new InMemoryStatsStorage();
  }

  @Bean
  public TrainingListener[] getTrainingListeners(StatsStorage storage) {
    return new TrainingListener[] {
      new StatsListener(storage),
      new ScoreIterationListener(1000)
    };
  }

}
