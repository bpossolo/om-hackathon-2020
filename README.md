## Mac OS X + VSCode Setup
Install java 11
```
brew tap homebrew/cask-versions
brew cask install java11
```

Install Maven for building the project
```
brew install maven
```

Verify Java and Maven were installed correctly
```
java --version
mvn --version
```

Get the java development kit install location
```
/usr/libexec/java_home -V
```

Get location of mvn binary
```
which mvn
```

Configure VSCode settings.json
```
"java.home": "/Library/Java/JavaVirtualMachines/openjdk-11.0.2.jdk/Contents/Home"
```

Add path to mvn executable to vscode settings  
Disable setting for using mvn wrapper  
TODO lookup setting names  

## Build the application
```
mvn compile
```

## Build the classifier models
Builds patient message neural-net classifiers.  
This will take a while...  but you should only need to do this once.  
```
mvn spring-boot:run -Dspring-boot.run.main-class="com.onemedical.ml.MainTraining" -Dspring-boot.run.arguments="--output-role-model=models/role --data-dir=data/training"
```
The serialized neural-net models are persisted to the `models` folder.  

## Run the web server
The web server deserializes the neural-nets from the `models` folder and exposes an api endpoint that can be used to classify patient messages.  
```
mvn spring-boot:run -Dspring-boot.run.main-class="com.onemedical.ml.MainApi" -Dspring-boot.run.arguments="--role-model=models/role --data-dir=data/training"
```

## Classify a message
```
curl --header "Content-Type: application/json" --request POST --data '{"subject":"covid", "body":"Id like to get tested for corona virus."}' http://localhost:8080/api/v1/classification/message
```
