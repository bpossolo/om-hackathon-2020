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

## Run the web server
```
mvn spring-boot:run -Dspring-boot.run.main-class="com.onemedical.ml.Main"
```

## Classify a message
```
curl --header "Content-Type: application/json" --request POST --data '{"subject":"covid", "body":"Id like to get tested for corona virus."}' http://localhost:8080/api/v1/classification/message
```
