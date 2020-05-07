brew tap homebrew/cask-versions
brew cask install java11

brew install maven

/usr/libexec/java_home -V

VS Code settings.json
"java.home": "/Library/Java/JavaVirtualMachines/openjdk-11.0.2.jdk/Contents/Home"

Add path to mvn executable to vscode settings
Disable setting for using mvn wrapper

mvn spring-boot:run -Dspring-boot.run.main-class="com.onemedical.ml.Main"
