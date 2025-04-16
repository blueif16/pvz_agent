#!/bin/bash

# Create build directory
mkdir -p build

# Create manifest file
echo "Manifest-Version: 1.0
Main-Class: core.Connector
" > MANIFEST.MF

# Compile Java files with Gson in the classpath
javac -cp ".:gson-2.12.1.jar" -d build src/core/*.java src/ui/*.java src/entities/plants/*.java src/entities/zombies/*.java src/utils/*.java src/entities/projectiles/*.java src/entities/Sun.java

# Optionally copy assets (e.g., images) into the build directory
# Adjust the source folder as necessary (here assuming your assets are in src/assets)
cp -r src/images build/images

# Extract Gson JAR contents into build directory
cd build
jar xf ../gson-2.12.1.jar
rm -rf META-INF  # Remove Gson's META-INF to avoid conflicts
cd ..

# Create JAR file, which now includes your compiled classes and assets
cd build
jar cfm ../PlantsVsZombies.jar ../MANIFEST.MF .
cd ..

# Make JAR executable
chmod +x PlantsVsZombies.jar

echo "Build complete. Run with: java -jar PlantsVsZombies.jar"
