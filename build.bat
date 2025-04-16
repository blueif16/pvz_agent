@echo off

REM Create build directory
mkdir build

REM Create manifest file
echo Manifest-Version: 1.0> MANIFEST.MF
echo Main-Class: core.Connector>> MANIFEST.MF
echo Class-Path: .>> MANIFEST.MF
echo.>> MANIFEST.MF

REM Compile Java files
javac -d build src/core/*.java src/ui/*.java src/entities/plants/*.java src/entities/zombies/*.java src/utils/*.java src/entities/projectiles/*.java src/entities/Sun.java

REM Create JAR file
cd build
jar cfm ..\PlantsVsZombies.jar ..\MANIFEST.MF .
cd ..

echo Build complete. Run with: java -jar PlantsVsZombies.jar