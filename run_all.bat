@echo off
set EPOCHS=200

echo [1/4] Running qmix for %EPOCHS% epochs...
python train.py --algo qmix --epochs %EPOCHS%
if errorlevel 1 ( echo qmix FAILED & exit /b 1 )

echo [2/4] Running maddpg for %EPOCHS% epochs...
python train.py --algo legacy_maddpg --epochs %EPOCHS%
if errorlevel 1 ( echo maddpg FAILED & exit /b 1 )

echo [3/4] Running vdn for %EPOCHS% epochs...
python train.py --algo vdn --epochs %EPOCHS%
if errorlevel 1 ( echo vdn FAILED & exit /b 1 )

echo [4/4] Running iql for %EPOCHS% epochs...
python train.py --algo iql --epochs %EPOCHS%
if errorlevel 1 ( echo iql FAILED & exit /b 1 )

echo All experiments done.
