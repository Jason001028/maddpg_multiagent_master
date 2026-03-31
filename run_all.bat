@echo off
set EPOCHS=200

echo [1/3] Running qmix for %EPOCHS% epochs...
python train.py --algo qmix --epochs %EPOCHS%
if errorlevel 1 ( echo qmix FAILED & exit /b 1 )

echo [2/3] Running legacy_maddpg for %EPOCHS% epochs...
python train.py --algo legacy_maddpg --epochs %EPOCHS%
if errorlevel 1 ( echo maddpg FAILED & exit /b 1 )

echo [3/3] Running vdn for %EPOCHS% epochs...
python train.py --algo vdn --epochs %EPOCHS%
if errorlevel 1 ( echo vdn FAILED & exit /b 1 )

echo All experiments done.
