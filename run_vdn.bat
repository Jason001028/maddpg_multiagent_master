@echo off
set EPOCHS=200

echo [1/2] Running ra_maddpg for %EPOCHS% epochs...
python train.py --algo ra_maddpg --epochs %EPOCHS%
if errorlevel 1 ( echo ra_maddpg FAILED & exit /b 1 )

echo [2/2] Running iql for %EPOCHS% epochs...
python train.py --algo iql --epochs %EPOCHS%
if errorlevel 1 ( echo iql FAILED & exit /b 1 )

echo All experiments done.
