@echo off
set EPOCHS=200

echo [1/3] Running maddpg for %EPOCHS% epochs...
python train.py --algo legacy_maddpg --epochs %EPOCHS%
if errorlevel 1 ( echo maddpg FAILED & exit /b 1 )

echo [2/3] Running ef_maddpg for %EPOCHS% epochs...
python train.py --algo ef_maddpg --epochs %EPOCHS%
if errorlevel 1 ( echo ef_maddpg FAILED & exit /b 1 )

echo [3/3] Running ra_maddpg for %EPOCHS% epochs...
python train.py --algo ra_maddpg --epochs %EPOCHS%
if errorlevel 1 ( echo ra_maddpg FAILED & exit /b 1 )

echo All experiments done.
