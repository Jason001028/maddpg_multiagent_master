@echo off
set EPOCHS=200

echo [1/5] Running qmix for %EPOCHS% epochs...
python train.py --algo qmix --epochs %EPOCHS%
if errorlevel 1 ( echo qmix FAILED & exit /b 1 )

echo [2/5] Running vdn for %EPOCHS% epochs...
python train.py --algo vdn --epochs %EPOCHS%
if errorlevel 1 ( echo vdn FAILED & exit /b 1 )

echo [3/5] Running maddpg for %EPOCHS% epochs...
python train.py --algo maddpg --epochs %EPOCHS%
if errorlevel 1 ( echo maddpg FAILED & exit /b 1 )

echo [4/5] Running ra_maddpg for %EPOCHS% epochs...
python train.py --algo ra_maddpg --epochs %EPOCHS%
if errorlevel 1 ( echo ra_maddpg FAILED & exit /b 1 )

echo [5/5] Running iql for %EPOCHS% epochs...
python train.py --algo iql --epochs %EPOCHS%
if errorlevel 1 ( echo iql FAILED & exit /b 1 )

echo All experiments done.
