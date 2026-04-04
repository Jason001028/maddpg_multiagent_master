@echo off
set EPOCHS=150

echo [1/2] Running vdn for %EPOCHS% epochs...
python train.py --algo vdn --epochs %EPOCHS%
if errorlevel 1 ( echo vdn FAILED & exit /b 1 )

echo [2/2] Running iql for %EPOCHS% epochs...
python train.py --algo iql --epochs %EPOCHS%
if errorlevel 1 ( echo iql FAILED & exit /b 1 )

echo All experiments done.
