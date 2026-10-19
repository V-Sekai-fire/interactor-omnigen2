@echo off
rem Start OmniGen2 + EditScore7B-v1.1. The first start unpacks the Python environment into env.
setlocal
cd /d "%~dp0"
if not exist env\python.exe (
  echo First start: unpacking the environment ^(one time, about 1 minute^)...
  pixi-unpack.exe environment.tar --output-directory "%~dp0." --env-name env || exit /b 1
)
env\python.exe app\omnigen2_edit.py %*
