@echo off
rem Start OmniGen2 + EditScore7B-v1.1. The first start unpacks the Python environment into env.
setlocal
set "D=%~dp0"
if not exist "%D%env\python.exe" (
  echo First start: unpacking the environment ^(one time, about 1 minute^)...
  "%D%pixi-unpack.exe" "%D%environment.tar" --output-directory "%D%." --env-name env || exit /b 1
)
rem cuDNN loads its parts by name: put the environment DLL folders first on PATH, as conda activation does.
set "PATH=%D%env;%D%env\Library\mingw-w64\bin;%D%env\Library\usr\bin;%D%env\Library\bin;%D%env\Scripts;%PATH%"
"%D%env\python.exe" "%D%app\omnigen2_edit.py" %*
