@ECHO OFF

REM Command file for Sphinx documentation

set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
echo. 
echo Build the documentation using: 
echo   make.bat [builder]
echo.
echo Available builders:
echo   html       to make HTML docs
echo   latexpdf   to make LaTeX and PDF docs
echo   clean      to clean build directory
echo.

:end
