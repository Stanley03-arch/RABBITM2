@echo off
setlocal
if not exist build mkdir build
where gcc >nul 2>nul
if errorlevel 1 (
  echo gcc not found. Install MSYS2 MinGW-w64 and add gcc to PATH.
  exit /b 1
)
gcc -O2 -std=c99 -Wall -Wextra -o build\rabbit2.exe src\rabbit2.c
if errorlevel 1 exit /b 1
echo Built build\rabbit2.exe
