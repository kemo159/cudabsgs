@echo off
setlocal EnableExtensions

set "CUDA_ARCH=%CUDA_ARCH%"
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=sm_120"

set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo [E] Visual Studio Build Tools vcvars64.bat not found.
  exit /b 1
)

call "%VCVARS%" >nul

if not exist build_win mkdir build_win

set "CL_COMMON=/nologo /O2 /I. /D_CRT_SECURE_NO_WARNINGS /DWIN64"
set "CL_CPP=%CL_COMMON% /EHsc /std:c++17"
set "NVCCFLAGS=-O3 -std=c++17 -arch=%CUDA_ARCH% -I. -DWIN64"

echo [+] CUDA_ARCH=%CUDA_ARCH%

nvcc %NVCCFLAGS% -dc cuda\bsgs_cuda.cu -o build_win\bsgs_cuda.obj || exit /b 1
nvcc %NVCCFLAGS% -dc cuda\secp256k1_cuda.cu -o build_win\secp256k1_cuda.obj || exit /b 1

cl %CL_CPP% /DKEYHUNT_CUDA /c keyhunt.cpp /Fobuild_win\keyhunt_cuda_main.obj || exit /b 1
cl %CL_CPP% /DKEYHUNT_CUDA /c cuda\keyhunt_cuda_bridge.cpp /Fobuild_win\keyhunt_cuda_bridge.obj || exit /b 1

cl %CL_CPP% /c oldbloom\bloom.cpp /Fobuild_win\oldbloom.obj || exit /b 1
cl %CL_CPP% /c bloom\bloom.cpp /Fobuild_win\bloom.obj || exit /b 1
cl %CL_COMMON% /TC /c base58\base58.c /Fobuild_win\base58.obj || exit /b 1
cl %CL_CPP% /TP /c rmd160\rmd160.c /Fobuild_win\rmd160.obj || exit /b 1
cl %CL_CPP% /TP /c sha3\sha3.c /Fobuild_win\sha3.obj || exit /b 1
cl %CL_CPP% /TP /c sha3\keccak.c /Fobuild_win\keccak.obj || exit /b 1
cl %CL_COMMON% /TC /c xxhash\xxhash.c /Fobuild_win\xxhash.obj || exit /b 1
cl %CL_CPP% /TP /c util.c /Fobuild_win\util.obj || exit /b 1

cl %CL_CPP% /c secp256k1\Int.cpp /Fobuild_win\Int.obj || exit /b 1
cl %CL_CPP% /c secp256k1\Point.cpp /Fobuild_win\Point.obj || exit /b 1
cl %CL_CPP% /c secp256k1\SECP256K1.cpp /Fobuild_win\SECP256K1.obj || exit /b 1
cl %CL_CPP% /c secp256k1\IntMod.cpp /Fobuild_win\IntMod.obj || exit /b 1
cl %CL_CPP% /c secp256k1\Random.cpp /Fobuild_win\Random.obj || exit /b 1
cl %CL_CPP% /c secp256k1\IntGroup.cpp /Fobuild_win\IntGroup.obj || exit /b 1

cl %CL_CPP% /c hash\ripemd160.cpp /Fobuild_win\ripemd160_hash.obj || exit /b 1
cl %CL_CPP% /c hash\ripemd160_sse.cpp /Fobuild_win\ripemd160_sse.obj || exit /b 1
cl %CL_CPP% /c hash\sha256.cpp /Fobuild_win\sha256_hash.obj || exit /b 1
cl %CL_CPP% /c hash\sha256_sse.cpp /Fobuild_win\sha256_sse.obj || exit /b 1

nvcc %NVCCFLAGS% ^
  build_win\keyhunt_cuda_main.obj ^
  build_win\oldbloom.obj build_win\bloom.obj build_win\base58.obj build_win\rmd160.obj ^
  build_win\sha3.obj build_win\keccak.obj build_win\xxhash.obj build_win\util.obj ^
  build_win\Int.obj build_win\Point.obj build_win\SECP256K1.obj build_win\IntMod.obj ^
  build_win\Random.obj build_win\IntGroup.obj ^
  build_win\ripemd160_hash.obj build_win\ripemd160_sse.obj ^
  build_win\sha256_hash.obj build_win\sha256_sse.obj ^
  build_win\keyhunt_cuda_bridge.obj build_win\bsgs_cuda.obj build_win\secp256k1_cuda.obj ^
  -o keyhunt_cuda.exe || exit /b 1

echo [+] Built keyhunt_cuda.exe
