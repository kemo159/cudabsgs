# cudabsgs build file
#
# Common targets:
#   make              CPU keyhunt
#   make keyhunt_cuda CUDA build
#   make cuda_selftest
#   make bsgsd
#   make legacy
#   make clean
#
# CUDA architecture can be overridden, for example:
#   make keyhunt_cuda CUDA_ARCH=sm_89

CXX ?= g++
CC ?= gcc
NVCC ?= nvcc
AR ?= ar
RM ?= rm -f

CUDA_ARCH ?= sm_120

CPU_ARCH_FLAGS ?= -m64 -march=native -mtune=native -mssse3
CPU_OPT_FLAGS ?= -Ofast -ftree-vectorize
CPU_LTO_FLAGS ?= -flto
CPU_WARN_FLAGS ?= -Wall -Wextra -Wno-deprecated-copy
C_WARN_FLAGS ?= -Wall -Wextra

CXXFLAGS ?= $(CPU_ARCH_FLAGS) $(CPU_WARN_FLAGS) $(CPU_OPT_FLAGS)
CFLAGS ?= $(CPU_ARCH_FLAGS) $(C_WARN_FLAGS) $(CPU_OPT_FLAGS)
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -I.
LDFLAGS ?=
LDLIBS ?= -lm -lpthread

CPU_COMMON_OBJS = \
	oldbloom.o \
	bloom.o \
	base58.o \
	rmd160.o \
	sha3.o \
	keccak.o \
	xxhash.o \
	util.o \
	Int.o \
	Point.o \
	SECP256K1.o \
	IntMod.o \
	Random.o \
	IntGroup.o \
	hash/ripemd160.o \
	hash/ripemd160_sse.o \
	hash/sha256.o \
	hash/sha256_sse.o

CUDA_LIBS = cuda/libkeyhuntcuda.a cuda/libkeyhuntsecp.a
CUDA_OBJS = cuda/bsgs_cuda.o cuda/secp256k1_cuda.o
CUDA_BRIDGE_OBJ = cuda/keyhunt_cuda_bridge.o

LEGACY_OBJS = \
	oldbloom.o \
	bloom.o \
	base58.o \
	xxhash.o \
	util.o \
	sha3.o \
	keccak.o \
	hashing.o \
	gmp_Int.o \
	gmp_Point.o \
	GMP256K1.o \
	gmp_IntMod.o \
	gmp_Random.o \
	gmp_IntGroup.o

.PHONY: all default clean cuda_backend cuda_selftest keyhunt_cuda legacy bsgsd

all: keyhunt
default: keyhunt

keyhunt: keyhunt.cpp $(CPU_COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ keyhunt.cpp $(CPU_COMMON_OBJS) $(LDFLAGS) $(LDLIBS)

bsgsd: bsgsd.cpp $(CPU_COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ bsgsd.cpp $(CPU_COMMON_OBJS) $(LDFLAGS) $(LDLIBS)

keyhunt_cuda: CPPFLAGS += -DKEYHUNT_CUDA
keyhunt_cuda: cuda_backend keyhunt_cuda_main.o $(CUDA_BRIDGE_OBJ) $(CPU_COMMON_OBJS)
	$(NVCC) $(NVCCFLAGS) keyhunt_cuda_main.o $(CPU_COMMON_OBJS) $(CUDA_BRIDGE_OBJ) $(CUDA_LIBS) -o $@ $(LDFLAGS) $(LDLIBS)

cuda_backend: $(CUDA_LIBS)

cuda/libkeyhuntcuda.a: cuda/bsgs_cuda.o
	$(AR) rcs $@ $^

cuda/libkeyhuntsecp.a: cuda/secp256k1_cuda.o
	$(AR) rcs $@ $^

cuda_selftest: cuda_backend cuda/bsgs_cuda_selftest cuda/secp256k1_cuda_selftest

cuda/bsgs_cuda_selftest: cuda/bsgs_cuda_selftest.o cuda/bloom_cuda_test.o cuda/xxhash_cuda_test.o cuda/Int_cuda_test.o cuda/Point_cuda_test.o cuda/SECP256K1_cuda_test.o cuda/IntMod_cuda_test.o $(CUDA_LIBS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections $^ -o $@

cuda/secp256k1_cuda_selftest: cuda/secp256k1_cuda_selftest.o cuda/libkeyhuntsecp.a
	$(NVCC) $(NVCCFLAGS) $^ -o $@

legacy: keyhunt_legacy.cpp $(LEGACY_OBJS)
	$(CXX) -march=native -mtune=native -Wall -Wextra $(CPU_OPT_FLAGS) -o keyhunt keyhunt_legacy.cpp $(LEGACY_OBJS) -lm -lpthread -lcrypto -lgmp

oldbloom.o: oldbloom/bloom.cpp oldbloom/oldbloom.h
	$(CXX) $(CXXFLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

bloom.o: bloom/bloom.cpp bloom/bloom.h
	$(CXX) $(CXXFLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

base58.o: base58/base58.c base58/libbase58.h
	$(CC) $(CFLAGS) -Wno-unused-parameter -c $< -o $@

rmd160.o: rmd160/rmd160.c rmd160/rmd160.h
	$(CC) $(CFLAGS) -c $< -o $@

sha3.o: sha3/sha3.c sha3/sha3.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

keccak.o: sha3/keccak.c sha3/keccak.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

xxhash.o: xxhash/xxhash.c xxhash/xxhash.h
	$(CC) $(CFLAGS) -c $< -o $@

util.o: util.c util.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

Int.o: secp256k1/Int.cpp secp256k1/Int.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

Point.o: secp256k1/Point.cpp secp256k1/Point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

SECP256K1.o: secp256k1/SECP256K1.cpp secp256k1/SECP256k1.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

IntMod.o: secp256k1/IntMod.cpp secp256k1/Int.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

Random.o: secp256k1/Random.cpp secp256k1/Random.h
	$(CXX) $(CXXFLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

IntGroup.o: secp256k1/IntGroup.cpp secp256k1/IntGroup.h
	$(CXX) $(CXXFLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

hash/ripemd160.o: hash/ripemd160.cpp hash/ripemd160.h
	$(CXX) $(CXXFLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

hash/ripemd160_sse.o: hash/ripemd160_sse.cpp hash/ripemd160.h
	$(CXX) $(CXXFLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

hash/sha256.o: hash/sha256.cpp hash/sha256.h
	$(CXX) $(CXXFLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

hash/sha256_sse.o: hash/sha256_sse.cpp hash/sha256.h
	$(CXX) $(CXXFLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

keyhunt_cuda_main.o: keyhunt.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

cuda/keyhunt_cuda_bridge.o: cuda/keyhunt_cuda_bridge.cpp cuda/keyhunt_cuda_bridge.h cuda/bsgs_cuda.h
	$(CXX) $(CPPFLAGS) -I. -m64 -march=native -mtune=native -O3 -c $< -o $@

cuda/bsgs_cuda.o: cuda/bsgs_cuda.cu cuda/bsgs_cuda.h cuda/secp256k1_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

cuda/secp256k1_cuda.o: cuda/secp256k1_cuda.cu cuda/secp256k1_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

cuda/bsgs_cuda_selftest.o: cuda/bsgs_cuda_selftest.cpp cuda/bsgs_cuda.h
	$(CXX) -m64 -march=native -mtune=native -O3 -I. -c $< -o $@

cuda/secp256k1_cuda_selftest.o: cuda/secp256k1_cuda_selftest.cpp cuda/secp256k1_cuda.cuh
	$(CXX) -m64 -march=native -mtune=native -O3 -I. -c $< -o $@

cuda/bloom_cuda_test.o: bloom/bloom.cpp bloom/bloom.h
	$(CXX) -m64 -march=native -mtune=native -O3 -I. -c $< -o $@

cuda/xxhash_cuda_test.o: xxhash/xxhash.c xxhash/xxhash.h
	$(CC) -m64 -march=native -mtune=native -O3 -I. -c $< -o $@

cuda/Int_cuda_test.o: secp256k1/Int.cpp secp256k1/Int.h
	$(CXX) -m64 -march=native -mtune=native -O3 -ffunction-sections -fdata-sections -I. -c $< -o $@

cuda/Point_cuda_test.o: secp256k1/Point.cpp secp256k1/Point.h
	$(CXX) -m64 -march=native -mtune=native -O3 -ffunction-sections -fdata-sections -I. -c $< -o $@

cuda/SECP256K1_cuda_test.o: secp256k1/SECP256K1.cpp secp256k1/SECP256k1.h
	$(CXX) -m64 -march=native -mtune=native -O3 -ffunction-sections -fdata-sections -I. -c $< -o $@

cuda/IntMod_cuda_test.o: secp256k1/IntMod.cpp secp256k1/Int.h
	$(CXX) -m64 -march=native -mtune=native -O3 -ffunction-sections -fdata-sections -I. -c $< -o $@

hashing.o: hashing.c hashing.h
	$(CXX) -march=native -mtune=native -Wall -Wextra $(CPU_OPT_FLAGS) -c $< -o $@

GMP256K1.o: gmp256k1/GMP256K1.cpp gmp256k1/GMP256K1.h
	$(CXX) -march=native -mtune=native -Wall -Wextra $(CPU_OPT_FLAGS) -c $< -o $@

gmp_Int.o: gmp256k1/Int.cpp gmp256k1/Int.h
	$(CXX) -march=native -mtune=native -Wall -Wextra $(CPU_OPT_FLAGS) -c $< -o $@

gmp_Point.o: gmp256k1/Point.cpp gmp256k1/Point.h
	$(CXX) -march=native -mtune=native -Wall -Wextra $(CPU_OPT_FLAGS) -c $< -o $@

gmp_IntMod.o: gmp256k1/IntMod.cpp gmp256k1/Int.h
	$(CXX) -march=native -mtune=native -Wall -Wextra $(CPU_OPT_FLAGS) -c $< -o $@

gmp_Random.o: gmp256k1/Random.cpp gmp256k1/Random.h
	$(CXX) -march=native -mtune=native -Wall -Wextra $(CPU_OPT_FLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

gmp_IntGroup.o: gmp256k1/IntGroup.cpp gmp256k1/IntGroup.h
	$(CXX) -march=native -mtune=native -Wall -Wextra $(CPU_OPT_FLAGS) $(CPU_LTO_FLAGS) -c $< -o $@

clean:
	$(RM) keyhunt keyhunt_cuda bsgsd
	$(RM) *.o hash/*.o cuda/*.o cuda/*.a
	$(RM) cuda/bsgs_cuda_selftest cuda/secp256k1_cuda_selftest
