# ----- Make Macros -----

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++11 -gencode arch=compute_60,code=sm_60 -Xptxas="-v" -Xcompiler -fopenmp

CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++11

TARGETS = inference
OBJECTS = main.o kernels.o

LD_FLAGS = -lgomp

# ----- Make Rules -----

all:	$(TARGETS)

%.o: %.cpp vars.h
	${CXX} ${CXXFLAGS} ${OPTFLAGS} $< -c -o $@

%.o : %.cu vars.h
	${NVCC} ${NVCCFLAGS} $< -c -o $@

inference: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml
