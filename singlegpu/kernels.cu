#include "vars.h"
#include "cuda_runtime.hpp"

#include <cuda.h>
#define WARPSIZE 32
#define MINIBATCH 12
#define BLOCKSIZE 256
#define BUFFER 24

extern int neuron;
extern int layer;
extern int batch;
extern float bias;

extern int **csrdispl;
extern unsigned short **csrindex;
extern float **csrvalue;

extern float *currfeat;
extern float *nextfeat;
extern int *active;
extern int *categories;
extern int *globalcategories;

extern int myid;
extern int numproc;
extern int numthreads;

extern int *batchdispl;
extern int mybatch;
extern int extbatch;

int **buffdispl;
int **mapdispl;
int **warpdispl;
unsigned short **map;
unsigned short **warpindex;
float **warpvalue;

int **buffdispl_d;
int **mapdispl_d;
int **warpdispl_d;
unsigned short *mapbuff_d;
unsigned short *indbuff_d;
float *valbuff_d;;

#ifdef OUTOFCORE
int  weightsizemax;
int  mapsizemax;
#ifdef OVERLAP
unsigned short *mapstream_d;
unsigned short *indstream_d;
float *valstream_d;
#endif
#else
unsigned short **map_d;
unsigned short **warpindex_d;
float **warpvalue_d;
#endif

float *currfeat_d;
float *nextfeat_d;
int *active_d;
int *categories_d;

int numblocks;
int numwarp;
int buffsize;

cudaStream_t copystream;
cudaStream_t kernelstream;
float elapsedTime;

__device__ float __ReLU(float x) {
   return x<0.0?0.0:x>32.0?32.0:x;
};

__global__ void __launch_bounds__(1024,1) dummy_kernel(float *nextfeat, float *currfeat, int buffsize, int *buffdispl, int *mapdispl, unsigned short *map, int *displ, unsigned short *index, float *value, float bias , int neuron, int *categories, int *active) {
  extern __shared__ float shared[];
  int wind = threadIdx.x%WARPSIZE;
  float reduce[MINIBATCH] = {0.0};
  for (int buff = buffdispl[blockIdx.x]; buff < buffdispl[blockIdx.x+1]; buff++) {
    int mapnz = mapdispl[buff+1]-mapdispl[buff];
    for (int n = threadIdx.x; n < mapnz; n += blockDim.x) {
      int ind = map[mapdispl[buff]+n];
      for (unsigned int f = 0; f < MINIBATCH; f++)
        shared[f*buffsize+n] = currfeat[categories[blockIdx.y*MINIBATCH+f]* (unsigned int) neuron+ind];
    }
    __syncthreads();
    int warp = (buff*blockDim.x+threadIdx.x)/WARPSIZE;
    for (int m = displ[warp]; m < displ[warp+1]; m++) {
      int ind = index[m*WARPSIZE+wind];
      float val = value[m*WARPSIZE+wind];
      for (int f = 0; f < MINIBATCH; f++)
        reduce[f] += shared[f*buffsize+ind]*val;
    }
    __syncthreads();
  }
  int m = blockIdx.x*blockDim.x+threadIdx.x;
  for (int f = 0; f < MINIBATCH; f++)
    if (nextfeat[(blockIdx.y*MINIBATCH+f)*neuron+m]=__ReLU(reduce[f]+bias))
      atomicAdd(active+blockIdx.y*MINIBATCH+f,1);
    
};

void setup_gpu() {

  OR_FATAL(cudaSetDevice(myid%6));

  OR_FATAL(cudaStreamCreate(&copystream));
  OR_FATAL(cudaStreamCreate(&kernelstream));

  buffsize = BUFFER*1024/sizeof(float)/MINIBATCH;
  numblocks = neuron/BLOCKSIZE;
  numwarp = BLOCKSIZE/WARPSIZE;

  preproc();

  double memother = 0.0;
  OR_FATAL(cudaMallocHost((void**)&globalcategories,sizeof(int)*mybatch));
  OR_FATAL(cudaMallocHost((void**)&categories,sizeof(int)*mybatch));
  OR_FATAL(cudaMallocHost((void**)&active,sizeof(int)*mybatch));
  OR_FATAL(cudaMalloc((void**)&active_d,sizeof(int)*extbatch));
  OR_FATAL(cudaMalloc((void**)&categories_d,sizeof(int)*extbatch));
  memother += sizeof(int)*extbatch/1.0e9;
  memother += sizeof(int)*extbatch/1.0e9;
  for (int k = 0; k < mybatch; k++) {
    active[k] = neuron;
    categories[k] = k;
    globalcategories[k] = batchdispl[myid]+k;
  }
  OR_FATAL(cudaMemset(active_d,0,sizeof(int)*extbatch));
  OR_FATAL(cudaMemset(categories_d,0,sizeof(int)*extbatch));
  OR_FATAL(cudaMemcpy(active_d,active,sizeof(int)*mybatch,cudaMemcpyHostToDevice));
  OR_FATAL(cudaMemcpy(categories_d,categories,sizeof(int)*mybatch,cudaMemcpyHostToDevice));

  double memweight = 0.0;
  double memdispl = 0.0;
  double memmap = 0.0;
  buffdispl_d = new int*[layer];
  mapdispl_d = new int*[layer];
  warpdispl_d = new int*[layer];
  #ifdef OUTOFCORE
  weightsizemax = 0;
  mapsizemax = 0;
  #else
  map_d = new unsigned short*[layer];
  warpindex_d = new unsigned short*[layer];
  warpvalue_d = new float*[layer];
  #endif
  for (int l = 0; l < layer; l++) {
    OR_FATAL(cudaMalloc((void**)&buffdispl_d[l],sizeof(int)*(numblocks+1)));
    OR_FATAL(cudaMalloc((void**)&mapdispl_d[l],sizeof(int)*(buffdispl[l][numblocks]+1)));
    OR_FATAL(cudaMalloc((void**)&warpdispl_d[l],sizeof(int)*(buffdispl[l][numblocks]*numwarp+1)));
    memdispl += sizeof(int)*(numblocks+1)/1.0e9;
    memdispl += sizeof(int)*(buffdispl[l][numblocks]+1)/1.0e9;
    memdispl += sizeof(int)*(buffdispl[l][numblocks]*numwarp+1)/1.0e9;
    OR_FATAL(cudaMemcpy(buffdispl_d[l],buffdispl[l],sizeof(int)*(numblocks+1),cudaMemcpyHostToDevice));
    OR_FATAL(cudaMemcpy(mapdispl_d[l],mapdispl[l],sizeof(int)*(buffdispl[l][numblocks]+1),cudaMemcpyHostToDevice));
    OR_FATAL(cudaMemcpy(warpdispl_d[l],warpdispl[l],sizeof(int)*(buffdispl[l][numblocks]*numwarp+1),cudaMemcpyHostToDevice));
    #ifdef OUTOFCORE
    int mapsize = mapdispl[l][buffdispl[l][numblocks]];
    if (mapsize > mapsizemax)
      mapsizemax = mapsize;
    int weightsize = warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE;
    if (weightsize > weightsizemax)
      weightsizemax = weightsize; 
    #else
    OR_FATAL(cudaMalloc((void**)&map_d[l],sizeof(unsigned short)*(mapdispl[l][buffdispl[l][numblocks]])));
    OR_FATAL(cudaMalloc((void**)&warpindex_d[l],sizeof(unsigned short)*warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE));
    OR_FATAL(cudaMalloc((void**)&warpvalue_d[l],sizeof(float)*warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE));
    memmap += sizeof(unsigned short)*(mapdispl[l][buffdispl[l][numblocks]])/1.0e9;
    memweight += sizeof(unsigned short)*warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE/1.0e9;
    memweight += sizeof(float)*warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE/1.0e9;
    OR_FATAL(cudaMemcpy(map_d[l],map[l],sizeof(unsigned short)*(mapdispl[l][buffdispl[l][numblocks]]),cudaMemcpyHostToDevice));
    OR_FATAL(cudaMemcpy(warpindex_d[l],warpindex[l],sizeof(unsigned short)*warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE,cudaMemcpyHostToDevice));
    OR_FATAL(cudaMemcpy(warpvalue_d[l],warpvalue[l],sizeof(float)*warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE,cudaMemcpyHostToDevice));
    #endif
  }
  #ifdef OUTOFCORE
  #ifdef OVERLAP
  OR_FATAL(cudaMalloc((void**)&mapstream_d,sizeof(unsigned short)*mapsizemax*2));
  OR_FATAL(cudaMalloc((void**)&indstream_d,sizeof(unsigned short)*weightsizemax*2));
  OR_FATAL(cudaMalloc((void**)&valstream_d,sizeof(float)*weightsizemax*2));
  memmap += 2*sizeof(unsigned short)*mapsizemax/1.0e9;
  memweight += 2*sizeof(unsigned short)*weightsizemax/1.0e9;
  memweight += 2*sizeof(float)*weightsizemax/1.0e9;
  OR_FATAL(cudaMemcpy(mapstream_d,map[0],sizeof(unsigned short)*mapdispl[0][buffdispl[0][numblocks]],cudaMemcpyHostToDevice));
  OR_FATAL(cudaMemcpy(indstream_d,warpindex[0],sizeof(unsigned short)*warpdispl[0][buffdispl[0][numblocks]*numwarp]*WARPSIZE,cudaMemcpyHostToDevice));
  OR_FATAL(cudaMemcpy(valstream_d,warpvalue[0],sizeof(float)*warpdispl[0][buffdispl[0][numblocks]*numwarp]*WARPSIZE,cudaMemcpyHostToDevice));
  #else
  OR_FATAL(cudaMalloc((void**)&mapbuff_d,sizeof(unsigned short)*mapsizemax));
  OR_FATAL(cudaMalloc((void**)&indbuff_d,sizeof(unsigned short)*weightsizemax));
  OR_FATAL(cudaMalloc((void**)&valbuff_d,sizeof(float)*weightsizemax));
  memmap += sizeof(unsigned short)*mapsizemax/1.0e9;
  memweight += sizeof(unsigned short)*weightsizemax/1.0e9;
  memweight += sizeof(float)*weightsizemax/1.0e9;
  #endif
  #endif

  double memfeat = 0.0;
  fprintf(stderr, "extbatch=%d, neuron=%d\n", extbatch, neuron);
  {
    const size_t bytes = sizeof(float) * size_t(extbatch) * size_t(neuron);
    fflush(stdout);
    fprintf(stderr, "cudaMalloc %lu MB\n", bytes/1024/1024);
    if (cudaSuccess != cudaMalloc((void**)&currfeat_d,bytes)) {
      fprintf(stderr, "ERROR: need more GPU memory\n");
      exit(EXIT_FAILURE);
    }
    fprintf(stderr, "cudaMalloc %lu MB\n", bytes/1024/1024);
    if (cudaSuccess != cudaMalloc((void**)&nextfeat_d,bytes)) {
      fprintf(stderr, "ERROR: need more GPU memory\n");
      exit(EXIT_FAILURE);
    }
    memfeat += bytes/1.0e9;
    memfeat += bytes/1.0e9;
    OR_FATAL(cudaMemset(currfeat_d,0,bytes));
    OR_FATAL(cudaMemset(nextfeat_d,0,bytes));
    OR_FATAL(cudaMemcpy(currfeat_d,currfeat,sizeof(float)*mybatch*neuron,cudaMemcpyHostToDevice));
  }

  double memothers[numproc];
  double memweights[numproc];
  double memdispls[numproc];
  double memmaps[numproc];
  double memfeats[numproc];

  memothers[0] = memother;
  memweights[0] = memweight;
  memdispls[0] = memdispl;
  memmaps[0] = memmap;
  memfeats[0] = memfeat;
}


/* 
Simultaneously launch the kernel and copy weights for the next layer.

Two streams: kernelStream and copyStream.
kernelStream contains the kernel, as well as the associated memset, and bookkeeping operations
copyStream just has the copy operations for the next layer

use copyStart / copyStop events to time the stream, and start/stop events to time the kernel

*/
void infer_gpu(int l) {

/* if OUTOFCORE and OVERLAP, point at the right part of the double-buffer to get the weights from the previous iteration
  if OUTOFCORE and !OVERLAP, copy arguments into the kernel
  otherwise, just get the right layer pointers
*/
  #ifdef OUTOFCORE
  #ifdef OVERLAP
  mapbuff_d = mapstream_d+(l%2)*mapsizemax;
  indbuff_d = indstream_d+(l%2)*weightsizemax;
  valbuff_d = valstream_d+(l%2)*weightsizemax;
  OR_FATAL(cudaStreamSynchronize(copystream));
  #else
  int weightsize = warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE;
  OR_FATAL(cudaMemcpyAsync(indbuff_d,warpindex[l],sizeof(unsigned short)*weightsize,cudaMemcpyHostToDevice,kernelstream));
  OR_FATAL(cudaMemcpyAsync(valbuff_d,warpvalue[l],sizeof(float)*weightsize,cudaMemcpyHostToDevice,kernelstream));

  int mapsize = mapdispl[l][buffdispl[l][numblocks]];
  OR_FATAL(cudaMemcpyAsync(mapbuff_d,map[l],sizeof(unsigned short)*mapsize,cudaMemcpyHostToDevice,kernelstream));
  #endif
  #else
  mapbuff_d = map_d[l];
  indbuff_d = warpindex_d[l];
  valbuff_d = warpvalue_d[l];
  #endif

  dim3 block(BLOCKSIZE);
  dim3 grid(numblocks,(mybatch+MINIBATCH-1)/MINIBATCH);

  // initialize active features in the batch
  OR_FATAL(cudaMemsetAsync(active_d,0,sizeof(int)*mybatch,kernelstream));
  dummy_kernel<<<grid,block,sizeof(float)*buffsize*MINIBATCH,kernelstream>>>(nextfeat_d,currfeat_d,buffsize,buffdispl_d[l],mapdispl_d[l],mapbuff_d,warpdispl_d[l],indbuff_d,valbuff_d,bias,neuron,categories_d,active_d);
  OR_FATAL(cudaMemcpyAsync(active,active_d,sizeof(int)*mybatch,cudaMemcpyDeviceToHost,kernelstream));

  #ifdef OUTOFCORE
  #ifdef OVERLAP
  if (l+1 < layer) {
    OR_FATAL(cudaMemcpyAsync(mapstream_d+((l+1)%2)*mapsizemax,map[l+1],sizeof(unsigned short)*mapdispl[l+1][buffdispl[l+1][numblocks]],cudaMemcpyHostToDevice,copystream));
    OR_FATAL(cudaMemcpyAsync(indstream_d+((l+1)%2)*weightsizemax,warpindex[l+1],sizeof(unsigned short)*warpdispl[l+1][buffdispl[l+1][numblocks]*numwarp]*WARPSIZE,cudaMemcpyHostToDevice,copystream));
    OR_FATAL(cudaMemcpyAsync(valstream_d+((l+1)%2)*weightsizemax,warpvalue[l+1],sizeof(float)*warpdispl[l+1][buffdispl[l+1][numblocks]*numwarp]*WARPSIZE,cudaMemcpyHostToDevice,copystream));
  }
  #else
  #endif
  #endif

  OR_FATAL(cudaStreamSynchronize(kernelstream));

  int feature = 0;
  for (int k = 0; k < mybatch; k++) {
    if (active[k]) {
      globalcategories[feature] = globalcategories[k];
      categories[feature] = k;
      feature++;
    }
  }
  mybatch = feature;

  OR_FATAL(cudaMemcpyAsync(categories_d,categories,sizeof(int)*feature,cudaMemcpyHostToDevice,kernelstream));

  float *tempfeat_d = currfeat_d;
  currfeat_d = nextfeat_d;
  nextfeat_d = tempfeat_d;
};


void preproc() {
  buffdispl = new int*[layer];
  mapdispl = new int*[layer];
  warpdispl = new int*[layer];
  map = new unsigned short*[layer];
  warpindex = new unsigned short*[layer];
  warpvalue = new float*[layer];

  int totbuff = 0;
  int totmapnz = 0;
  int totwarpnz = 0;
  int *temptag = new int[neuron*numthreads];

  for (int l = 0; l < layer; l++) {
    int *numbuff = new int[numblocks];
    buffdispl[l] = new int[numblocks+1];
    
    #pragma omp parallel for
    for (int b = 0; b < numblocks; b++) {
      int *temp = temptag+omp_get_thread_num()*neuron;

      for (int n = 0; n < neuron; n++) {
        temp[n] = 0;
      }
        
      for (int m = b*BLOCKSIZE; m < (b+1)*BLOCKSIZE; m++) {
        for (int n = csrdispl[l][m]; n < csrdispl[l][m+1]; n++) {
          temp[csrindex[l][n]]++;
        }
      }
        
      int footprint = 0;
      for (int n = 0; n < neuron; n++) {
        if (temp[n]) {
          footprint++;
        } 
      }
      numbuff[b] = (footprint+buffsize-1)/buffsize;
    }

    buffdispl[l][0] = 0;
    for (int b = 0; b < numblocks; b++) {
      buffdispl[l][b+1] = buffdispl[l][b]+numbuff[b];
    }
      
    totbuff += buffdispl[l][numblocks];
    int *warpnz = new int[buffdispl[l][numblocks]*numwarp];
    #pragma omp parallel for
    for (int n = 0; n < buffdispl[l][numblocks]*numwarp; n++) {
      warpnz[n] = 0;
    }
      
    int *mapnz = new int[buffdispl[l][numblocks]];
    #pragma omp parallel for
    for (int n = 0; n < buffdispl[l][numblocks]; n++) {
      mapnz[n] = 0;
    }
      
    #pragma omp parallel for
    for (int b = 0; b < numblocks; b++) {
      int *temp = temptag+omp_get_thread_num()*neuron;
      for (int n = 0; n < neuron; n++) {
        temp[n] = 0;
      }
        
      for (int m = b*BLOCKSIZE; m < (b+1)*BLOCKSIZE; m++) {
        for (int n = csrdispl[l][m]; n < csrdispl[l][m+1]; n++) {
          temp[csrindex[l][n]]++;
        }
      }
        
      int footprint = 0;
      for (int n = 0; n < neuron; n++) {
        if (temp[n]) {
          int buff = footprint/buffsize;
          mapnz[buffdispl[l][b]+buff]++;
          temp[n] = buff;
          footprint++;
        }
      }
        
      for (int buff = 0; buff < numbuff[b]; buff++) {
        for (int warp = 0; warp < numwarp; warp++) {
          int tempnz[WARPSIZE] = {0};
          for (int t = 0; t < WARPSIZE; t++) {
            for (int n = csrdispl[l][b*BLOCKSIZE+warp*WARPSIZE+t]; n < csrdispl[l][b*BLOCKSIZE+warp*WARPSIZE+t+1]; n++) {
              if (temp[csrindex[l][n]]==buff) {
                tempnz[t]++;
              }
            }
          }
            
          int warpmax = 0;
          for (int t = 0; t < WARPSIZE; t++) {
            if (tempnz[t]>warpmax) {
              warpmax = tempnz[t];
            }
          }
            
          warpnz[(buffdispl[l][b]+buff)*numwarp+warp] = warpmax;
        }
      }
    }

    warpdispl[l] = new int[buffdispl[l][numblocks]*numwarp+1];
    warpdispl[l][0] = 0;
    for (int warp = 0; warp < buffdispl[l][numblocks]*numwarp; warp++) {
      warpdispl[l][warp+1] = warpdispl[l][warp]+warpnz[warp];
    }
      
    totwarpnz += warpdispl[l][buffdispl[l][numblocks]*numwarp];
    OR_FATAL(cudaMallocHost((void**)&warpindex[l],sizeof(unsigned short)*warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE));
    OR_FATAL(cudaMallocHost((void**)&warpvalue[l],sizeof(float)*warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE));

    #pragma omp parallel for
    for (int n = 0; n < warpdispl[l][buffdispl[l][numblocks]*numwarp]*WARPSIZE; n++) {
      warpindex[l][n] = 0;
      warpvalue[l][n] = 0.0;
    }

    mapdispl[l] = new int[buffdispl[l][numblocks]+1];
    mapdispl[l][0] = 0;

    for (int buff = 0; buff < buffdispl[l][numblocks]; buff++) {
      mapdispl[l][buff+1] = mapdispl[l][buff] + mapnz[buff];
    }
      
    totmapnz += mapdispl[l][buffdispl[l][numblocks]];
    OR_FATAL(cudaMallocHost((void**)&map[l],sizeof(unsigned short)*mapdispl[l][buffdispl[l][numblocks]]));

    #pragma omp parallel for
    for (int n = 0; n < buffdispl[l][numblocks]; n++) {
      mapnz[n] = 0;
    }
      
    #pragma omp parallel for
    for (int b = 0; b < numblocks; b++) {
      int *temp = temptag+omp_get_thread_num()*neuron;
      for (int n = 0; n < neuron; n++) {
        temp[n] = 0;
      }
        
      for (int m = b*BLOCKSIZE; m < (b+1)*BLOCKSIZE; m++) {
        for (int n = csrdispl[l][m]; n < csrdispl[l][m+1]; n++) {
          temp[csrindex[l][n]]++;
        } 
      }
        
      int footprint = 0;
      for (int n = 0; n < neuron; n++) {
        if (temp[n]) {
          int buff = footprint/buffsize;
          map[l][mapdispl[l][buffdispl[l][b]+buff]+mapnz[buffdispl[l][b]+buff]] = n;
          mapnz[buffdispl[l][b]+buff]++;
          temp[n] = footprint;
          footprint++;
        }
      }
        
      for (int buff = 0; buff < numbuff[b]; buff++) {
        for (int warp = 0; warp < numwarp; warp++) {
          int tempnz[WARPSIZE] = {0};
          for (int t = 0; t < WARPSIZE; t++) {
            for (int n = csrdispl[l][b*BLOCKSIZE+warp*WARPSIZE+t]; n < csrdispl[l][b*BLOCKSIZE+warp*WARPSIZE+t+1]; n++) {
              if (temp[csrindex[l][n]]/buffsize==buff) {
                 int ind = (warpdispl[l][(buffdispl[l][b]+buff)*numwarp+warp]+tempnz[t])*WARPSIZE+t;
                 warpindex[l][ind] = temp[csrindex[l][n]]%buffsize;
                 warpvalue[l][ind] = csrvalue[l][n];
                 tempnz[t]++;
              }
            }
          }
        }
      }   
    }

    delete[] numbuff;
    delete[] mapnz;
    delete[] warpnz;
    delete[] csrdispl[l];
    delete[] csrindex[l];
    delete[] csrvalue[l];
  }

  delete[] temptag;
  delete[] csrdispl;
  delete[] csrindex;
  delete[] csrvalue;
};
