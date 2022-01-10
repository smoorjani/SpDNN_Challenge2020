#include "vars.h"
#include <cassert>
#include <cstring>


#define MINIBATCH 12

char *dataset;

int neuron;
int layer;
int batch;
int input;
float bias;

int **csrdispl;
unsigned short **csrindex;
float **csrvalue;

float *currfeat;
float *nextfeat;
int *active;
int *categories;
int *globalcategories;

int myid;
int numproc;
int numthreads;

int *numbatch;
int *batchdispl;
int mybatch;
int extbatch;

int main(int argc, char **argv) {

  myid = 0;
  numproc = 1;

#pragma omp parallel
  {
#pragma omp single
    numthreads = omp_get_num_threads();
  }

  dataset = getenv("DATASET");
  char *chartemp;
  chartemp = getenv("NEURON");
  neuron = atoi(chartemp);
  chartemp = getenv("LAYER");
  layer = atoi(chartemp);
  chartemp = getenv("BATCH");
  batch = atoi(chartemp);
  chartemp = getenv("INPUT");
  input = atoi(chartemp);
  chartemp = getenv("BIAS");
  bias = atof(chartemp);

  numbatch = new int[numproc];
  batchdispl = new int[numproc + 1];
  int totbatch = batch / numproc * numproc;
  batchdispl[0] = 0;
  for (int p = 0; p < numproc; p++) {
    numbatch[p] = batch / numproc;
    if (totbatch < batch) {
      totbatch++;
      numbatch[p]++;
    }
    batchdispl[p + 1] = batchdispl[p] + numbatch[p];
  }
  mybatch = numbatch[myid];
  extbatch = (mybatch + MINIBATCH - 1) / MINIBATCH * MINIBATCH;

  csrdispl = new int *[layer];
  csrindex = new unsigned short *[layer];
  csrvalue = new float *[layer];
  currfeat = new float[neuron * (long)mybatch];
  nextfeat = new float[neuron * (long)mybatch];

  fflush(0);
  readinput();
  fflush(0);

  setup_gpu();

  for (int l = 0; l < layer; l++)
    infer_gpu(l);

  int batches[numproc];
  batches[myid] = mybatch;

  int batchesdispl[numproc + 1];
  batchesdispl[0] = 0;

  for (int p = 1; p < numproc + 1; p++)
    batchesdispl[p] = batchesdispl[p - 1] + batches[p - 1];

  int *allcategories = new int[batchesdispl[numproc]];
  std::memcpy(allcategories + batchesdispl[0], globalcategories,
              mybatch * sizeof(*allcategories));
}

void readinput() {
  char chartemp[500];
  float *tempfeat;
  if (myid == 0) {
    sprintf(chartemp, "%s/sparse-images-%d.bin", dataset, neuron);
    FILE *inputf = fopen(chartemp, "rb");
    if (!inputf) {
      fprintf(stderr, "missing %s\n", chartemp);
      exit(1);
    }
    int *row = new int[input];
    int *col = new int[input];
    float *val = new float[input];
    assert(input == fread(row, sizeof(int), input, inputf));
    assert(input == fread(col, sizeof(int), input, inputf));
    assert(input == fread(val, sizeof(float), input, inputf));
    if (myid == 0) {
      tempfeat = new float[neuron * (long)batch];
#pragma omp parallel for
      for (long n = 0; n < neuron * (long)batch; n++)
        tempfeat[n] = 0.0;
#pragma omp parallel for
      for (int n = 0; n < input; n++)
        if (col[n] - 1 < batch)
          tempfeat[(col[n] - 1) * (long)neuron + row[n] - 1] = val[n];
    }
    fclose(inputf);
    delete[] row;
    delete[] col;
    delete[] val;
  }

  {
    size_t numBytes = size_t(neuron) * size_t(mybatch) * sizeof(float);
    std::memcpy(currfeat, tempfeat, numBytes);
  }
#if 0
  int packetsize = 1000;
  MPI_Request *request;
  //MPI_Request request;
  {
    int numpacket = (mybatch+packetsize-1)/packetsize;
    request = new MPI_Request[numpacket];
    for(int packet = 0; packet < numpacket; packet++){
      int size = packetsize;
      if((packet+1)*packetsize>mybatch)
        size = mybatch%size;
      MPI_Irecv(currfeat+packet*packetsize*(long)neuron,sizeof(float)*size*neuron,MPI_BYTE,0,0,MPI_COMM_WORLD,request+packet);
    }
    //MPI_Irecv(currfeat,mybatch*neuron,MPI_FLOAT,0,0,MPI_COMM_WORLD,&request);
  }
  if(myid==0){
    long displ = 0;
    for(int p = 0; p < numproc; p++){
      int numpacket = (numbatch[p]+packetsize-1)/packetsize;
      for(int packet = 0; packet < numpacket; packet++){
        int size = packetsize;
        if((packet+1)*packetsize>numbatch[p])
          size = numbatch[p]%size;
        MPI_Ssend(tempfeat+displ+packet*packetsize*(long)neuron,sizeof(float)*size*neuron,MPI_BYTE,p,0,MPI_COMM_WORLD);
      }
      //MPI_Ssend(tempfeat+displ,numbatch[p]*neuron,MPI_FLOAT,p,0,MPI_COMM_WORLD);
      displ += numbatch[p]*(long)neuron;
    }
  }

  {
    int numpacket = (mybatch+packetsize-1)/packetsize;
    MPI_Waitall(numpacket,request,MPI_STATUS_IGNORE);
    delete[] request;
  }
  //MPI_Wait(&request,MPI_STATUS_IGNORE);
#endif
  if (myid == 0) {
    delete[] tempfeat;
  }
}
