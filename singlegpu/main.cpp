#include "vars.h"
#include <cassert>
#include <cstring>

// QUESTION: minibatch is batch? (B, N, M)
#define MINIBATCH 12

int neuron;
int layer;
int batch;
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

  char *chartemp;
  chartemp = getenv("NEURON");
  neuron = atoi(chartemp);
  chartemp = getenv("LAYER");
  layer = atoi(chartemp);
  chartemp = getenv("BATCH");
  batch = atoi(chartemp);
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

  // QUESTION: Matrix A
  csrdispl = new int *[layer];
  csrindex = new unsigned short *[layer];
  csrvalue = new float *[layer];
  // QUESTION: Matrix B
  currfeat = new float[neuron * (long)mybatch];
  // QUESTION: Matrix C
  nextfeat = new float[neuron * (long)mybatch];

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
