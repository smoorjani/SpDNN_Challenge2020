#include "vars.h"
#include <cassert>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>

// QUESTION: minibatch is batch? (B, N, M)
#define MINIBATCH 12

// int *categories;
// int *globalcategories;

int main(int argc, char **argv) {

  int myid = 0;
  int numproc = 1;
  int numthreads;

#pragma omp parallel
  {
#pragma omp single
    numthreads = omp_get_num_threads();
  }

  std::string dataset(argv[1]);
  int neuron = atoi(argv[2]);
  // int layer = atoi(argv[3]);
  int batch = atoi(argv[3]);
  int input = atoi(argv[4]);
  int bias = atoi(argv[5]);

  int *numbatch = new int[numproc];
  int *batchdispl = new int[numproc + 1];
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
  int mybatch = numbatch[myid];
  int extbatch = (mybatch + MINIBATCH - 1) / MINIBATCH * MINIBATCH;

  int *csrdispl = new int*;
  unsigned short *csrindex = new unsigned short*;
  float *csrvalue = new float*;

  float *currfeat = new float[neuron * (long)mybatch];
  float *nextfeat = new float[neuron * (long)mybatch];

  std::cout << std::flush;
  readinput(dataset, neuron, input, batch, mybatch, numbatch);
  std::cout << std::flush;

  // setting up GPU- moving global variables to execution-only local context
  int *buffdispl;
  int *mapdispl;
  int *warpdispl;
  unsigned short *map;
  unsigned short *warpindex;
  float *warpvalue;

  int *buffdispl_d;
  int *mapdispl_d;
  int *warpdispl_d;

  unsigned short *map_d;
  unsigned short *warpindex_d;
  float *warpvalue_d;

  float *currfeat_d;
  float *nextfeat_d;
  int *active;
  int *active_d;

  setup_gpu(neuron, myid, mybatch, numproc, extbatch, numthreads, csrdispl, csrindex,
            csrvalue, batchdispl, buffdispl, mapdispl, warpdispl, map, warpindex, warpvalue,
            buffdispl_d, mapdispl_d, warpdispl_d, map_d, warpindex_d, warpvalue_d,
            currfeat, nextfeat, currfeat_d, nextfeat_d);

  infer_gpu(map_d, warpindex_d, warpvalue_d, mybatch, active, active_d, nextfeat_d, currfeat_d,
            buffdispl_d, mapdispl_d, warpdispl_d, bias, neuron, active_d);

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
