#include <cstdio>
#include <cstdlib>
#include <omp.h>

using namespace std;

void readweights();
void preproc();
void readinput();

void setup_gpu();
void final_gpu();
void infer_gpu(int);

// #define OUTOFCORE //COMMENT THIS OUT IF YOU HAVE ENOUGH MEMORY
// #define OVERLAP //WORKS ONLY WHEN OUTOFCORE IS ENABLED



