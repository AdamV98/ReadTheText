#ifndef VDR_H
#define VDR_H

#include <functional>
#include <iostream>
#include "cimageslib.h"
// #include "imageslib.h"
#include "fourier.h"
#include "fconvolvelib.h"

using namespace CImages;
using namespace std;

void VDR (
  const Image& f,              // blurred image
  const Image& h,              // kernel
  Image& u,                    // deconvolved image
  double alpha,                // regularisation weight
  function <double (double)> phideriv, // data term penaliser weight
  function <double (double)> psideriv, // regulariser term diffusivity
  double tau,
  int kmax );                  // iteration count

void VDR (
  const Image& f,              // blurred image
  const Image& h,              // kernel
  Image& u,                    // deconvolved image
  double alpha,                // regularisation weight
  double dteps,                // regularisation of L1 penalty in data term
  double uteps,               // regularisation of TV penalty in regul term
  double tau,
  int kmax );                  // iteration count
#endif // VDR_H

