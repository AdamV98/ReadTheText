#include "vdr.h"
#include "diffusivities.h"
#include "dtpenalisers.h"
#include <cmath>
#include <vector>

void VDR (
  const Image& f,              // blurred image
  const Image& h,              // kernel
  Image& u,			          // deconvolved image
  double alpha,                // regularisation weight
  double dteps,                // regularisation of L1 penalty in data term
  double uteps,               // regularisation of TV penalty in regul term
  double tau,
  int kmax )
{
  return VDR (f, h, u, alpha,
               [dteps] (double s) -> double
                   { return sqrtweight (s, dteps); },
               [uteps] (double s2) -> double
                   { return pmdiffusivity (s2, uteps); },tau,
               kmax);
}

void VDR (
  const Image& f0,             // blurred image
  const Image& h,              // kernel
  Image& u,                    // deconvolved image
  double alpha,                // utularisation weight
  function <double (double)> phideriv, // data term penaliser weight
  function <double (double)> psideriv, // utulariser term diffusivity
  double tau,
  int kmax )                   // iteration count
{
  // time savers
  int nx = f0.sizex(), ny = f0.sizey(), nc = f0.noc();
  Image f = f0;

  // suppress zeros
  for (int i=0; i<nx; ++i)
    for (int j=0; j<ny; ++j)
      for (int cc=0; cc<nc; ++cc) {
        if (f(i,j,cc)<0.01) f(i,j,cc) = 0.01;
        if (u(i,j,cc)<0.01) u(i,j,cc) = 0.01;
      }

  // prepare kernel for fconvolve
  Image hr (nx, ny);
  Image hi (nx, ny);
  padandshiftkernel (h, hr);
  fft2D (hr, hi);
  normaliseftkernel (hr, hi);

  // prepare auxiliary images
  Image uh (nx, ny, nc);          // u # h
  Image dtw (nx, ny);             // data term weight Phi'(res)
  Image dt (nx, ny,nc);             // data term
  Image ux (nx+1, ny+1, nc);      // derivative u_x
  Image uy (nx+1, ny+1, nc);      // derivative u_y
  Image g (nx+1, ny+1);           // diffusivity field for utulariser
  Image inter (nx, ny,nc);
  Image unew(nx, ny,nc);

  // iterate
  for (int k=0; k<kmax; ++k) {
    cerr << ".";
    // uh := u # h
fconvolve (u, hr, hi, uh);
    for (int i=0; i<nx; ++i)
          for (int j=0; j<ny; ++j)
            for (int cc=0; cc<nc; ++cc) {
             inter(i,j,cc)=uh(i,j,cc)-f(i,j,cc); //err
            }
    // compute dataterm
/*////////////////////////////////////////////////////////////////////////////////////////////////*/
    for (int i=0; i<nx; ++i)
      for (int j=0; j<ny; ++j)
        {
        dtw (i, j) = phideriv (inter(i, j) * inter(i, j)) * inter(i, j);
        }
      fconvolveadj (dtw, hr, hi, dt); //data term
/*//////////////////////////////////////////////////////////////////////////////////////////////////*/
    // compute utulariser
    for (int i=0; i<nx+1; ++i)
      for (int j=0; j<ny+1; ++j) {
        double gradu2 = 0.0;
        for (int cc=0; cc<nc; ++cc) {
          // ux, uy approximate derivatives of u at location
          // (i-1/2, j-1/2)
          ux (i, j, cc) = 0.5 * (  u (i, j,   cc) - u (i-1, j, cc)
                                 + u (i, j-1, cc) - u (i-1, j-1, cc));
          uy (i, j, cc) = 0.5 * (  u (i, j,   cc) + u (i-1, j,   cc)
                                 - u (i, j-1, cc) - u (i-1, j-1, cc));
          gradu2 +=   ux (i, j, cc) * ux (i, j, cc)
                    + uy (i, j, cc) * uy (i, j, cc);
        }
        g (i, j) = psideriv (gradu2);
        for (int cc=0; cc<nc; ++cc) {
          ux (i, j, cc) *= g (i, j);
          uy (i, j, cc) *= g (i, j);
        }
      }
    for (int i=0; i<nx; ++i)
      for (int j=0; j<ny; ++j)
        for (int cc=0; cc<nc; ++cc) {
          double ut = -dt (i, j)+ 0.5 * alpha * (  ux (i+1, j+1, cc) + ux (i+1, j, cc)
                                      - ux (i,   j+1, cc) - ux (i,   j, cc)
                                      + uy (i+1, j+1, cc) - uy (i+1, j, cc)
                                      + uy (i,   j+1, cc) - uy (i,   j, cc));


          unew(i,j,cc) = u(i,j,cc) + tau * ut;

        }
    u.valcopy (unew);
  }

  return;
}

