#ifndef DIFFUSIVITIES_H
#define DIFFUSIVITIES_H

using namespace std;

// linear diffusivity
double lindiffusivity (double squaredgradient);

// Perona-Malik diffusivity
double pmdiffusivity (double squaredgradient, double lambda);

// TV diffusivity
double tvdiffusivity (double squaredgradient, double eps = 0.01);

#endif // #ifndef DIFFUSIVITIES_H
