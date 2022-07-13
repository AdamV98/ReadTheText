#include "diffusivities.h"
#include <cmath>

using namespace std;

// linear diffusivity
double lindiffusivity (double squaredgradient) {
  return 1.0;
}

// Perona-Malik diffusivity
double pmdiffusivity (double squaredgradient, double lambda) {
  return 1.0 / (1.0 + squaredgradient / (lambda*lambda));
}

// TV diffusivity
double tvdiffusivity (double squaredgradient, double eps /* = 0.01 */) {
  return 1.0 / sqrt (squaredgradient + eps);
}

