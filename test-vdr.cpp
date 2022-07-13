#include <iostream>
#include <cstdio>
#include <string>
#include "cimageslib.h"
// #include "imageslib.h"
#include "vdr.h"

using namespace std;
using namespace CImages;

// Hauptfunktion: Liest ein Bild ein und filtert es.
int main(int argc,char* argv[]) {
  int i, j;
  if (argc<7) {
    cerr << "Aufruf: " << argv[0]
         << " infile kernel outfile alpha tau iterations"
         << endl;
    return 1;
  }
  // Einlesen des Eingabebildes
  Image f (argv[1]);
  // Einlesen des Kerns
  Image h (argv[2]);
  h.setorigin (h.sizex()/2, h.sizey()/2);
  // Initialisieren des Ausgabebildes
  Image u = f;

  // Einlesen der Parameter aus den Kommandozeilenargumenten
  double alpha;
  sscanf(argv[4],"%lf",&alpha);
  double tau;
  sscanf(argv[5],"%lf",&tau);
  int iterations;
  sscanf(argv[6],"%d",&iterations);

  // VDR
  VDR (f, h, u, alpha, 0.01, 5,tau, iterations);

  // Schreiben des Ausgabebildes
  u.writepnm(argv[3]);
  return 0;
}

