

#include "Python.h"
#include "plzffi/FFIModule.hpp"
#include <numeric>
#include <string>
#include <vector>

using namespace plzffi;

// Put all module functions in here
namespace LegacyMBPol {

    static FFIModule Data("LegacyMBPol", "provides linkage for legacy version of MB-Pol");
    static PyModuleDef Module;

    // Declare our linkage -> could be in TTM.hpp, but why bother
    extern "C" {
        double calcpot_(int*, double*, const double*);// comes from libmbpol.so
        double calcpotg_(int *nw, double *Vpot, const double *x, double *g);// comes from libmbpol.so
        }

    // all functions should take a single argument "FFIParameters"
    // data can be extracted by using the `.value` method with the name of the parameter
    double mbpol(FFIParameters &params) {

        auto nwaters = params.value<int>("nwaters");
        auto coords = params.value<double*>("coords");

        double pot_val;
        calcpot_(&nwaters, &pot_val, coords);
        return pot_val / 627.5094740631;

    }

    struct energy_grad {
        double energy;
        double* grad;
        std::vector<std::string> keys = {"energy", "grad"};
        std::vector<FFIType> types = {FFIType::Double, FFIType::Double};
        std::vector<std::vector<int> > shapes = {{}, {0, 3}};
    }

    energy_grad mbpol_grad(FFIParameters &params) {

        auto nwaters = params.value<int>("nwaters");
        auto coords = params.value<double*>("coords");

        double derivs[nwaters * 9];
        double pot_val;

        calcpotg_(&nwaters, &pot_val, coords, &derivs);
        return {
            pot_val / 627.5094740631,
            energy_grad
            }
        }

    }

    std::vector<double> mbpol_vec(FFIParameters &params) {

        auto nwaters = params.value<int>("nwaters");
        auto coords = params.value<double*>("coords");
        auto coords_shape = params.shape("coords");

        std::vector<double> energies(coords_shape[0]);
        auto block_size = std::accumulate(coords_shape.begin()+1, coords_shape.end(), 1, std::multiplies<>());

        for (size_t w = 0; w < coords_shape[0]; w++) {

//            printf("    > calling structure %lu\n", w);

            double pot_val;
            calcpot_(&nwaters, &pot_val, coords);
            energies[w] = pot_val / 627.5094740631;

        }

//        printf("    > sucessfully got energy %f %f %f!\n", energies[0], energies[1], energies[2]);

        return energies;
    }

    // need a load function that can be called in PYMODINIT
    void load() {
        // load modules and return python def

        // add data for first obj
        Data.add<double>(
                "get_pot",
                {
                        {"nwaters", FFIType::Int, {}},
                        {"coords", FFIType::Double, {0, 0, 3}},
                },
                mbpol
                );
        // add data for vectorized version
        Data.add<double>(
                "get_pot_vec",
                {
                        {"nwaters", FFIType::Int, {}},
                        {"coords", FFIType::Double, {0, 0, 3}},
                },
                mbpol_vec
        );

        Data.add<double>(
                "get_pot_vec",
                {
                        {"nwaters", FFIType::Int, {}},
                        {"coords", FFIType::Double, {0, 0, 3}},
                },
                mbpol_vec
        );

        Module = Data.get_def(); // uses new
    }

}

PyMODINIT_FUNC PyInit_TTM(void)
{

    MBPol::load();
    PyObject *m;
    m = PyModule_Create(&MBPol::Module);
    if (m == NULL) { return NULL; }
    MBPol::Data.attach(m);

    return m;
}