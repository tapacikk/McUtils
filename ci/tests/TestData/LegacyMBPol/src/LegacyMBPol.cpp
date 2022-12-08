

#include "Python.h"
#include "plzffi/FFIModule.hpp"
#include <numeric>
#include <string>
#include <vector>

using namespace plzffi;

// Put all module functions in here
// Declare our linkage
double calcpot_(int*, double*, const double*);// comes from libmbpol.so
double calcpotg_(int *nw, double *Vpot, const double *x, double *g);// comes from libmbpol.so

namespace LegacyMBPol {

    // all functions should take a single argument "FFIParameters"
    // data can be extracted by using the `.value` method with the name of the parameter
    double mbpol(FFIParameters &params) {

        auto nwaters = params.value<int>("nwaters");
        auto coords = params.value<double*>("coords");

        double pot_val;
        calcpot_(&nwaters, &pot_val, coords);
        return pot_val / 627.5094740631;

    }

    double test_pot(FFIParameters &params) {

        return 0.0;

    }

    double test_val(FFIParameters &params) {

        auto val = params.value<double>("val");
        return val;

    }



    std::vector<double> mbpol_vec(FFIParameters &params) {

        auto nwaters = params.value<int>("nwaters");
        auto coords = params.value<double*>("coords");
        auto coords_shape = params.shape("coords");

        std::vector<double> energies(coords_shape[0]);
        auto block_size = std::accumulate(coords_shape.begin()+1, coords_shape.end(), 1, std::multiplies<>());

        double pot_val;
        for (size_t w = 0; w < coords_shape[0]; w++) {
            calcpot_(&nwaters, &pot_val, coords + (block_size*w));
            energies[w] = pot_val / 627.5094740631;
        }

//        printf("    > sucessfully got energy %f %f %f!\n", energies[0], energies[1], energies[2]);

        return energies;
    }

    FFICompoundType energy_grad_type {
            {"energy", "grad"},
            {FFIType::Double, FFIType::Double},
            {{}, {0, 3}}
    };

//    FFICompoundType energy_grad_type {
//            {"energy", "grad"},
//            {FFIType::Double, FFIType::Double},
//            {{}, {}}
//    };

//    FFICompoundType energy_grad_type {
//            {"energy"},//, "grad"},
//            {FFIType::Double}//, FFIType::Double},
//    };

//    FFICompoundType energy_grad_type {
//    };

    FFICompoundReturn mbpol_grad(FFIParameters &params) {

        FFICompoundReturn res(energy_grad_type);

        auto nwaters = params.value<int>("nwaters");
        auto coords = params.value<double*>("coords");

        std::vector<double> grad(nwaters*9);
        double pot_val;


        calcpotg_(&nwaters, &pot_val, coords, grad.data());
        pot_val = pot_val / 627.5094740631;
        for (size_t i = 0; i < grad.size(); i++) {
            grad[i] = grad[i] / 627.5094740631; // Convert to Hartree
        }

        res.set<double>("energy", pot_val);
        res.set<double>("grad", std::move(grad));
        res.set_shape("grad", {(size_t)nwaters, 3, 3});

        return res;

    }

    FFICompoundReturn mbpol_grad_vec(FFIParameters &params) {

        FFICompoundReturn res(energy_grad_type);

        auto nwaters = params.value<int>("nwaters");
        auto coords = params.value<double*>("coords");
        auto coords_shape = params.shape("coords");

        std::vector<double> energies(coords_shape[0]);
        std::vector<double> grad(coords_shape[0]*nwaters*9);

        auto block_size = std::accumulate(coords_shape.begin()+1, coords_shape.end(), 1, std::multiplies<>());
        size_t grad_size = nwaters*9;
        size_t grad_offset = 0;
        double pot_val;
        for (size_t w = 0; w < coords_shape[0]; w++) {
            grad_offset = w * grad_size;
            calcpotg_(&nwaters, &pot_val, coords + (block_size*w), grad.data()+grad_offset);
            energies[w] = pot_val / 627.5094740631;
            for (size_t i = 0; i < grad_size; i++) {
                grad[grad_offset+i] = grad[grad_offset+i] / 627.5094740631; // Convert to Hartree
            }
        }

        res.set<double>("energy", pot_val);
        res.set<double>("grad", std::move(grad));
        res.set_shape("grad", {coords_shape[0], (size_t)nwaters, 3, 3});

//        printf("    > sucessfully got energy %f %f %f!\n", energies[0], energies[1], energies[2]);

        return res;
    }

        // need a load function that can be called in PYMODINIT
    void load(FFIModule *mod) {
        // load modules and return python def

        // add data for test pots
        mod->add<double>(
                "test_pot",
                {
                },
                test_pot
                );

        // add data for test pots
        mod->add<double>(
                "test_val",
                {
                    {"val", FFIType::Double, {}},
                },
                test_val
                );

        // add data for first obj
        mod->add<double>(
                "get_pot",
                {
                        {"nwaters", FFIType::Int, {}},
                        {"coords", FFIType::Double, {0, 3, 3}},
                },
                mbpol
                );

        // add data for version with gradient
        mod->add(
                "get_pot_grad",
                {
                        {"nwaters", FFIType::Int, {}},
                        {"coords", FFIType::Double, {0, 3, 3}},
                },
                energy_grad_type,
                mbpol_grad
        );

        // add data for vectorized version
        mod->add<double>(
                "get_pot_vec",
                {
                        {"nwaters", FFIType::Int, {}},
                        {"coords", FFIType::Double, {0, 0, 3, 3}},
                },
                mbpol_vec
        );

        // add data for vectorized version with gradient
        mod->add(
                "get_pot_grad_vec",
                {
                        {"nwaters", FFIType::Int, {}},
                        {"coords", FFIType::Double, {0, 0, 3, 3}},
                },
                energy_grad_type,
                mbpol_grad_vec
        );

    }

    static FFIModule Data(
        "LegacyMBPol",
        "provides linkage for legacy version of MB-Pol",
        load
        );
}

PyMODINIT_FUNC PyInit_LegacyMBPol(void) {
    return LegacyMBPol::Data.attach();
}