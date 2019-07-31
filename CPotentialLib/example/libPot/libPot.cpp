#include "Python.h"
#include <vector>
#include <string>
#include "shared_lib.h"

// We predeclare this potential, saying that it'll be hooked in from some other library in reality

double libPot(std::vector<std::vector<double> > coords, std::vector<std::string> atoms) {
    return sharedLibraryPot(coords, atoms);
}

/******************** FIND AND REPLACE ON libPot FROM HERE ON OUT ****************************/
// Everything else can be handled through find-replace
PyObject *_WrapFunction( void *ptr) {
    PyObject *link_cap = PyCapsule_New(ptr, "potential", NULL);
    if (link_cap == NULL) {
        PyErr_SetString(PyExc_TypeError, "couldn't create capsule object");
        return NULL;
    } else if (!PyCapsule_IsValid(link_cap, "potential")) {
        PyErr_SetString(PyExc_ValueError, "couldn't add pointer to invalid capsule object");
        Py_XDECREF(link_cap);
        return NULL;
    }
    return link_cap;
}

static PyMethodDef libPotMethods[] = {
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION > 2

const char libPot_doc[] = "exposes libPot as a potential";
static struct PyModuleDef libPotModule = {
    PyModuleDef_HEAD_INIT,
    "libPot",   /* name of module */
    libPot_doc, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
   libPotMethods
};

PyMODINIT_FUNC PyInit_libPot(void)
{
    PyObject * m = PyModule_Create(&libPotModule);
    if ( m == NULL ) return NULL;
    PyModule_AddObject(m, "potential", _WrapFunction( (void *) *libPot ));

    return m;
}
#else

PyMODINIT_FUNC initlibPot(void)
{
    m = Py_InitModule("libPot", libPotMethods);
    if ( m == NULL ) return NULL;
    PyModule_AddObject(m, "potential", _WrapFunction( (void *) *libPot ));
}

#endif
