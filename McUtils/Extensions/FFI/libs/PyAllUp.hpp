
#ifndef RYNLIB_PYALLUP_HPP
#define RYNLIB_PYALLUP_HPP

#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include <vector>
#include <numeric>
#include <string>
#include <stdexcept>


// Stack Overflow helper for demangling...
#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

namespace mcutils {
    std::string demangle(const char *name) {

        int status = -4; // some arbitrary value to eliminate the compiler warning
        std::unique_ptr<char, void (*)(void *)> res{
                abi::__cxa_demangle(name, NULL, NULL, &status),
                std::free
        };

        return (status == 0) ? res.get() : name;
    }

    template <typename T>
    struct type_name {
        static std::string value;
        static const char* raw_name;
        static const char* c_str() {
            return value.c_str();
        }
    };
    template <typename T>
    auto type_name<T>::value = demangle(typeid(T).name());
    template <typename T>
    auto type_name<T>::raw_name = typeid(T).name();

}

#else

namespace mcutils {
// does nothing if not g++
    std::string demangle(const char* name) {
        return name;
    }

    template <typename T>
    std::string type_name() {
        return demangle(typeid(T).name());
    }

}


#endif

namespace mcutils {
    namespace python {

        enum DebugLevel {
            Quiet = 0,
            Normal = 5,
            More = 10,
            All = 1
        };

        class PyallupDebugFuckery {
        public:
            static int debug_level;
#ifdef _NODEBUG
            static bool debug_print(int level) {
                return false;
            }
#else
            static bool debug_print(int level) {
                return level <= debug_level;
            }
#endif
            static bool debug_print() {
                return debug_print(DebugLevel::Normal);
            }
            static void set_debug_level(int level) {
                debug_level = level;
            }
        };

        using pyadeeb = PyallupDebugFuckery;
        auto pyadeeb::debug_level = static_cast<int>(DebugLevel::Quiet);
#ifdef _NODEBUG
        void py_printf(const char* fmt, ...) {}
        void py_printf(int debug_level, const char* fmt) {}
#else
        void py_printf(const char* fmt, ...) {
            va_list args;
            va_start(args, fmt);
//            printf("PRINTING: ");
            auto print_err = vprintf(fmt, args);
            va_end(args);
            if (print_err <= 0) {
                std::string msg = "printf shat the bed with ferror = " + std::to_string(print_err);
                PyErr_SetString(
                        PyExc_SystemError,
                        msg.c_str()
                        );
                throw std::runtime_error(msg);
            }
        }
        void py_printf(int debug_level, const char* fmt) {
            if (pyadeeb::debug_print(debug_level)) {
                py_printf(fmt);
            }
        }
#endif

        std::string get_python_repr(PyObject* obj);

        std::string python_type_name(PyTypeObject* type);

        /*
         * PyObject* interface type to keep the raw pointers safe
         *
         *
         */
        class pyiter;

        class pyobj {
            std::shared_ptr<PyObject> ptr;
            bool stolen = false;
        public:
            pyobj() : ptr(NULL) {}
            explicit pyobj(PyObject* py_obj, bool new_ref = false) :
                ptr(
                        std::shared_ptr<PyObject>(
                                py_obj,
                                [](PyObject* ref){
                                    if (ref != NULL && pyadeeb::debug_print(DebugLevel::All) && ref->ob_refcnt == 1) {
                                        py_printf("     --->  freeing shared_ptr to object of type %s at %p\n", ref->ob_type->tp_name, ref);
                                    }
                                    Py_XDECREF(ref);
                                }
                                )
                        ),
                stolen(new_ref) {
                if (!stolen) {
                    Py_XINCREF(ptr.get());
                };
            }
            pyobj(const pyobj& other) : ptr(other.ptr), stolen(other.stolen) {
                // we assume the other object managed this
//                Py_XINCREF(ptr.get());
            }
            pyobj& operator=(const pyobj& other) {
                    if ( &other != this )
                    {
                        this->ptr = other.ptr;
                        this->stolen = other.stolen; // don't want to steal twice
                    }
                    return *this;
            }
            ~pyobj() {
                if (valid() && pyadeeb::debug_print(DebugLevel::All) && refcnt() == 1) {
                    py_printf("     -->  deleting object of type %s at %p\n", type_name().c_str(), obj(), refcnt());
                }
//                Py_XDECREF(ptr); // handled by the shared ptr now
            }

            std::string type_name() {
                return python_type_name(obj()->ob_type);
            }


            PyObject* obj() {return ptr.get();}

            bool operator==(const pyobj &other) {return obj() == other.ptr.get();}
            bool operator!=(const pyobj &other) {return obj() != other.ptr.get();}

            void incref() {Py_XINCREF(obj());}
            void decref() { // don't let PyObject* ref count get to 0 while pyobj still alive
                auto p = ptr.get();
                if (p!=NULL && p->ob_refcnt > 1) {
                    Py_XDECREF(ptr.get());
                }
            }
            void steal() { // mark as stolen ref so refcount can drop when necessary
                if (!stolen) {
                    decref();
                    stolen = true;
                }
            }
            int refcnt() {
                return obj()->ob_refcnt;
            }
            bool valid() {return obj() != NULL;}
            std::string repr() { return get_python_repr(obj()); }

            template<typename T>
            T convert();
            template <typename T>
            static pyobj cast(T val, bool is_new);
            template <typename T>
            static pyobj cast(T val);
            template <typename T>
            static pyobj cast(std::vector<T> val, bool is_new);
            template <typename T>
            static pyobj cast(std::vector<T> val);
//            template <typename T>
//            static pyobj cast(T* val, bool is_new);
//            template <typename T>
//            static pyobj cast(T* val) { return cast<T>(val, false);}

            template<typename T>
            T getattr(std::string& attr);
            template<typename T>
            T getattr(const char* attr) {
                std::string ats = attr;
                return getattr<T>(ats);
            }

//            template<typename T>
//            std::vector<T> getattr_iterable(std::string& attr);
//            template<typename T>
//            std::vector<T> getattr_iterable(std::string attr) {
//                std::string ats = attr;
//                return getattr_iterable<T>(ats);
//            }
//            template<typename T>
//            std::vector<T> getattr_iterable(const char* attr) {
//                std::string ats = attr;
//                return getattr_iterable<T>(ats);
//            }
//            template<typename T>
//            T* getattr_ptr(std::string& attr);
//            template<typename T>
//            T* getattr_ptr(std::string attr) {
//                std::string ats = attr;
//                return getattr_ptr<T>(ats);
//            }
//            template<typename T>
//            T* getattr_ptr(const char* attr) {
//                std::string ats = attr;
//                return getattr_ptr<T>(ats);
//            }

            template<typename T>
            T get_item(size_t idx);
            template<typename T>
            T get_key(const char* key);
            template<typename T>
            T get_key(std::string& key) {
                return get_item<T>(key.c_str());
            }


            pyiter get_iter();
            Py_ssize_t len() { return PyObject_Size(obj()); }

        };

        /*
         * Iterator interface for pyobj
         *
         *
         */
        // Start/end sentinels
        pyobj IteratorStart(PyTuple_New(0));
        pyobj IteratorEnd(PyTuple_New(0));
        class pyiter {
            using iterator_category = std::input_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = pyobj;
            using pointer           = pyobj*;  // or also value_type*
            using reference         = pyobj&;  // or also value_type&

            pyobj consumer;
            pyobj cur;
        public:
            explicit pyiter(pyobj vals) : consumer(vals), cur(IteratorStart) {
                next();
            };
            pyiter(pyobj vals, pyobj cur) : consumer(vals), cur(cur) {};
            bool valid() { return cur.valid(); }

//            explicit operator bool() { return !done; }

            reference operator*() {
//                printf("...*\n");
                return cur;
            }
            pointer operator->() {
//                printf("...->\n");
                return &cur;
            }

            void next() {
                if (!cur.valid()) { throw std::runtime_error("iterator exhausted..."); }
                cur = pyobj(PyIter_Next(consumer.obj()), true);
                if (!cur.valid()) {
//                    printf("...done?\n");
                    cur = IteratorEnd;
                }
            }

            // Prefix increment
            pyiter& operator++() {
//                printf("++...\n");
                next();
                return *this;
            }
            // Postfix increment
            const pyiter operator++(int) {
//                printf("++...\n");
                pyiter tmp = *this;
                ++(*this);
                return tmp;
            }

            bool operator== (pyiter& b) {
//                printf("comparing %s == %s\n", cur.repr().c_str(), b.cur.repr().c_str());
                return cur == b.cur;
            };
            bool operator!= (pyiter& b) {
//                printf("comparing %s != %s\n", cur.repr().c_str(), b.cur.repr().c_str());
                return cur != b.cur;
            };

            pyiter begin() { return *this; }
            pyiter end()   { return pyiter(pyobj(), IteratorEnd); }

        };
        pyiter pyobj::get_iter() {
            auto iter_obj = pyobj(PyObject_GetIter(obj()), true);
            return pyiter(iter_obj);
        }

        // random print function...
        void print_obj(const char* fmt, pyobj obj);


        /*
         * NECESSARY NUMPY FUCKERY
         *
         *
         *
         */
        long* _np_fuckery() {
            if(PyArray_API == NULL) {
                import_array();
                return NULL;
            } else {
                return NULL;
            }
        }
        void _np_init() {
            auto p = _np_fuckery();
            if (p != NULL) throw std::runtime_error("NumPy failed to load");
        }

        void _check_py_arr(PyObject* array) {
            if (!PyArray_Check(array)) {
                std::string msg = "expected numpy array got " + get_python_repr(array);
                PyErr_SetString(PyExc_TypeError, msg.c_str());
                throw std::runtime_error("requires NumPy array");
            }
        }


        /*
         * CONVERSIONS FROM C++ OBJECTS TO PYTHON OBJECTS
         *
         *
         *
         */

        template <typename T>
        class conversion_type_error : public std::runtime_error {
            std::string msg;
        public:
//            conversion_type_error() : std::runtime_error(typeid(T).name()), msg(typeid(T).name()) {};
            explicit conversion_type_error(const char* message) : std::runtime_error(message), msg(message) {};
            explicit conversion_type_error(std::string message) : std::runtime_error(message), msg(message) {};
            static std::string type_id() { return mcutils::type_name<T>::value; }
            const char * what() const noexcept override { return msg.c_str(); }
        };

        template<typename T>
        T from_python(pyobj data) {
            auto typestr = conversion_type_error<T>::type_id();
            std::string msg = "Failed to convert from python for dtype: " + typestr + "\n";
            if (pyadeeb::debug_print(DebugLevel::Normal)) py_printf( "ERROR: Failed to convert from python for dtype %s\n", typestr.c_str());
            throw conversion_type_error<T>(msg);
        }
        template<typename T>
        T from_python(pyobj data, const char* typechar) {
            PyObject* argtup = PyTuple_Pack(1, data.obj());
            T val;
            int successy = PyArg_ParseTuple(argtup, typechar, &val);
            Py_XDECREF(argtup);
            if (!successy || PyErr_Occurred()) throw std::runtime_error("failed to build value...");
            return val;
        }
        template <>
        pyobj from_python<pyobj>(pyobj data) {
            return data;
        }
        template <>
        int from_python<int>(const pyobj data) { return from_python<int>(data, "i"); }
        template <>
        char from_python<char>(const pyobj data) { return from_python<int>(data); } // meh
        template <>
        unsigned char from_python<unsigned char>(const pyobj data) { return from_python<unsigned char>(data, "b"); }
        template <>
        unsigned int from_python<unsigned int>(const pyobj data) { return from_python<unsigned int>(data, "I"); }
        template <>
        short from_python<short>(const pyobj data) { return from_python<short>(data, "h"); }
        template <>
        unsigned short from_python<unsigned short>(const pyobj data) { return from_python<unsigned short>(data, "H"); }
        template <>
        long from_python<long>(const pyobj data) { return from_python<long>(data, "l"); }
        template <>
        unsigned long from_python<unsigned long>(const pyobj data) { return from_python<unsigned long>(data, "k"); }
        template <>
        long long from_python<long long>(const pyobj data) { return from_python<long long>(data, "L"); }
        template <>
        unsigned long long from_python<unsigned long long>(const pyobj data) { return from_python<unsigned long long>(data, "K"); }
        template <>
        float from_python<float>(const pyobj data) { return from_python<float>(data, "f"); }
        template <>
        double from_python<double>(const pyobj data) { return from_python<double>(data, "d"); }
        template <>
        bool from_python<bool>(const pyobj data) { return from_python<int>(data, "p"); }
        template <>
        std::string from_python<std::string >(pyobj data) {
            // we're ditching even the pretense of python 2 support
            PyObject* pyStr = NULL;
            pyStr = PyUnicode_AsEncodedString(data.obj(), "utf-8", "strict");
            if (pyStr == NULL) { throw std::runtime_error("bad python shit"); };
            const char *strExcType =  PyBytes_AsString(pyStr);
            std::string str = strExcType; // data needs to be copied...will this do it?
            Py_XDECREF(pyStr);
            return str;
        }

        template<typename T>
        std::vector<T> from_python_iterable(pyobj data, Py_ssize_t num_els) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting from iterable %s\n", data.repr().c_str());

            std::vector<T> vec(num_els);
            // iterate through list
            auto iterator = data.get_iter();
            if (!iterator.valid()) { throw std::runtime_error("Iteration error"); }
            Py_ssize_t i = 0;
            for (pyobj item:iterator) {
//                printf("item %zd\n", i);
                if (i >= num_els) {
                    std::string msg =
                            "object was expected to have length "
                            + std::to_string(num_els) +
                            " but got more than that";
                    PyErr_SetString(
                            PyExc_ValueError,
                            msg.c_str()
                    );
                    throw std::runtime_error("Iterating beyond len()");
                }
                if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> getting item %zu\n", i);
                if (PyErr_Occurred()) {
                    py_printf(DebugLevel::All, "     --> ...and we failed somehow\n");
                    throw std::runtime_error("Iteration error");
                }
                py_printf(DebugLevel::All, "     --> converting from python types\n");
                vec[i] = item.convert<T>();
                i+=1;
            }
            if (PyErr_Occurred()) {
                throw std::runtime_error("Iteration error");
            }
            if (i < num_els) {
                std::string msg =
                        "object was expected to have length "
                        + std::to_string(num_els) +
                        " but got " + std::to_string(i) + " elements";
                PyErr_SetString(
                        PyExc_ValueError,
                        msg.c_str()
                );
                throw std::runtime_error("Iteration error");
            }

            return vec;
        }
        template<typename T>
        std::vector<T> from_python_iterable(pyobj data) {
            auto num_els = data.len();
            if (num_els < 0) {
                if (!PyErr_Occurred()) {
                    std::string msg = "expected iterable object but got " + data.repr();
                    PyErr_SetString(
                            PyExc_ValueError,
                            msg.c_str()
                    );
                }
                throw std::runtime_error("size issues in iterable %s");
            }
            return from_python_iterable<T>(data, num_els);
        }

        /*
         * Definition of print functions now that we have conversions
         *
         *
         *
         */
        std::string get_python_repr(PyObject* obj) {
            PyObject* repr = PyObject_Repr(obj);
            auto rep = from_python<std::string>(pyobj(repr));
            Py_XDECREF(repr);
            return rep;
        }

        void print_obj(const char* fmt, pyobj obj) {
            auto garb = obj.repr();
            py_printf(fmt, garb.c_str());
        }


        /*
         * CONVERSIONS FROM C++ OBJECTS TO PYTHON OBJECTS
         *
         *
         *
         */

        template<typename T>
        PyObject* as_python_object(T data) {
            auto typestr = conversion_type_error<T>::type_id();
            std::string msg = "Failed to convert to python for dtype: " + typestr + "\n";
            if (pyadeeb::debug_print(DebugLevel::Normal))py_printf( "ERROR: Failed to convert to python for dtype %s\n", typestr.c_str());
            throw conversion_type_error<T>(msg);
        }
        template <>
        PyObject* as_python_object<PyObject*>(PyObject* data) {
            if (pyadeeb::debug_print()) py_printf("     --> Converting as PyObject*");
            return data;
        }
        template <>
        PyObject* as_python_object<char>(char data) { py_printf(DebugLevel::All, "     --> Converting as char\n"); return Py_BuildValue("b", data); }
        template <>
        PyObject* as_python_object<unsigned char>(unsigned char data) { py_printf(DebugLevel::All, "     --> Converting as unsigned char\n"); return Py_BuildValue("B", data); }
        template <>
        PyObject* as_python_object<short>(short data) { py_printf(DebugLevel::All, "     --> Converting as short\n"); return Py_BuildValue("h", data); }
        template <>
        PyObject* as_python_object<unsigned short>(unsigned short data) { py_printf(DebugLevel::All, "     --> Converting as unsigned short\n"); return Py_BuildValue("H", data); }
        template <>
        PyObject* as_python_object<int>(int data) { py_printf(DebugLevel::All, "     --> Converting as int\n"); return Py_BuildValue("i", data); }
        template <>
        PyObject* as_python_object<unsigned int>(unsigned int data) { py_printf(DebugLevel::All, "     --> Converting as unsigned int\n"); return Py_BuildValue("I", data); }
        template <>
        PyObject* as_python_object<long>(long data) { py_printf(DebugLevel::All, "     --> Converting as long\n"); return Py_BuildValue("l", data); }
        template <>
        PyObject* as_python_object<unsigned long>(unsigned long data) { py_printf(DebugLevel::All, "     --> Converting as unsigned long\n"); return Py_BuildValue("k", data); }
        template <>
        PyObject* as_python_object<long long>(long long data) { py_printf(DebugLevel::All, "     --> Converting as long long\n"); return Py_BuildValue("L", data); }
        template <>
        PyObject* as_python_object<unsigned long long>(unsigned long long data) { py_printf(DebugLevel::All, "     --> Converting as unsigned long long\n"); return Py_BuildValue("K", data); }
//        template <>
//        Py_ssize_t convert<Py_ssize_t>(pyobj data) { return PyLong_AsSsize_t(data);  }
//        template <>
//        size_t convert<size_t>(pyobj data) { return PyLong_AsSize_t(data); }
        template <>
        PyObject* as_python_object<float>(float data) { py_printf(DebugLevel::All, "     --> Converting as float\n"); return Py_BuildValue("f", data); }
        template <>
        PyObject* as_python_object<double>(double data) { py_printf(DebugLevel::All, "     --> Converting as double\n"); return Py_BuildValue("d", data); }
        template <>
        PyObject* as_python_object<bool>(bool data) {
            py_printf(DebugLevel::All, "     --> Converting as bool\n");
            if (data) {
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
        }
        template <>
        PyObject* as_python_object<std::string>(std::string data) {
            py_printf(DebugLevel::All, "     --> Converting as string\n");
            return Py_BuildValue("s", data.c_str());
        }
        template <>
        PyObject* as_python_object<const char*>(const char* data) {
            py_printf(DebugLevel::All, "     --> Converting as string\n");
            return Py_BuildValue("s", data);
        }

        template<typename T>
        pyobj as_python(T data) {
            return pyobj(as_python_object<T>(data));
        }
        template<>
        pyobj as_python<pyobj>(pyobj data) {
            return data;
        }
        template <>
        PyObject* as_python_object<pyobj>(pyobj data) {
            return data.obj();
        }

        template<typename T>
        PyObject* as_python_tuple_object(std::vector<T> data, Py_ssize_t num_els) {
            auto tup = PyTuple_New(num_els);
            if (tup == NULL) return NULL; // Assume the error is caught lower down
            for (size_t i = 0; i < data.size(); i++) {
                PyTuple_SET_ITEM(tup, i, as_python_object<T>(data[i]));
            }
            return tup;
        }
        template<typename T>
        PyObject* as_python_tuple_object(std::vector<T> data) {
            return as_python_tuple_object<T>(data, data.size());
        }

        template<typename T>
        pyobj as_python_tuple(std::vector<T> data, Py_ssize_t num_els) {
            return pyobj(as_python_tuple_object<T>(data, num_els));
        }
        template<typename T>
        pyobj as_python_tuple(std::vector<T> data) {
            return pyobj(as_python_tuple_object<T>(data, data.size()));
        }


        PyObject* as_python_dict_object(
                const std::vector<std::string>& keys,
                std::vector<pyobj>& values
        ) {
            // very basic dict builder

            auto dict = PyDict_New();
            for (size_t i = 0; i < keys.size(); i++) {

                if (pyadeeb::debug_print(DebugLevel::All)) {
                    py_printf("     --> Assigning object %s to key %s\n", values[i].repr().c_str(), keys[i].c_str());
                }

                auto k = keys[i];
                if (PyDict_SetItemString(dict, k.c_str(), values[i].obj()) < 0) {
                    Py_XDECREF(dict);
                    return NULL;
                }
            }

            return dict;

        }

        pyobj as_python_dict(
                const std::vector<std::string>& keys,
                std::vector<pyobj>& values
        ) {
            return pyobj(as_python_dict_object(keys, values));
        }

        /*
         * PyCapsule CONVERSIONS
         *
         *
         */

        template<typename T>
        T get_pycapsule_ptr(PyObject* cap, const char* name) {
            auto obj = PyCapsule_GetPointer(cap, name);
            if (obj == NULL) { throw std::runtime_error("Capsule error"); }
            return T(obj); // explicit cast
        }
        template<typename T>
        T get_pycapsule_ptr(PyObject* cap, std::string& name) {
            return get_pycapsule_ptr<T>(cap, name.c_str());
        }
        template<typename T>
        T from_python_capsule(PyObject* cap, const char* name) {
            return *get_pycapsule_ptr<T*>(cap, name); // explicit dereference
        }
        template<typename T>
        T from_python_capsule(PyObject* cap, std::string& name) {
            return from_python_capsule<T>(cap, name.c_str());
        }
        template<typename T>
        T from_python_capsule(pyobj cap, const char* name) {
            return *get_pycapsule_ptr<T*>(cap.obj(), name); // explicit dereference
        }
        template<typename T>
        T from_python_capsule(pyobj cap, std::string& name) {
            return from_python_capsule<T>(cap.obj(), name);
        }
        template<typename T>
        T from_python_capsule(pyobj cap, std::string name) {
            return from_python_capsule<T>(cap.obj(), name);
        }

        /*
         * NUMPY CONVERSIONS
         *
         *
         */

        template<typename T>
        T* get_numpy_data(PyObject *array) {
            _np_init();
            _check_py_arr(array);
            auto py_arr = (PyArrayObject*) array;
            return (T*) PyArray_DATA(py_arr);
        }
        template<typename T>
        T* from_python_buffer(PyObject* data) { // Pointer types _only_ allowed for NumPy arrays
            return get_numpy_data<T>(data);
        }
        template<typename T>
        T* from_python_buffer(pyobj data) { // Pointer types _only_ allowed for NumPy arrays
            return get_numpy_data<T>(data.obj());
        }

        template<typename T>
        NPY_TYPES numpy_type() {
            auto typestr = conversion_type_error<T>::type_id();
            std::string msg = "Failed to convert from python for numpy dtype: " + typestr + "\n";
            if (pyadeeb::debug_print(DebugLevel::Normal)) py_printf( "ERROR: Failed to convert from python for numpy dtype %s\n", typestr.c_str());
            throw conversion_type_error<T>(msg);
        };

        template <>
        NPY_TYPES numpy_type<npy_bool>() { return NPY_BOOL; }
        template <>
        NPY_TYPES numpy_type<npy_int8>() { return NPY_INT8; }
        template <>
        NPY_TYPES numpy_type<npy_int16>() { return NPY_INT16; }
        template <>
        NPY_TYPES numpy_type<npy_int32>() { return NPY_INT32; }
        template <>
        NPY_TYPES numpy_type<npy_int64>() { return NPY_INT64; }
//        template <>
//        NPY_TYPES numpy_type<npy_uint8>() { return NPY_UINT8; }
        template <>
        NPY_TYPES numpy_type<npy_uint16>() { return NPY_UINT16; }
        template <>
        NPY_TYPES numpy_type<npy_uint32>() { return NPY_UINT32; }
        template <>
        NPY_TYPES numpy_type<npy_uint64>() { return NPY_UINT64; }
//        template <>
//        NPY_TYPES numpy_type<npy_float16>() { return NPY_FLOAT16; }
        template <>
        NPY_TYPES numpy_type<npy_float32>() { return NPY_FLOAT32; }
        template <>
        NPY_TYPES numpy_type<npy_float64>() { return NPY_FLOAT64; }
        template <>
        NPY_TYPES numpy_type<npy_float128>() { return NPY_FLOAT128; }
//        template <>
//        NPY_TYPES numpy_type<double>() { return NPY_DOUBLE; }
//        template <>
//        NPY_TYPES numpy_type<float>() { return NPY_FLOAT; }

        std::vector<size_t> manage_shape(size_t num_els, std::vector<size_t>& base_shape) {
            size_t zero_loc = base_shape.size();
            size_t block_size = 1;
            size_t s;
            for (size_t i=0; i < base_shape.size(); i++) {
                s = base_shape[i];
                if (s > 0) {
                    block_size *= s;
                } else if (zero_loc > -1) {
                    throw std::runtime_error("indeterminate shape");
                } else {
                    zero_loc = i;
                }
            }

            if (zero_loc == base_shape.size()) {
                return base_shape;
            } else {
                auto new_shape(base_shape);
                new_shape[zero_loc] = num_els / block_size;
                return new_shape;
            }

        }
        template<typename T>
        PyObject* numpy_object_from_data(
                T* buffer,
                NPY_TYPES dtype,
                std::vector<size_t>& shape,
                bool copy = true
        ) {

            _np_init();
            auto dims = (npy_intp*) shape.data();
            auto nd = shape.size();
//            py_printf("huh fack %lu %lu %lu %lu\n", dims[0], dims[1], dims[2], dims[3]);
            if (copy) {
                if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> creating new numpy array of dtype %s by copying buffer\n", mcutils::type_name<T>::c_str());
                PyObject* arr = PyArray_SimpleNew(
                        nd,
                        dims,
                        dtype
                );
                if (arr == NULL) {
                    throw std::runtime_error("bad numpy shit");
                }
                T* npy_buf = get_numpy_data<T>(arr);
                size_t bytes = sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
                memcpy(npy_buf, buffer, bytes);
 //               auto arr = PyArray_EMPTY(0, 0, dtype, false);
 //               if (arr == NULL) {
 //                   throw std::runtime_error("bad numpy shit");
 //               }
                Py_XINCREF(arr);
                return arr;
            } else {
                if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> creating new numpy array of dtype %s by reusing buffer\n", mcutils::type_name<T>::c_str());

                PyObject* arr = PyArray_SimpleNewFromData(
                        nd,
                        dims,
                        dtype,
                        buffer
                );
                if (arr == NULL) {
                    throw std::runtime_error("bad numpy shit");
                }
                Py_XINCREF(arr);
                return arr;
            }

        }
        template<typename T>
        PyObject* numpy_object_from_data(
                T* buffer,
                std::vector<size_t>& shape,
                bool copy = true
        ) {
            auto npy_type = numpy_type<T>();
            return numpy_object_from_data<T>(buffer, npy_type, shape, copy );
        }
//        template<>
//        PyObject* numpy_object_from_data<double>(
//                double* buffer,
//                std::vector<size_t>& shape,
//                bool copy
//        ) {
////            py_printf("....? %f %f %f\n", buffer[0], buffer[1], buffer[2]);
//            auto wat = numpy_object_from_data<double>(buffer, NPY_FLOAT64, shape, copy );
////            print_obj("wat %s\n", wat);
//            return wat;
//        }

        PyObject* numpy_copy_array(PyObject* obj) {
            _np_init();
            _check_py_arr(obj);
            auto arr = (PyArrayObject*) obj; // I think this is how one builds an array obj...?
            auto descr = PyArray_DESCR(arr);
            Py_XINCREF(descr);
            return PyArray_FromArray(arr, descr, NPY_ARRAY_ENSURECOPY);
        }
        pyobj numpy_copy_array(pyobj obj) {
            return pyobj(numpy_copy_array(obj.obj()));
        }

        template<typename T>
        PyObject* numpy_object_from_data(
                std::vector<T>& vec,
                std::vector<size_t>& shape,
                bool copy = true
        ) {
            if (!vec.empty()) {
                T *buffer = vec.data();
                auto new_shape = manage_shape(vec.size(), shape);
//                std::vector<size_t> new_new_shape(new_shape); // another copy needed???
//                for (auto s:shape) {
//                    printf("--> %lu ", s);
//                }
//                for (auto s:new_shape) {
//                    printf("??? %lu ", s);
//                }
//                std::vector<size_t> new_new_shape = {vec.size()};
//                for (auto s:new_new_shape) {
//                    printf("!!! %lu ", s);
//                }
//                printf("\n");
                return numpy_object_from_data<T>(buffer, new_shape, copy);
            } else {
                npy_intp dims[0];
                auto npy_type = numpy_type<T>();
                auto arr = PyArray_EMPTY(0, dims, npy_type, false);
                return arr;
            }
        }
        template<>
        PyObject* numpy_object_from_data<bool>(
                std::vector<bool>& vec,
                std::vector<size_t>& shape,
                bool copy
        ) {
            if (!vec.empty()) {
                auto nels = vec.size();
                bool buffer[nels];
                auto new_shape = manage_shape(nels, shape);
                for (auto s:new_shape) {
                    printf("???? %zu ", s);
                }
                printf("\n");
                for (size_t i=0; i < nels; i++) { buffer[nels] = vec[i]; }
                return numpy_object_from_data<bool>(buffer, new_shape, copy);
//                return numpy_copy_array(arr); // We'll let other parts of the code-base do copies if they want
            } else {
                npy_intp dims[0];
                auto npy_type = numpy_type<bool>();
                auto arr = PyArray_EMPTY(0, dims, npy_type, false);
                return arr;
            }
        }
        template<typename T>
        PyObject* as_python_object(std::vector<T> data) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting std::vector<%s> to PyObject*\n", mcutils::type_name<T>::c_str());
//            return as_python_tuple_object<T>(data);
            std::vector<size_t> shape = {data.size()};
            return numpy_object_from_data<T>(data, shape);
        }
        template<>
        PyObject* as_python_object<std::vector<pyobj>>(std::vector<pyobj> data) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting std::vector<pyobj> to pyobj\n");
            return as_python_tuple_object<pyobj>(data);
        }
        template<typename T>
        pyobj as_python(std::vector<T> data) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting std::vector<%s> to pyobj\n", mcutils::type_name<T>::c_str());
            return pyobj(as_python_object<T>(data));
        }

        template<typename T>
        pyobj numpy_from_data(
                T* buffer,
                NPY_TYPES dtype,
                std::vector<size_t>& shape,
                bool copy = true
        ) {
            return pyobj(numpy_object_from_data<T>(buffer, dtype, shape, copy));
        }
        template<typename T>
        pyobj numpy_from_data(
                T* buffer,
                std::vector<size_t>& shape,
                bool copy = true
        ) {
            return pyobj(numpy_object_from_data<T>(buffer, shape, copy));
        }
        template<typename T>
        pyobj numpy_from_data(
                std::vector<T>& vec,
                std::vector<size_t>& shape,
                bool copy = true
        ) {

//            if (mcutils::python::pyadeeb::debug_print(DebugLevel::All)) {
//                mcutils::python::py_printf("Converting object of type std::vector<%s> to numpy\n", typeid(T).name());
//            }
            return pyobj(numpy_object_from_data<T>(vec, shape, copy));
        }

        std::vector<size_t> numpy_shape_as_size_t(PyObject* obj) {
            _np_init();
            _check_py_arr(obj);
            auto arr = (PyArrayObject*) obj; // I think this is how one builds an array obj...?
            auto shp = (size_t*) PyArray_SHAPE(arr);
            return std::vector<size_t>(shp, shp + PyArray_NDIM(arr));
        }
        template <typename T>
        std::vector<T> numpy_shape(PyObject* obj) {
            auto base_shape = numpy_shape_as_size_t(obj);
            return std::vector<T>(base_shape.begin(), base_shape.end());
        }
        template <>
        std::vector<size_t> numpy_shape<size_t>(PyObject* obj) {
            return numpy_shape_as_size_t(obj);
        }

        std::string get_py_err_msg() {
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            std::string err_msg;
            if (ptype != NULL) {
                if (pvalue != NULL && ptype != PyExc_SystemError) { err_msg = get_python_repr(pvalue); }
                PyErr_Restore(ptype, pvalue, ptraceback);
            }
            return err_msg;
        }

        /*
         * Full suite of conversions for pyobj using template specification
         *
         *
         */

        template <typename T>
        class PyObject_converter {
            pyobj obj;
        public:
            explicit PyObject_converter(pyobj py_obj) : obj(py_obj) {};
            T value() {
                if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting pyobj to %s\n", mcutils::type_name<T>::c_str());
                return from_python<T>(obj);
            }
        };
        template <typename T>
        class PyObject_converter<std::vector<T>> {
            pyobj obj;
        public:
            explicit PyObject_converter(pyobj py_obj) : obj(py_obj) {};
            std::vector<T> value() {
                if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting pyobj to std::vector<%s>\n", mcutils::type_name<T>::c_str());
                return from_python_iterable<T>(obj);
            }
        };
        template <typename T>
        class PyObject_converter<T*> {
            pyobj obj;
        public:
            explicit PyObject_converter(pyobj py_obj) : obj(py_obj) {};
            T* value() {
                if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting pyobj to %s*\n", mcutils::type_name<T>::c_str());
                return from_python_buffer<T>(obj);
            }
        };
        template <>
        class PyObject_converter<PyObject*> {
            pyobj obj;
        public:
            explicit PyObject_converter(pyobj py_obj) : obj(py_obj) {};
            PyObject* value() {
                if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting pyobj to PyObject*\n");
                return obj.obj();
            }
        };
        template<typename T>
        T pyobj::convert() {
            return PyObject_converter<T>(*this).value();
        }
        template<typename T>
        pyobj pyobj::cast(T val, bool is_new) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> casting %s to pyobj\n", mcutils::type_name<T>::c_str());
            auto obj = mcutils::python::as_python<T>(val);
            if (is_new) {obj.steal();}
            return obj;
        }
        template<typename T>
        pyobj pyobj::cast(T val) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> casting %s to pyobj\n", mcutils::type_name<T>::c_str());
            return mcutils::python::as_python<T>(val);
        }
        template<typename T>
        pyobj pyobj::cast(std::vector<T> val, bool is_new) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> casting std::vector<%s> to pyobj\n", mcutils::type_name<T>::c_str());
            auto obj = mcutils::python::as_python<T>(val);
            if (is_new) {obj.steal();}
            return obj;
        }
        template<typename T>
        pyobj pyobj::cast(std::vector<T> val) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> casting std::vector<%s> to pyobj\n", mcutils::type_name<T>::c_str());
            return mcutils::python::as_python<T>(val);
        }
//        template<typename T>
//        pyobj pyobj::cast(T* val, bool is_new) {
//            auto obj = mcutils::python::as_python<T>(val);
//            if (is_new) {obj.steal();}
//            return obj;
//        }
//        template<typename T>
//        pyobj pyobj::cast(T val) {
//            return pyobj::cast<T>(val, false);
//        }


        /*
         * Attribute extractors
         *
         */
        pyobj get_python_attr_base(PyObject* obj, std::string& attr) {
            if (obj == NULL) {
                if (pyadeeb::debug_print(DebugLevel::Normal)) py_printf("ERROR: object is NULL! (can't get %s)\n", attr.c_str());
                std::string err = "requested attrib " + attr + " from NULL object";
                PyErr_SetString(PyExc_TypeError, err.c_str());
                throw std::runtime_error("no obj issue");
            }
            PyObject* ret = PyObject_GetAttrString(obj, attr.c_str());
            if (ret == NULL) {
                if (pyadeeb::debug_print(DebugLevel::Normal)) py_printf( "ERROR: failed to get attr %s\n", attr.c_str());
                if (get_py_err_msg().empty()) {
                    std::string err = "Requested attrib. \"" + attr + "\" but got NULL back...?";
                    PyErr_SetString(PyExc_AttributeError, err.c_str());
                }
                throw std::runtime_error("attribute error");
            }
            return pyobj(ret);
        }
        template<typename T>
        T pyobj::getattr(std::string& attr) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> getting attribute %s from object\n", attr.c_str());
            return get_python_attr_base(obj(), attr).convert<T>();
        }
        template<typename T>
        T pyobj::get_item(size_t idx) {
            return pyobj(PySequence_GetItem(obj(), idx), true).convert<T>();
        }
        template<typename T>
        T pyobj::get_key(const char* item) {
            return pyobj(PyDict_GetItemString(obj(), item)).convert<T>();
        }

        /*
         * Typename helper
         *
         *
         */
        std::string python_type_name(PyTypeObject* type) {
            if (type == NULL) return "NULL";
            return type->tp_name;
//            if (type == &PyType_Type) return "type";
//            if (type == &PyUnicode_Type) return "string";
//            if (type == &PyLong_Type) return "int";
//            if (type == &PyFloat_Type) return "float";
//            if (type == &PyList_Type) return "list";
//            if (type == &PyTuple_Type) return "tuple";
//            if (type == &PyDict_Type) return "dict";
//            if (type == &PyCapsule_Type) return "capsule";
//            _np_init();
//            if (type == &PyArray_Type) { // needs to go after the _np_init I think...
//                return "np.ndarray";
//            }
//            return std::to_string(reinterpret_cast<unsigned long>(type));
        }
    }

}

#endif //RYNLIB_PYALLUP_HPP
