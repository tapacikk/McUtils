
#ifndef RYNLIB_FFIMODULE_HPP
#define RYNLIB_FFIMODULE_HPP

#include "PyAllUp.hpp"
#include "FFIParameters.hpp"
#include <string>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h> // comes with -fopenmp
#endif

#ifdef _TBB // Generally one _won't_ have TBB available
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"
#endif


namespace plzffi {

//        template <typename >
//        typedef T (*Func)(const FFIParameters&);

    // data for an FFI method so that FFIModule gets a uniform interface

    auto FFIEmptyCompoundType = FFICompoundType {};
    struct FFIMethodData {
        std::string name;
        std::vector<FFIArgument> args;
        FFIType ret_type;
        FFICompoundType comp_type;
        bool vectorized;
    };
    template<typename T>
    class FFIMethod {
        FFIMethodData data;
        T (*function_pointer)(FFIParameters &);

    public:
        FFIMethod(
                FFIMethodData& data,
                T (*function)(FFIParameters &)
        ) : data(data), function_pointer(function) {};
        FFIMethod(
                std::string &method_name,
                std::vector<FFIArgument> &arg,
                FFIType return_type,
                bool vectorized,
                T (*function)(FFIParameters &)
        ) : data(FFIMethodData{method_name, arg, return_type, FFIEmptyCompoundType,  vectorized}), function_pointer(function) { type_check(); };
        FFIMethod(
                const char *method_name,
                std::vector<FFIArgument> arg,
                FFIType return_type,
                bool vectorized,
                T (*function)(FFIParameters &)
        ) : data(FFIMethodData{method_name, arg, return_type, FFIEmptyCompoundType,  vectorized}), function_pointer(function) { type_check(); };

        FFIMethod(
                std::string &method_name,
                std::vector<FFIArgument> &arg,
                FFICompoundType return_type,
                bool vectorized,
                T (*function)(FFIParameters &)
        ) : data(FFIMethodData{method_name, arg, FFIType::Compound, return_type,  vectorized}), function_pointer(function) { type_check(); };
        FFIMethod(
                const char *method_name,
                std::vector<FFIArgument> arg,
                FFICompoundType return_type,
                bool vectorized,
                T (*function)(FFIParameters &)
        ) : data(FFIMethodData{method_name, arg, FFIType::Compound, return_type,  vectorized}), function_pointer(function) { type_check(); };

        void type_check();

        T call(FFIParameters &params);

        FFIMethodData method_data() { return data; }
        std::string method_name() { return data.name; }
        std::vector<FFIArgument> method_arguments() { return data.args; }
        FFIType return_type() { return data.ret_type; }

//            PyObject * python_signature() {
//                std::vector<PyObject*> py_args(args.size(), NULL);
//                for (size_t i=0; i < args.size(); i++) {
//                    py_args[i] = args[i].as_tuple();
//                }
//                return Py_BuildValue(
//                        "(NNN)",
//                        mcutils::python::as_python<std::string>(name),
//                        mcutils::python::as_python_tuple<PyObject *>(py_args),
//                        mcutils::python::as_python<int>(static_cast<int>(ret_type))
//                        );
//            }

    };

    enum class FFIThreadingMode {
        OpenMP,
        TBB,
        SERIAL
    };

    template<typename T, typename C> // return type and coords data type
    class FFIThreader {
        FFIMethod<T> method;
        FFIThreadingMode mode;
    public:
        FFIThreader(FFIMethod<T> &method, FFIThreadingMode mode) :
        method(method), mode(mode) {}

        FFIThreader(FFIMethod<T> &method, std::string &mode_name) : method(method) {
            if (mode_name == "OpenMP") {
                mode = FFIThreadingMode::OpenMP;
            } else if (mode_name == "TBB") {
                mode = FFIThreadingMode::TBB;
            } else if (mode_name == "serial") {
                mode = FFIThreadingMode::SERIAL;
            } else {
                throw std::runtime_error("FFIThreader: unknown threading method");
            }
        }

        std::vector<T>
        call(FFIParameters &params, std::string &var) {
            auto threaded_param = params.get_parameter(var);
            auto coords = threaded_param.value<C*>();
            auto shape = threaded_param.shape();
            std::vector<T> ret_data(shape[0]); // set up values for return
            switch (mode) {
                case FFIThreadingMode::OpenMP:
                    _call_omp(ret_data, coords, shape, params, var);
                    break;
                case FFIThreadingMode::TBB:
                    _call_tbb(ret_data, coords, shape, params, var);
                    break;
                case FFIThreadingMode::SERIAL:
                    _call_serial(ret_data, coords, shape, params, var);
                    break;
                default:
                    throw std::runtime_error("FFIThreader: unknown threading method");
            }
            return ret_data;
        }

        void _loop_inner(
                std::vector<T>& data, size_t i,
                C* coords, std::vector<size_t>& shape,
                FFIParameters &params, std::string &var
        );
        void _call_serial(
                std::vector<T>& data,
                C* coords, std::vector<size_t>& shape,
                FFIParameters &params, std::string &var);
        void _call_omp(
                std::vector<T>& data,
                C* coords, std::vector<size_t>& shape,
                FFIParameters &params, std::string &var);
        void _call_tbb(
                std::vector<T>& data,
                C* coords, std::vector<size_t>& shape,
                FFIParameters &params, std::string &var);
    };

    template <typename T, typename C>
    void FFIThreader<T, C>::_loop_inner(
            std::vector<T>& data, size_t i,
            C* coords, std::vector<size_t>& shape,
            FFIParameters &params, std::string &var
            ) {

        std::vector<size_t> shp;
        auto new_params = params;
        size_t block_size;
        if (shape.size() > 1) {
            shp = std::vector<size_t>(shape.size()-1);
            block_size = shape[1];
            shp[0] = block_size;
            for (size_t b = 2; b < shape.size(); b++) {
                shp[b-1] = shape[b];
                block_size *= shape[b];
            }
        } else {
            block_size=1;
        }
        auto chunk = coords + (i * block_size);
        auto data_ptr = std::shared_ptr<void>(chunk, [](C*){});
        FFIArgument arg(var, FFITypeHandler<C>().ffi_type(), shp);
        FFIParameter coords_param(data_ptr, arg);
        new_params.set_parameter(var, coords_param);
        auto val = method.call(new_params);

        data[i] = val;
    }

    template <typename T, typename C>
    void FFIThreader<T, C>::_call_serial(
            std::vector<T>& data,
            C* coords, std::vector<size_t>& shape,
            FFIParameters &params, std::string &var) {

        for (size_t w = 0; w < shape[0]; w++) {
            _loop_inner(data, w, coords, shape, params, var);
        }

//            printf(">>>> boopy %f\n", pots.vector()[0]);
    }

    template<typename T, typename C>
    void FFIThreader<T, C>::_call_omp(
            std::vector<T> &data,
            C *coords, std::vector<size_t> &shape,
            FFIParameters &params, std::string &var) {
#ifdef _OPENMP

#pragma omp parallel for
        for (size_t w = 0; w < shape[0]; w++) {
            _loop_inner(data, w, coords, shape, params, var);
        }
#else
        throw std::runtime_error("OpenMP not installed");

#endif
    }

    template<typename T, typename C>
    void FFIThreader<T, C>::_call_tbb(
            std::vector<T> &data,
            C *coords, std::vector<size_t> &shape,
            FFIParameters &params, std::string &var) {
#ifdef _TBB
        tbb::parallel_for(
                tbb::blocked_range<size_t>(0, shape[0]),
                [&](const tbb::blocked_range <size_t> &r) {
                    for (size_t w = r.begin(); w < r.end(); ++w) {
                        _loop_inner(data, w, coords, shape, params, var);
                    }
                }
        );
#else
        throw std::runtime_error("TBB not installed");
#endif
    }

    class FFIModule {
        // possibly memory leaky, but barely so & thus we won't worry too much until we _know_ it's an issue
        std::string name;
        std::string docstring;
        int size = -1; // size of module per interpreter...for future use
        std::string attr = "_FFIModule"; // attribute use when attaching to Python module
        std::string capsule_name;
        std::vector<void *> method_pointers; // pointers to FFI methods, but return types are ambiguous
        // we maintain a secondary cache of this data just because it's easier
        std::vector<FFIMethodData> method_data;
        void (*loader)(FFIModule *mod);
        PyModuleDef module_def;
    public:
        FFIModule() = default;

        FFIModule(std::string &module_name, std::string &module_doc) :
                name(module_name),
                docstring(module_doc) { init(); }
        FFIModule(const char *module_name, const char *module_doc) :
                name(module_name),
                docstring(module_doc) { init(); }

        FFIModule(std::string &module_name, std::string &module_doc, void (*module_loader)(FFIModule *mod)) :
                name(module_name),
                docstring(module_doc),
                loader(module_loader) { init(); }
        FFIModule(const char *module_name, const char *module_doc, void (*module_loader)(FFIModule *mod)) :
                name(module_name),
                docstring(module_doc),
                loader(module_loader) { init(); }

        void init();

        PyObject* create_module() {
            if (loader == NULL) {
                std::string msg = "in loading module " + name + ": ";
                msg += "no module loader defined";
                PyErr_SetString(
                        PyExc_ImportError,
                        msg.c_str()
                );
                return NULL;
            }
            try {
                loader(this);
                get_def();
                return PyModule_Create(&module_def);
            } catch (std::exception &e) {
                std::string msg = "in loading module " + name + ": ";
                msg += e.what();
                PyErr_SetString(
                        PyExc_ImportError,
                        msg.c_str()
                );
                return NULL;
            };
        }

        template<typename T>
        void add_method(FFIMethod<T> &method);

        template<typename T>
        void add(const char *method_name,
                 std::vector<FFIArgument> arg,
                 FFIType return_type,
                 T (*function)(FFIParameters &));
        template<typename T>
        void add(const char *method_name,
                 std::vector<FFIArgument> arg,
                 T (*function)(FFIParameters &));
        void add(
                const char *method_name,
                std::vector<FFIArgument> arg,
                FFICompoundType return_type,
                FFICompoundReturn (*function)(FFIParameters &)
        );

        template<typename T>
        void add(const char *method_name,
                 std::vector<FFIArgument> arg,
                 FFIType return_type,
                 std::vector<T> (*function)(FFIParameters &));
        template<typename T>
        void add(const char *method_name,
                 std::vector<FFIArgument> arg,
                 std::vector<T> (*function)(FFIParameters &));
        void add(
                const char *method_name,
                std::vector<FFIArgument> arg,
                FFICompoundType return_type,
                std::vector<FFICompoundReturn> (*function)(FFIParameters &)
        );

        FFIMethodData get_method_data(std::string &method_name);

        template<typename T>
        FFIMethod<T> get_method(std::string &method_name);

        template<typename T>
        FFIMethod<T> get_method_from_index(size_t i);

        // pieces necessary to hook into the python runtime
        PyObject *get_py_name();

        PyObject *get_capsule();

        bool attach(PyObject *module);

        const char *doc();

        void get_def();

        std::string ffi_module_attr() { return capsule_name; };

        template<typename T>
        T call_method(std::string &method_name, FFIParameters &params) {
            return get_method<T>(method_name).call(params);
        }

        template<typename T, typename C>
        std::vector<T> call_method_threaded(std::string &method_name,
                                            FFIParameters &params, std::string &threaded_var,
                                            std::string &mode) {
            auto meth = get_method<T>(method_name);
            auto wat = FFIThreader<T, C>(meth, mode);
            return wat.call(params, threaded_var);
        }

        size_t get_method_index(std::string &method_name);

        PyObject *python_signature();

        PyObject *py_call_method(PyObject *method_name, PyObject *params);

        PyObject *py_call_method_threaded(PyObject *method_name, PyObject *params, PyObject *looped_var,
                                          PyObject *threading_mode);

    };

    FFIModule ffi_from_capsule(PyObject *capsule);

    //region Template Fuckery
    template<typename T>
    T FFIMethod<T>::call(FFIParameters &params) {
        if (debug_print()) printf("  > calling function pointer on parameters...\n");
        return function_pointer(params);
    }

    template<typename T>
    void FFIMethod<T>::type_check() {
        FFITypeHandler<T>::validate(return_type());
    }

    template<typename T>
    void FFIModule::add_method(FFIMethod<T> &method) {
//        plzffi::set_debug_print(true);
        if (plzffi::debug_print()) {
            printf(" > adding method %s to module %s\n",
                   method.method_data().name.c_str(),
                   name.c_str()
                   );
        }
        method_data.push_back(method.method_data());
        method_pointers.push_back((void *) &method);
    }

    template<typename T>
    void FFIModule::add(
            const char *method_name,
            std::vector<FFIArgument> arg,
            FFIType return_type,
            T (*function)(FFIParameters &)
            ) {
        // TODO: need to introduce destructor to FFIModule to clean up all of these methods once we go out of scope
        auto meth = new FFIMethod<T>(method_name, arg, return_type, false, function);
        add_method(*meth);
    }
    template<typename T>
    void FFIModule::add(
            const char *method_name,
            std::vector<FFIArgument> arg,
            T (*function)(FFIParameters &)
    ) {
        // need to introduce destructor to FFIModule to clean up all of these methods once we go out of scope
        auto constexpr return_type = FFITypeHandler<T>::ffi_type();
        auto meth = new FFIMethod<T>(method_name, arg, return_type, false, function);
        add_method(*meth);
    }

    template<typename T>
    void FFIModule::add(
            const char *method_name,
            std::vector<FFIArgument> arg,
            FFIType return_type,
            std::vector<T> (*function)(FFIParameters &)
    ) {
        auto meth = new FFIMethod<std::vector<T>>(method_name, arg, return_type, true, function);
        add_method(*meth);
    }
    template<typename T>
    void FFIModule::add(
            const char *method_name,
            std::vector<FFIArgument> arg,
            std::vector<T> (*function)(FFIParameters &)
    ) {
        auto constexpr return_type = FFITypeHandler<T>::ffi_type();
        auto meth = new FFIMethod<std::vector<T>>(method_name, arg, return_type, true, function);
        add_method(*meth);
    }

    void FFIModule::add(
            const char *method_name,
            std::vector<FFIArgument> arg,
            FFICompoundType return_type,
            FFICompoundReturn (*function)(FFIParameters &)
    ) {
        // TODO: need to introduce destructor to FFIModule to clean up all of these methods once we go out of scope
        auto meth = new FFIMethod<FFICompoundReturn>(method_name, arg, return_type, false, function);
        add_method(*meth);
    }
    void FFIModule::add(
            const char *method_name,
            std::vector<FFIArgument> arg,
            FFICompoundType return_type,
            std::vector<FFICompoundReturn> (*function)(FFIParameters &)
    ) {
        auto meth = new FFIMethod<std::vector<FFICompoundReturn>>(method_name, arg, return_type, true, function);
        add_method(*meth);
    }

    template<typename T>
    FFIMethod<T> FFIModule::get_method(std::string &method_name) {
//            printf("Uh...?\n");
        for (size_t i = 0; i < method_data.size(); i++) {
            if (method_data[i].name == method_name) {
//                if (debug_print()) printf(" > FFIModuleMethodCaller found appropriate type dispatch!\n");
                if (debug_print()) printf("  > method %s is the %lu-th method in %s\n", method_name.c_str(), i, name.c_str());
                return FFIModule::get_method_from_index<T>(i);
            }
        }
        throw std::runtime_error("method " + method_name + " not found");
    }

    template<typename T>
    FFIMethod<T> FFIModule::get_method_from_index(size_t i) {

        if (debug_print()) printf("  > checking return type...\n");
        FFITypeHandler<T>::validate(method_data[i].ret_type);
        if (debug_print()) printf("  > casting method pointer...\n");
        auto methodptr = static_cast<FFIMethod<T> *>(method_pointers[i]);
        if (methodptr == NULL) {
            std::string err = "Bad pointer for method '%s'" + method_data[i].name;
            throw std::runtime_error(err.c_str());
        }

        auto method = *methodptr;

        return method;
    }

    template <typename...>
    class FFIModuleMethodCaller;
    template<>
    class FFIModuleMethodCaller<> {
    public:
        static PyObject* call(FFIType type, FFIModule& mod, std::string& method, FFIParameters& params) {
            std::string garb =
                    "unhandled type specifier in threaded call to "
                    + method + ": " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
    };
    template <typename T, typename... Args> // expects FFITypePair objects
    class FFIModuleMethodCaller<T, Args...> {
    public:
        static PyObject* call_direct(FFIModule& mod, std::string& method_name, FFIParameters& params) {
            if (debug_print()) printf(" > FFIModuleMethodCaller found appropriate type dispatch!\n");
            PyObject* obj;
            if (mod.get_method_data(method_name).vectorized) {
                if (debug_print()) printf("  > evaluating vectorized potential\n");
                auto val = mod.call_method<std::vector<typename T::type> >(method_name, params);
                if (debug_print()) printf("  > constructing python return value\n");
                auto arr = mcutils::python::as_python<typename T::type>(val);
                if (debug_print()) mcutils::python::print_obj("  > got %s\n", arr);
                obj = mcutils::python::numpy_copy_array(arr);
            } else {
                if (debug_print()) printf("  > evaluating non-vectorized potential\n");
                auto val = mod.call_method<typename T::type>(method_name, params);
                if (debug_print()) printf("  > constructing python return value\n");
                obj = mcutils::python::as_python<typename T::type>(val);
            }
            // need to actually return the values...
            return obj;
        }
        static PyObject* call(FFIType type, FFIModule& mod, std::string& method_name, FFIParameters& params) {
            if (type == T::value) {
                return call_direct(mod, method_name, params);
            } else {
                return FFIModuleMethodCaller<Args...>::call(type, mod, method_name, params);
            }
        }
    };

    template <typename T>
    inline PyObject* ffi_call_method(FFIModule& mod, std::string& method_name, FFIParameters& params) {
        return FFIModuleMethodCaller<T>::call_direct(mod, method_name, params);
    }
    template <FFIType F>
    inline PyObject* ffi_call_method(FFIModule& mod, std::string& method_name, FFIParameters& params) {
        using T = typename FFITypeMap::find_type<F>;
        return FFIModuleMethodCaller<T>::call_direct(mod, method_name, params);
    }
    template <size_t... Idx>
    inline PyObject* ffi_call_method(
            FFIType type, FFIModule& mod, std::string& method_name, FFIParameters& params,
            std::index_sequence<Idx...> inds) {
        return FFIModuleMethodCaller<std::tuple_element_t<Idx, FFITypePairs>...>::call(type, mod, method_name, params);
    }
    inline PyObject* ffi_call_method(
            FFIType type, FFIModule& mod, std::string& method_name, FFIParameters& params) {
        return ffi_call_method(type, mod, method_name, params,
                               std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{});
    }

    class FFIThreaderTypeIterationError : public std::exception {};
    template <typename, typename...>
    class FFIModuleMethodThreadingCaller;
    template<typename T>
    class FFIModuleMethodThreadingCaller<T> {
    public:
        static PyObject* call(FFIType type, FFIType threaded_type,
                              FFIModule& mod, std::string& method_name,
                              std::string& threaded_var, std::string& mode,
                              FFIParameters& params) {
            std::string garb =
                    "unhandled type specifier in calling method "
                    + method_name + ": " + std::to_string(static_cast<int>(threaded_type));
            throw std::runtime_error(garb.c_str());
        }
    };
    template <typename T, typename C, typename... Args> // expects a type and then nested FFITypePair objects
    class FFIModuleMethodThreadingCaller<T, C, Args...> {
    public:
        static PyObject* call(FFIType type, FFIType threaded_type,
                              FFIModule& mod, std::string& method_name,
                              std::string& threaded_var, std::string& mode,
                              FFIParameters& params) {
            if (std::is_same_v<typename C::type, T>) {
                if (threaded_type == C::value) {
                    auto val = mod.call_method_threaded<T, typename C::type>(
                            method_name, params, threaded_var, mode
                    );
                    auto np = mcutils::python::as_python<T>(val);
                    // now copy before returning to put memory on heap
                    auto new_arr = mcutils::python::numpy_copy_array(np);
                    Py_XDECREF(np);
                    return new_arr;
                } else {
                    std::string garb =
                            "type specifier mismatch in threading method "
                            + method_name + ": " + std::to_string(static_cast<int>(threaded_type));
                    throw std::runtime_error(garb.c_str());
                }
            } else {
//                static_assert(sizeof...(Args) > 0, "unhandled type specifier in threading method");
                if (sizeof...(Args) > 0) {
                    return FFIModuleMethodThreadingCaller<T, Args...>::call(type, threaded_type,
                                                                            mod, method_name,
                                                                            threaded_var, mode,
                                                                            params
                    );
                } else {
                    std::string garb =
                            "unhandled type specifier in calling method "
                            + method_name + ": " + std::to_string(static_cast<int>(threaded_type));
                    throw std::runtime_error(garb.c_str());
                }
            }
        }
    };

    // annoying hack to get outer-product of types...
    template <typename T, size_t... Idx>
    inline PyObject* ffi_call_method_threaded_dispatch(
            FFIType type, FFIType threaded_type,
            FFIModule& mod, std::string& method_name,
            std::string& threaded_var, std::string& mode,
            FFIParameters& params,
            std::index_sequence<Idx...> inds) {
        return FFIModuleMethodThreadingCaller<T, std::tuple_element_t<Idx, FFITypePairs>...>::call(
                type, threaded_type, mod,
                method_name, threaded_var, mode, params
        );
    }
    template <typename T>
    inline PyObject* ffi_call_method_threaded_dispatch(
            FFIType type, FFIType threaded_type,
            FFIModule& mod, std::string& method_name,
            std::string& threaded_var, std::string& mode,
            FFIParameters& params
            ) {
        return ffi_call_method_threaded_dispatch<T>(
                type, threaded_type, mod,
                method_name, threaded_var, mode, params,
                std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{}
                );
    }
    template <typename...>
    class FFIModuleMethodThreadingDispatcher;
    template<>
    class FFIModuleMethodThreadingDispatcher<> {
    public:
        static PyObject* call(FFIType type, FFIType threaded_type,
                              FFIModule& mod, std::string& method_name,
                              std::string& threaded_var, std::string& mode,
                              FFIParameters& params) {
            std::string garb =
                    "unhandled type specifier in calling method "
                    + method_name + ": " + std::to_string(static_cast<int>(threaded_type));
            throw std::runtime_error(garb.c_str());
        }
    };
    template <typename T, typename... Args> // expects a type and then nested FFITypePair objects
    class FFIModuleMethodThreadingDispatcher<T, Args...> {
    public:
        static PyObject* call(FFIType type, FFIType threaded_type,
                              FFIModule& mod, std::string& method_name,
                              std::string& threaded_var, std::string& mode,
                              FFIParameters& params) {
            if (type == T::value) {
                return ffi_call_method_threaded_dispatch<typename T::type>(
                        type, threaded_type,
                        mod, method_name,
                        threaded_var, mode,
                        params
                );
            } else {
                return FFIModuleMethodThreadingDispatcher<T, Args...>::call(type, threaded_type,
                                                                            mod, method_name,
                                                                            threaded_var, mode,
                                                                            params);
            }
        }
    };

    template <size_t... Idx>
    inline PyObject* ffi_call_method_threaded(
            FFIType type, FFIType threaded_type,
            FFIModule& mod, std::string& method_name,
            std::string& threaded_var, std::string& mode,
            FFIParameters& params,
            std::index_sequence<Idx...> inds) {
        return FFIModuleMethodThreadingDispatcher<std::tuple_element_t<Idx, FFITypePairs>...>::call(
                type, threaded_type,
                mod, method_name,
                threaded_var, mode, params
                );
    }
    inline PyObject* ffi_call_method_threaded(
            FFIType type, FFIType threaded_type,
            FFIModule& mod, std::string& method_name,
            std::string& threaded_var, std::string& mode,
            FFIParameters& params
            ) {
        return ffi_call_method_threaded(
                type, threaded_type,
                mod, method_name,
                threaded_var, mode, params,
                std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{});
    }

    //endregion

    PyObject *_pycall_python_signature(PyObject *self, PyObject *args);
    PyObject *_pycall_module_name(PyObject *self, PyObject *args);
    PyObject *_pycall_evaluate_method(PyObject *self, PyObject *args);
    PyObject *_pycall_evaluate_method_threaded(PyObject *self, PyObject *args);



    // This used to be in FFIModule.cpp but I'm going header-only

    using mcutils::python::py_printf;

    void FFIModule::init() {
        capsule_name = name + "." + attr;
    }

    PyObject *FFIModule::get_capsule() {
//            auto full_name = ffi_module_attr();
//            printf("wat %s\n", capsule_name.c_str());
        auto cap = PyCapsule_New((void *) this, capsule_name.c_str(), NULL); // do I need a destructor?
        return Py_BuildValue(
                "(NN)",
                get_py_name(),
                cap
        );
    }

    PyObject *FFIModule::get_py_name() {
        return mcutils::python::as_python<std::string>(name);
    }

    bool FFIModule::attach(PyObject *module) {
        PyObject *capsule = get_capsule();
        if (capsule == NULL) return false;
        bool i_did_good = (PyModule_AddObject(module, attr.c_str(), capsule) == 0);
        if (!i_did_good) {
            Py_XDECREF(capsule);
            Py_DECREF(module);
        } else {
            PyObject *pyname = get_py_name();
            i_did_good = (PyModule_AddObject(module, "name", pyname) == 0);
            if (!i_did_good) {
                Py_XDECREF(capsule);
                Py_XDECREF(pyname);
                Py_DECREF(module);
            }
        }

        return i_did_good;
    }

    const char *FFIModule::doc() {
        return docstring.c_str();
    }

    void FFIModule::get_def() {
        // once I have them, I should hook into python methods to return, e.g. the method names and return types
        // inside the module
        auto *methods = new PyMethodDef[5]; // I think Python manages this memory if def() only gets called once
        // but we'll need to be careful to avoid any memory leaks
        methods[0] = {"get_signature", _pycall_python_signature, METH_VARARGS, "gets the signature for an FFI module"};
        methods[1] = {"get_name", _pycall_module_name, METH_VARARGS, "gets the module name for an FFI module"};
        methods[2] = {"call_method", _pycall_evaluate_method, METH_VARARGS, "calls a method from an FFI module"};
        methods[3] = {"call_method_threaded", _pycall_evaluate_method_threaded, METH_VARARGS, "calls a method from an FFI module using a threading strategey"};
        methods[4] = {NULL, NULL, 0, NULL};
        module_def = {
                PyModuleDef_HEAD_INIT,
                name.c_str(),   /* name of module */
                doc(), /* module documentation, may be NULL */
                size,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
                methods
        };
    }

    PyObject *FFIModule::python_signature() {

        std::vector<PyObject *> py_sigs(method_data.size(), NULL);
        for (size_t i = 0; i < method_data.size(); i++) {

            auto args = method_data[i].args;
            if (debug_print()) py_printf(" > constructing signature for %s\n",
                                         method_data[i].name.c_str());
//                    printf("....wat %lu\n", args.size());
            std::vector<PyObject *> subargs(args.size(), NULL);
            for (size_t j = 0; j < args.size(); j++) {
                subargs[j] = args[j].as_tuple();
            }

            py_sigs[i] = Py_BuildValue(
                    "(NNNN)",
                    mcutils::python::as_python<std::string>(method_data[i].name),
                    mcutils::python::as_python_tuple<PyObject *>(subargs),
                    mcutils::python::as_python<int>(static_cast<int>(method_data[i].ret_type)), // to be python portable
                    mcutils::python::as_python<bool>(method_data[i].vectorized)
            );
        }

        return Py_BuildValue(
                "(NN)",
                mcutils::python::as_python<std::string>(name),
                mcutils::python::as_python_tuple<PyObject *>(py_sigs)
        );

    }

    FFIModule ffi_from_capsule(PyObject *captup) {
//        set_debug_print(true); // temporary debug hack
        if (!PyTuple_Check(captup)) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "FFIModule spec. expected to be a tuple looking like (name, capsule)"
            );
            throw std::runtime_error("bad tuple shiz");
        }

        if (debug_print()) py_printf("Got FFIModule spec \"%s\"\n", mcutils::python::get_python_repr(captup).c_str());
        auto name_obj = PyTuple_GetItem(captup, 0);
        if (name_obj == NULL) throw std::runtime_error("bad tuple indexing");
        if (debug_print())
            py_printf("Pulling FFIModule for module \"%s\"\n", mcutils::python::get_python_repr(name_obj).c_str());
        auto cap_obj = PyTuple_GetItem(captup, 1);
        if (cap_obj == NULL) throw std::runtime_error("bad tuple indexing");
        if (debug_print())
            py_printf("  extracting from capsule \"%s\"\n", mcutils::python::get_python_repr(cap_obj).c_str());
        std::string name = mcutils::python::from_python<std::string>(name_obj);
        std::string doc;
        FFIModule mod(name, doc); // empty module
        if (debug_print()) py_printf("  pulling pointer with name \"%s\"\n", mod.ffi_module_attr().c_str());
        return mcutils::python::from_python_capsule<FFIModule>(cap_obj, mod.ffi_module_attr().c_str());
    }

    size_t FFIModule::get_method_index(std::string &method_name) {
        for (size_t i = 0; i < method_data.size(); i++) {
            if (method_data[i].name == method_name) { return i; }
        }
        throw std::runtime_error("method " + method_name + " not found");
    }

    FFIMethodData FFIModule::get_method_data(std::string &method_name) {
        for (auto data : method_data) {
            if (data.name == method_name) {
//                py_printf("Method %s is the %lu-th method in %s\n", method_name.c_str(), i, name.c_str());
                return data;
            }
        }
        throw std::runtime_error("method " + method_name + " not found");
    }

    PyObject* FFIModule::py_call_method(PyObject *method_name, PyObject *params) {

        if (debug_print()) py_printf("Calling from python ");
        auto mname = mcutils::python::from_python<std::string>(method_name);
        if (debug_print()) py_printf(" into method %s\n", mname.c_str());
        auto meth_idx = get_method_index(mname);
        auto argtype = method_data[meth_idx].ret_type;

        if (debug_print()) py_printf(" > loading parameters...\n");
        auto args = FFIParameters(params);

        if (debug_print()) py_printf(" > calling on parameters...\n");
        return ffi_call_method(
                argtype,
                *this,
                mname,
                args
        );
    }

    PyObject *FFIModule::py_call_method_threaded(PyObject *method_name,
                                                 PyObject *params,
                                                 PyObject *looped_var,
                                                 PyObject *threading_mode
    ) {

        auto mname = mcutils::python::from_python<std::string>(method_name);
        auto meth_idx = get_method_index(mname);
        auto argtype = method_data[meth_idx].ret_type;
        auto args = FFIParameters(params);

        auto varname = mcutils::python::from_python<std::string>(looped_var);
        auto mode = mcutils::python::from_python<std::string>(threading_mode);
        auto thread_var = args.get_parameter(varname);
        auto ttype = thread_var.type();

        return ffi_call_method_threaded(
                argtype,
                ttype,
                *this,
                mname,
                varname, mode,
                args
        );

    }

    PyObject *_pycall_python_signature(PyObject *self, PyObject *args) {

        PyObject *cap;
        auto parsed = PyArg_ParseTuple(args, "O", &cap);
        if (!parsed) { return NULL; }

        try {
            auto obj = ffi_from_capsule(cap);
//            printf("!!!!!!!?????\n");
            auto sig = obj.python_signature();

            return sig;
        } catch (std::exception &e) {
            if (!PyErr_Occurred()) {
                std::string msg = "in signature call: ";
                msg += e.what();
                PyErr_SetString(
                        PyExc_SystemError,
                        msg.c_str()
                );
            }
            return NULL;
        }

    }

    PyObject *_pycall_module_name(PyObject *self, PyObject *args) {

        PyObject *cap;
        auto parsed = PyArg_ParseTuple(args, "O", &cap);
        if (!parsed) { return NULL; }

        try {
            auto obj = ffi_from_capsule(cap);
//            printf(".....?????\n");
            auto name = obj.get_py_name();

            return name;
        } catch (std::exception &e) {
            if (!PyErr_Occurred()) {
                std::string msg = "in module_name call: ";
                msg += e.what();
                PyErr_SetString(
                        PyExc_SystemError,
                        msg.c_str()
                );
            }
            return NULL;
        }

    }

    PyObject *_pycall_evaluate_method(PyObject *self, PyObject *args) {
        PyObject *cap, *method_name, *params;
        bool debug;//, *looped_var, *threading_mode;
        auto parsed = PyArg_ParseTuple(args, "OOOp",
                                       &cap,
                                       &method_name,
                                       &params,
                                       &debug
        );
        if (!parsed) { return NULL; }

        set_debug_print(debug);
        mcutils::python::pyadeeb.set_debug_print(debug);

        try {
            auto obj = ffi_from_capsule(cap);
            return obj.py_call_method(method_name, params);
        } catch (std::exception &e) {
            if (!PyErr_Occurred()) {
                std::string msg = "in module_name call: ";
                msg += e.what();
                PyErr_SetString(
                        PyExc_SystemError,
                        msg.c_str()
                );
            }
            return NULL;
        }

    }

    PyObject *_pycall_evaluate_method_threaded(PyObject *self, PyObject *args) {
        PyObject *cap, *method_name, *params, *looped_var, *threading_mode;
        bool debug;
        auto parsed = PyArg_ParseTuple(args, "OOOOOp", &cap, &method_name, &params, &looped_var, &threading_mode, &debug);
        if (!parsed) { return NULL; }

        set_debug_print(debug);
        mcutils::python::pyadeeb.set_debug_print(debug);

        try {
            auto obj = ffi_from_capsule(cap);
            return obj.py_call_method_threaded(method_name, params, looped_var, threading_mode);
        } catch (std::exception &e) {
            if (!PyErr_Occurred()) {
                std::string msg = "in method call: ";
                msg += e.what();
                PyErr_SetString(
                        PyExc_SystemError,
                        msg.c_str()
                );
            }
            return NULL;
        }

    }

}

#endif //RYNLIB_FFIMODULE_HPP
