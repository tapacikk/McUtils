#ifndef RYNLIB_FFIDYNAMICLIBRARY_HPP
#define RYNLIB_FFIDYNAMICLIBRARY_HPP

#include "FFIModule.hpp"
#include "ffi.h"
using libffi_type = ffi_type;

struct CFuncPtrStub { 
    // only taking the first few fields of a true PyObjectCFuncPtr
    // as the ctypes.h header isn't provided by default
    // and as long as python doesn't change it's parameter packing order
    // this will be fine, but obviously this is playing with _maaajor_ fire
    // and has a decent chance of just dying altogether
    PyObject_HEAD
    char *b_ptr;                /* pointer to memory block */
    int  b_needsfree;           /* need _we_ free the memory? */
};

namespace DynamicFFILibrary {
    // class FFILibraryFunction {
    //     pyobj func;
    // };
    using namespace plzffi;

    template <typename T>
    struct libffi_type_converter { 
        // going for a compile time error here if we ask for a ::value for
        // an uninstantiated template
        // static const libffi_type& value = &L;
    };
    template <> struct libffi_type_converter<unsigned char> { constexpr static libffi_type* const value = &ffi_type_uchar; };
    template <> struct libffi_type_converter<unsigned short> { constexpr static const libffi_type* const value = &ffi_type_ushort; };
    template <> struct libffi_type_converter<unsigned int> { constexpr static const libffi_type* const value = &ffi_type_uint; };
    template <> struct libffi_type_converter<unsigned long> { constexpr static const libffi_type* const value = &ffi_type_ulong; };

    template <> struct libffi_type_converter<char> { constexpr static const libffi_type* const value = &ffi_type_schar; };
    template <> struct libffi_type_converter<short> { constexpr static const libffi_type* const value = &ffi_type_sshort; };
    template <> struct libffi_type_converter<int> { constexpr static const libffi_type* const value = &ffi_type_sint; };
    template <> struct libffi_type_converter<long> { constexpr static const libffi_type* const value = &ffi_type_slong; };

    template <> struct libffi_type_converter<bool> { constexpr static const libffi_type* const value = &ffi_type_uint8; }; // ??

    template <typename T> struct libffi_type_converter<T*> { constexpr static const libffi_type* const value = &ffi_type_pointer; };
    template <> struct libffi_type_converter<std::string> { constexpr static const libffi_type* const value = &ffi_type_pointer; }; // pointer to c_str()

    struct libffi_type_converter_caller {
        template <typename T>
        static libffi_type* call(FFIType type) { return libffi_type_converter<T>::value; }
    };
    auto libffi_type_convert(FFIType type) {
        return FFITypeDispatcher::dispatch<libffi_type_converter_caller>::call(type);
    }

    template <typename T>
    void* libffi_convert(T& val) {
        if constexpr (std::is_pointer_v<T>) {
            return (void*) val;
        } else {
            return (void*) &val; 
        }
    }
    template <>
    void* libffi_convert<std::string>(std::string& val) {
        auto char_buf = val.c_str();
        return libffi_convert<const char*>(char_buf);
    }

    struct libffi_convert_caller {
        template<typename T> 
        static void* call(FFIType type, FFIParameters& params, const std::string& name, const std::vector<size_t>& shape) {
            if (ffiobj::is_array_type(type, shape, FFIContainerType::None)) {
                return libffi_convert<T>(params.value<T*>(name)); // there's no real way to store a vector when coming from python
            } else {
                return libffi_convert<T>(params.value<T>(name));
            }
        }
    };
    void* libffi_convert(FFIType type, FFIParameters& params, const std::string& name, const std::vector<size_t>& shape) {
        auto shape = params.shape(name);
        return FFITypeDispatcher::dispatch<libffi_convert_caller, FFIParameters&, const std::string&, const std::vector<size_t>&>::call(
            type, params, name, shape
        );
    }

    void* extract_func(pyobj ctypes_func) {
        auto cfptr = (CFuncPtrStub*) ctypes_func.obj();
        return *(void **)cfptr->b_ptr; // mimicking ctypes.h
    }

    class LibFFIMethod {
        void* func;
        FFIType rtype;
        std::vector<FFIArgument> args;
        bool vectorized;
        ffi_cif cif;
        bool initialized = false;
    public:
        LibFFIMethod(void* f,  FFIType res_type, std::vector<FFIArgument>& arg_list, bool is_vectorized) :
            func(f), rtype(res_type), args(arg_list), vectorized(is_vectorized) {};
        LibFFIMethod(pyobj& ctypes_func, FFIType res_type, std::vector<FFIArgument>& arg_list, bool is_vectorized) :
            func(extract_func(ctypes_func)), rtype(res_type), args(arg_list), vectorized(is_vectorized) {};
        static LibFFIMethod from_data(pyobj fdata) {

            auto func = fdata.getattr<pyobj>("function");
            auto args = fdata.getattr<std::vector<FFIArgument>>("args");
            auto rtype = fdata.getattr<FFIType>("return_type");
            auto vectorized = fdata.getattr<bool>("vectorized");

            return LibFFIMethod(func, args, rtype, vectorized);
        }

        static std::vector<libffi_type*> get_libffi_types(const std::vector<FFIArgument>& arg_list) {
            std::vector<libffi_type*> types(arg_list.size());
            size_t i = 0;
            for (auto &a : arg_list) {
                if (ffiobj::is_array_type(a.type(), s.shape(), FFIContainerType::None)) {
                    types[i] = &ffi_type_pointer;
                } else {
                    types[i] = libffi_type_convert(a.type);
                }
                i++;
            }
            return types;
        }

        static std::vector<void*> get_libffi_vals(FFIParameters& params, const std::vector<FFIArgument>& arg_list) {
            std::vector<void*> values(arg_list.size());
            size_t i = 0;
            for (auto &a : arg_list) {
                auto name = a.name(); // need const reference
                values[i] = libffi_convert(a.type(), params, name);
                i++;
            }
            return values;
        }


        template <typename restype>
        bool initialize_cif() { // I should really be checking if the rtype has changed...
                auto types = get_libffi_types(args);
                initialized = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, types.size(), libffi_type_converter<rtype>::value, types) == FFI_OK;
                return initialized;
        }

        template <typename restype>
        pyobj call(std::vector<void*>& values) {
            if (!initialized) {
                initialize_cif();
                if (!initialized) throw std::runtime_error("failed to initialize CIF");
            }

            if (vectorized) {
              std::vector<rtype> res;
              ffi_call(&cif, func, &res, values);
              return pyobj::cast<D>(std::move(res));
            } else {
              rtype res;
              ffi_call(&cif, func, &res, values);
              return pyobj::cast<D>(std::move(res));
            }
        }
        pyobj call(FFIParameters& params) {
            auto values = get_libffi_vals(params, args);
            return call(values);
        }

    }

    struct LibFFIMethodCaller {

        // static std::vector<libffi_type*> get_libffi_types(FFIParameters& params);
        // static std::vector<void*> get_libffi_vals(FFIParameters& params);

        template <typename rtype>
        static pyobj call_method(LibFFIMethod& func, FFIParameters& params) {
            return func.call<rtype>(params);
        }
        template <FFIType F>
        static pyobj call_method(LibFFIMethod& func, FFIParameters& params) {
            using T = typename FFITypeMap::find_type<F>;
            return call_method<T>(func, cif, values, vectorized);
        }
        struct call_method_caller {
            template <typename T>
            static auto call(
                [[maybe_unused]] FFIType type, LibFFIMethod& func, FFIParameters& params
                ) {
                return call_method<T>(func, params);
            }
        };
        static auto call_method(
            FFIType type, LibFFIMethod& func, FFIParameters& params
            ) {
            return FFITypeDispatcher::dispatch<call_method>::call(
                type, func, params
            );
        }

    //    template <typename T, typename D>
    //    static pyobj call_threaded(
    //            void* func, ffi_cif& cif, const std::vector<void*>& values,
    //            const std::string& threaded_var, const std::string& mode
    //            ) {
    //         auto val = mod.call_method_threaded<T, D>(method_name, params, threaded_var, mode);
    //         return pyobj::cast<T>(std::move(val)); // we explicitly don't need this value anymore
    //    }
    //    template <typename T>
    //    struct call_threaded_thread_caller {
    //         template <typename D>
    //         static auto call(  // note the loss of type
    //             [[maybe_unused]] FFIType thread_type,
    //             FFIModule& mod, const std::string& method_name, 
    //             const std::string& threaded_var, const std::string& mode,
    //             FFIParameters& params
    //         ) {
    //             return call_threaded<T, D>(mod, method_name, threaded_var, mode, params);
    //         }
    //    };
    //    struct call_threaded_caller { // we resolve the return type first
    //         template <typename T>
    //         static auto call(
    //             [[maybe_unused]] FFIType type, FFIType thread_type,
    //             FFIModule& mod, const std::string& method_name, 
    //             const std::string& threaded_var, const std::string& mode,
    //             FFIParameters& params
    //         ) {
    //             return FFITypeDispatcher::dispatch<call_threaded_thread_caller<T>, FFIModule&, const std::string&, const std::string&, const std::string&, FFIParameters&>::call(
    //                 thread_type,
    //                 mod, method_name, threaded_var, mode, params
    //             );
    //         }
    //     };
    //     static auto call_threaded(
    //         FFIType type, FFIType thread_type, 
    //         FFIModule& mod, const std::string& method_name, 
    //         const std::string& threaded_var, const std::string& mode,
    //         FFIParameters& params
    //     ) {
    //         return FFITypeDispatcher::dispatch<call_threaded_caller, FFIType, FFIModule&, const std::string&, const std::string&, const std::string&, FFIParameters&>::call(
    //             type, thread_type, mod, method_name, threaded_var, mode, params
    //         );
    //     }

    }

    pyobj call_ffi_func(Arguments &base_params) {
        // I could probably link this in better but this is fine as 
        // a stub to demo how these modules work

        auto fdat = base_params.value<pyobj>("function_data");
        auto method = LibFFIMethod::from_data(fdat);
        auto params = base_params.value<FFIParameters>("parameters");
        return LibFFIMethodCaller::call_method(method, params);

    }

        // need a load function that can be called in PYMODINIT
    void load(FFIModule *mod) {

        // add data for vectorized version
        mod->add<pyobj>(
                "call",
                {
                        {"function_data", FFIType::PyObject, {}},
                        {"parameters", FFIType::PyObject, {}},
                },
                call_ffi_func
        );

        // // add data for threaded version
        // mod->add<pyobj>(
        //         "call_threaded",
        //         {
        //                 {"function_data", FFIType::PyObject, {}},
        //                 {"parameters", FFIType::PyObject, {}},
        //         },
        //         call_ffi_func_threaded
        // );

    }

    static FFIModule Data(
        "DynamicFFILibrary",
        "provides dynamic library linkage for plzffi",
        load
        );
}


PyMODINIT_FUNC PyInit_DynamicFFILibrary(void) {
    return DynamicFFILibrary::Data.attach();
}

#endif