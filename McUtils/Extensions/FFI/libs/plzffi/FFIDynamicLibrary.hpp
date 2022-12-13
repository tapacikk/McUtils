#ifndef RYNLIB_FFIDYNAMICLIBRARY_HPP
#define RYNLIB_FFIDYNAMICLIBRARY_HPP

#include "FFIModule.hpp"
#include "ffi.h"
using libffi_type = ffi_type;

namespace plzffi {
    // class FFILibraryFunction {
    //     pyobj func;
    // };

    template <typename T, libffi_type L>
    struct libffi_type_converter { 
        // going for a compile time error here if we ask for a ::value for
        // an uninstantiated template
        static const libffi_type& value = &L;
    }
    template <> libffi_type_converter<unsigned char, ffi_type_uchar> {};
    template <> libffi_type_converter<unsigned short, ffi_type_ushort> {};
    template <> libffi_type_converter<unsigned int, ffi_type_uint> {};
    template <> libffi_type_converter<unsigned long, ffi_type_ulong> {};

    template <> libffi_type_converter<char, ffi_type_schar> {};
    template <> libffi_type_converter<short, ffi_type_sshort> {};
    template <> libffi_type_converter<int, ffi_type_sint> {};
    template <> libffi_type_converter<long, ffi_type_slong> {};

    template <> libffi_type_converter<bool, ffi_type_uint8> {}; // ??

    template <T> libffi_type_converter<T*, ffi_type_pointer> {};
    template <> libffi_type_converter<std::string, ffi_type_pointer> {}; // pointer to c_str()

    template <typename T>
    void* libffi_convert(T& val) {
        if (std::is_pointer_v<T>) {
            return (void*)(T);
        } else {
            return (void*)(&T); 
        }
    }
    template <>
    void* libffi_convert<std::string>(std::string& val) { return libffi_convert<const char*>(val.c_str()); }

    class LibFFICaller {
        pyobj func;
        std::vector<FFIArgument>
    }

    int call_ffi() {
        ffi_cif cif;
        ffi_type *args[1];
        void *values[1];
        char *s;
        ffi_arg rc;
        
        /* Initialize the argument info vectors */    
        args[0] = &ffi_type_pointer;
        values[0] = &s;
        
        /* Initialize the cif */
        if (ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1, 
                    &ffi_type_sint, args) == FFI_OK)
            {
            s = "Hello World!";
            ffi_call(&cif, puts, &rc, values);
            /* rc now holds the result of the call to puts */
            
            /* values holds a pointer to the function's arg, so to 
                call puts() again all we need to do is change the 
                value of s */
            s = "This is cool!";
            ffi_call(&cif, puts, &rc, values);
            }
        
        return 0;
        }
        
}

#endif