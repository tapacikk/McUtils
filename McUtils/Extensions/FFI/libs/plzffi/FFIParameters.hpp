
#ifndef RYNLIB_FFIPARAMETERS_HPP
#define RYNLIB_FFIPARAMETERS_HPP

#include "PyAllUp.hpp"
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <type_traits>
#include <algorithm>

namespace plzffi {

    bool debug_print();
    void set_debug_print(bool);

    // Set up enum for type mapping
    // Must be synchronized with the types on the python side
    enum class FFIType {

        GENERIC = -1, // fallback for when things aren't really expected to have a type...

        PY_TYPES = 1000,
        UnsignedChar = PY_TYPES + 10,
        Short = PY_TYPES + 20,
        UnsignedShort = PY_TYPES + 21,
        Int = PY_TYPES + 30,
        UnsignedInt = PY_TYPES + 31,
        Long = PY_TYPES + 40,
        UnsignedLong = PY_TYPES + 41,
        LongLong = PY_TYPES + 50,
        UnsignedLongLong = PY_TYPES + 51,
        PySizeT = PY_TYPES + 60,

        Float = PY_TYPES + 70,
        Double = PY_TYPES + 71,

        Bool = PY_TYPES + 80,
        String = PY_TYPES + 90,
        PyObject = PY_TYPES + 100,

        Compound = PY_TYPES + 500,

        NUMPY_TYPES = 2000,

        NUMPY_Int8 = NUMPY_TYPES + 10,
        NUMPY_UnsignedInt8 = NUMPY_TYPES + 11,
        NUMPY_Int16 = NUMPY_TYPES + 12,
        NUMPY_UnsignedInt16 = NUMPY_TYPES + 13,
        NUMPY_Int32 = NUMPY_TYPES + 14,
        NUMPY_UnsignedInt32 = NUMPY_TYPES + 15,
        NUMPY_Int64 = NUMPY_TYPES + 16,
        NUMPY_UnsignedInt64 = NUMPY_TYPES + 17,

        NUMPY_Float16 = NUMPY_TYPES + 20,
        NUMPY_Float32 = NUMPY_TYPES + 21,
        NUMPY_Float64 = NUMPY_TYPES + 22,
        NUMPY_Float128 = NUMPY_TYPES + 23,

        NUMPY_Bool = NUMPY_TYPES + 30
    };

    struct FFICompoundType {
        std::vector<std::string> keys;
        std::vector<FFIType> types;
        std::vector<std::vector<size_t> > shapes;
    };


    class FFICompoundReturn {
        FFICompoundType type;
        std::vector<std::shared_ptr<void> > data;
    public:
        FFICompoundReturn(FFICompoundType comp_type, std::vector<std::shared_ptr<void> > init_data) : type(comp_type), data(init_data) {};
        explicit FFICompoundReturn(FFICompoundType comp_type) : type(comp_type) {
            data.reserve(type.keys.size());
        }
        size_t key_index(std::string& key);
        template <typename T>
        T get(std::string key);
        template <typename T>
        void set(std::string key, T value);

        PyObject* as_python();
    };
}
// register a conversion for FFIType
namespace mcutils::python {
        template<>
        inline PyObject* as_python<plzffi::FFIType>(plzffi::FFIType data) {
            return as_python<int>(static_cast<int>(data));
        }
        template<>
        inline plzffi::FFIType from_python<plzffi::FFIType>(PyObject* data) {
            return static_cast<plzffi::FFIType>(get_python_attr<int>(data, "value"));
        }

        template<>
        inline PyObject* as_python<plzffi::FFICompoundType>(plzffi::FFICompoundType data) {

            return Py_BuildValue( // Does this memory leak?
                    "{s:O,s:O,s:O}",
                    "keys", as_python_tuple(data.keys),
                    "types", as_python_tuple(data.types),
                    "shapes", as_python_tuple(data.shapes)
            );

        }
        template<>
        inline plzffi::FFICompoundType from_python<plzffi::FFICompoundType>(PyObject* dict) {

            auto keys = PyDict_GetItemString(dict, "keys");
            auto types = PyDict_GetItemString(dict, "types");
            auto shapes = PyDict_GetItemString(dict, "shapes");

            return plzffi::FFICompoundType {
                    from_python<std::vector<std::string> >(keys),
                    from_python<std::vector<plzffi::FFIType> >(types),
                    from_python<std::vector<std::vector<size_t> > >(shapes)
            };
        }

        template<>
        inline PyObject* as_python<plzffi::FFICompoundReturn>(plzffi::FFICompoundReturn data) {
            return data.as_python();
        }
//        template<>
//        inline plzffi::FFICompoundReturn from_python<plzffi::FFICompoundReturn>(PyObject* data) {
//            return data.from_python();
//        }
}

namespace plzffi {

    // we set up an object that can store a sequence
    // of FFITypes
    template <FFIType...ffi_types>
    struct FFITypeTuple {
    };

    template <typename T, FFIType...ffi_types>
    struct FFIValidTypeset{
        using type=T;
        constexpr static const auto valid = std::make_tuple(ffi_types...);
    };
    using FFITypesets = std::tuple<
            FFIValidTypeset<npy_bool, FFIType::NUMPY_Bool, FFIType::NUMPY_UnsignedInt8, FFIType::UnsignedChar>
            ,FFIValidTypeset<npy_int8, FFIType::NUMPY_Int8>
            ,FFIValidTypeset<npy_int16, FFIType::Short, FFIType::NUMPY_Int16>
            ,FFIValidTypeset<npy_int32, FFIType::NUMPY_Int32, FFIType::Int>
            ,FFIValidTypeset<npy_int64, FFIType::NUMPY_Int64, FFIType::Long, FFIType::PySizeT>
            ,FFIValidTypeset<npy_uint16, FFIType::NUMPY_UnsignedInt16, FFIType::NUMPY_Float16, FFIType::UnsignedShort>
            ,FFIValidTypeset<npy_uint32, FFIType::NUMPY_UnsignedInt32, FFIType::UnsignedInt>
            ,FFIValidTypeset<npy_uint64, FFIType::NUMPY_UnsignedInt64, FFIType::UnsignedLong>
            ,FFIValidTypeset<npy_float32, FFIType::NUMPY_Float32, FFIType::Float>
            ,FFIValidTypeset<npy_float64, FFIType::NUMPY_Float64, FFIType::Double>
            ,FFIValidTypeset<npy_float128, FFIType::NUMPY_Float128>
            ,FFIValidTypeset<long long, FFIType::LongLong>
            ,FFIValidTypeset<unsigned long long, FFIType::UnsignedLongLong>
            ,FFIValidTypeset<bool, FFIType::Bool>
            ,FFIValidTypeset<std::string, FFIType::String>
            ,FFIValidTypeset<FFICompoundReturn, FFIType::Compound>
    >;

    // register a conversion for FFIType
//    namespace mcutils::python {
//        template<>
//        inline PyObject* as_python<plzffi::FFIType>(plzffi::FFIType data) {
//            return as_python<int>(static_cast<int>(data));
//        }
//        template<>
//        inline plzffi::FFIType from_python<plzffi::FFIType>(PyObject* data) {
//            return static_cast<plzffi::FFIType>(get_python_attr<int>(data, "value"));
//        }
//    }

    template <typename T>
    struct get_ffi {
        static const FFIType value = FFIType::GENERIC;
    };
    template <FFIType F, typename T>
    struct FFITypePair {
        static const FFIType value = F;
        using type=T;
    };

    // define the mapping between FFIType and true types
    using FFITypePairs = std::tuple<
            FFITypePair<FFIType::PyObject, PyObject*>
            ,FFITypePair<FFIType::UnsignedChar, unsigned char>
            ,FFITypePair<FFIType::Short, short>
            ,FFITypePair<FFIType::UnsignedShort, unsigned short>
            ,FFITypePair<FFIType::Int, int>
            ,FFITypePair<FFIType::UnsignedInt, unsigned int>
            ,FFITypePair<FFIType::Long, long>
            ,FFITypePair<FFIType::UnsignedLong, unsigned long>
            ,FFITypePair<FFIType::LongLong, long long>
            ,FFITypePair<FFIType::UnsignedLongLong, unsigned long long>
            ,FFITypePair<FFIType::PySizeT, Py_ssize_t>
            ,FFITypePair<FFIType::Float, float>
            ,FFITypePair<FFIType::Double, double>
            ,FFITypePair<FFIType::Bool, bool>
            ,FFITypePair<FFIType::String, std::string>
            ,FFITypePair<FFIType::Compound, FFICompoundReturn >
            ,FFITypePair<FFIType::NUMPY_Bool, npy_bool>
            ,FFITypePair<FFIType::NUMPY_Int8, npy_int8>
            ,FFITypePair<FFIType::NUMPY_Int16, npy_int16>
            ,FFITypePair<FFIType::NUMPY_Int32, npy_int32>
            ,FFITypePair<FFIType::NUMPY_Int64, npy_int64>
            ,FFITypePair<FFIType::NUMPY_UnsignedInt8, npy_uint8>
            ,FFITypePair<FFIType::NUMPY_UnsignedInt16, npy_uint16>
            ,FFITypePair<FFIType::NUMPY_UnsignedInt32, npy_uint32>
            ,FFITypePair<FFIType::NUMPY_UnsignedInt64, npy_uint64>
            ,FFITypePair<FFIType::NUMPY_Float16, npy_float16>
            ,FFITypePair<FFIType::NUMPY_Float32, npy_float32>
            ,FFITypePair<FFIType::NUMPY_Float64, npy_float64>
            ,FFITypePair<FFIType::NUMPY_Float128, npy_float128>
    >;


    // thank you again StackOverflow
    template <typename pairs>
    struct FFITypeMapper
    {
//        template <FFIType F>
//        static const FFITypePair find_pair = decltype(get_pair(std::integral_constant<FFIType, F>{}));
//        template <typename T>
//        static const FFITypePair find_pair = decltype(gget_type_pair<T>());
//        template <typename, typename...>
//        static FFIType ffi_typecode();
//        template <typename T>
//        static FFIType ffi_typecode<T>() {
////            throw std::runtime_error("no FFIType found");
//            static_assert(false, "no FFIType found");
////            return FFIType::GENERIC;
//        };
//        template <typename T, typename D, typename... rest>
//        static FFIType ffi_typecode<T, D, rest>() {
////            if (sizeof...(rest) == 0) {
////                static_assert(std::is_same_v<T, typename D::type>, "no FFIType found");
////                return D::value;
////            } else {
//            if (std::is_same_v<T, typename D::type>) {
//                return D::value;
//            } else {
//                return ffi_typecode<T, rest...>();
//            }
////            }
//        };

        template <typename, typename...>
        struct ffi_type_resolver;
//        template <typename T>
//        struct ffi_type_resolver<T> {
//            static constexpr FFIType typecode() {
//                static_assert(false, "can't be here");
//                return FFIType::GENERIC;
//
////                std::string msg = "ERROR: unhandled typename";
////                msg += typeid(T).name();
////                printf("%s\n", msg.c_str());
////                throw std::runtime_error(msg);
//            }
//        };
        template <typename D, typename T, typename... Args>
        struct ffi_type_resolver<D, T, Args...> {
            static constexpr FFIType typecode() {
                if (std::is_same_v<D, typename T::type>) {
                    return T::value;
                } else {
//                    static_assert(sizeof...(Args) > 0, "invalid type resolution");
                    if constexpr (sizeof...(Args) == 0) {
                        throw std::runtime_error(typeid(T).name());
                    } else {
                        return ffi_type_resolver<D, Args...>::typecode();
                    }
                }
            }
        };


        template <typename T, size_t... Idx>
        static constexpr FFIType ffi_typecode(std::index_sequence<Idx...>) {
            return ffi_type_resolver<T, std::tuple_element_t<Idx, pairs>...>::typecode();
        }
        template <typename T>
        static constexpr FFIType ffi_typecode() {
            return ffi_typecode<T>(std::make_index_sequence<std::tuple_size<pairs>{}>{});
        };

        template <FFIType F>
        using find_type = typename decltype(get_pair(std::integral_constant<FFIType, F>{}))::type;

    };
    using FFITypeMap = FFITypeMapper<FFITypePairs>;

    const std::vector<FFIType> FFIPointerTypes {
            FFIType::PyObject,
            FFIType::NUMPY_Bool,
            FFIType::NUMPY_Int8,
            FFIType::NUMPY_Int16,
            FFIType::NUMPY_Int32,
            FFIType::NUMPY_Int64,
            FFIType::NUMPY_UnsignedInt8,
            FFIType::NUMPY_UnsignedInt16,
            FFIType::NUMPY_UnsignedInt32,
            FFIType::NUMPY_UnsignedInt64,
            FFIType::NUMPY_Float16,
            FFIType::NUMPY_Float32,
            FFIType::NUMPY_Float64
    };


    template <typename, typename...>
    struct FFITypeValidator;
    template <typename T>
    struct FFITypeValidator<T> {
        static void validate (FFIType type) {
            std::string msg = "ERROR: unhandled typename/FFIType pair";
            std::string tname = typeid(T).name();
            msg += "(" + tname + "/" + std::to_string(static_cast<int>(type)) + ")";
            printf("%s\n", msg.c_str());
            throw std::runtime_error(msg);
        }
        static FFIType typecode() {
            std::string msg = "ERROR: unhandled typename";
            msg += typeid(T).name();
            printf("%s\n", msg.c_str());
            throw std::runtime_error(msg);
        }
    };
    template <typename D, typename T, typename... Args> // expects FFITypeset objects
    struct FFITypeValidator<D, T, Args...> {
        static void validate (FFIType type) {
            if (std::is_same<D, typename T::type>::value) {
//                auto valids = T::valid;
//                auto nels = std::tuple_size_v<decltype(T::valid)>;
                // https://stackoverflow.com/a/40233852/5720002
                // Functional-programming loop over tuple elements
                // to check if the type is in the held tuple
                // This is basically a bunch of template fuckery
                // just to avoid a huge switch...I think it's worth it
                bool found = std::apply(
                        [type](auto &&... args) {
                            return ( (decltype(args)(args) == type) || ...);
                        },
                        T::valid
                );
                if ( !found ) {
                    throw std::runtime_error("typename/FFIType mismatch");
                }
            } else {
                FFITypeValidator<D, Args...>::validate(type);
            }
        }
        static FFIType typecode() {
            if (std::is_same<D, typename T::type>::value) {
                return T::value;
            } else {
                return FFITypeValidator<D, Args...>::typecode();
            }
        }
    };


    template <typename T, size_t... Idx>
    inline void validate_type(FFIType type, std::index_sequence<Idx...>) {
        return FFITypeValidator<T, std::tuple_element_t<Idx, FFITypesets>...>::validate(type);
    }
    template <typename T>
    inline void validate_type(FFIType type) {
        return validate_type<T>(type, std::make_index_sequence<std::tuple_size<FFITypesets>{}>{});
    }

    template <typename T>
    inline constexpr FFIType ffi_typecode() {
        return FFITypeMap::ffi_typecode<T>();
    }

    template <typename T>
    class FFITypeHandler {
        // class that helps us maintain a map between type codes & proper types
    public:
        static constexpr FFIType ffi_type() { return ffi_typecode<T>(); }
        static void validate(FFIType type_code) { validate_type<T>(type_code); }
        static T cast(FFIType type_code, std::shared_ptr<void> &data) {
            validate(type_code);
            return *static_cast<T *>(data.get());
        }
        static std::shared_ptr<void> genericize(T data) {
//            validate(type_code);

            return std::shared_ptr<void>(
                    &data,
                    [](T* val) { delete val; }
            );

//            auto pos = std::find(FFIPointerTypes.begin(), FFIPointerTypes.end(), type_code);
//            if (pos == FFIPointerTypes.end() || !shape.empty()) {
//                // not a pointer to data so we want to extract with a deleter
//                return std::shared_ptr<void>(
//                        &data,
//                        [](T* val) { delete val; }
//                );
//            } else {
//                // is a pointer so no deletion
//                return std::make_shared<T>(data);
//            }

        }
        static PyObject *as_python(FFIType type_code, std::shared_ptr<void> &data, std::vector<size_t> &shape);
    };

    template <typename T>
    class FFITypeHandler<std::vector<T> > {
        // specialization to handle vector types
    public:
        static constexpr FFIType ffi_type() {return ffi_typecode<T>();}
        static void validate(FFIType type_code) { validate_type<T>(type_code); }
        static std::vector<T> cast(FFIType type_code, std::shared_ptr<void>& data) {
            return *static_cast<std::vector<T>>(data.get());
        }
        static std::shared_ptr<void> genericize(std::vector<T> data) {
            return std::shared_ptr<void>(
                    &data,
                    [](std::vector<T>* val) { delete val; }
            );
        }
        static PyObject* as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape);
    };
    template <>
    class FFITypeHandler<PyObject*> {
        // specialization to handle pointer types
    public:
        static constexpr FFIType ffi_type() {return ffi_typecode<PyObject*>();}
        static void validate(FFIType type_code) { validate_type<PyObject*>(type_code); }
        static PyObject* cast(FFIType type_code, std::shared_ptr<void>& data) {
            return static_cast<PyObject*>(data.get());
        }
        static std::shared_ptr<void> genericize(PyObject* data) {
            return std::make_shared<PyObject*>(data);
        }
        static PyObject* as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape);
    };
    template <typename T>
    class FFITypeHandler<T*> {
        // specialization to handle pointer types
    public:
        static constexpr FFIType ffi_type() {return ffi_typecode<T>();}
        static void validate(FFIType type_code) { validate_type<T>(type_code); }
        static T* cast(FFIType type_code, std::shared_ptr<void>& data) {
            return static_cast<T*>(data.get());
        }
        static std::shared_ptr<void> genericize(T* data) {
            return std::make_shared<T*>(data);
        }
        static PyObject* as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape);
    };

    template <typename T>
    inline PyObject* FFITypeHandler<T>::as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        if (!shape.empty()) {
            return FFITypeHandler<T*>().as_python(type_code, data, shape);
        }
        return mcutils::python::as_python<T>(*static_cast<T*>(data.get()));
    }
    template <typename T>
    inline PyObject* FFITypeHandler<std::vector<T> >::as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        return mcutils::python::as_python<std::vector<T> >(*static_cast<std::vector<T>*>(data.get()));
    }
    template <typename T>
    inline PyObject* FFITypeHandler<T*>::as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
        // we use NumPy for all pointer types
        return mcutils::python::numpy_from_data<T>(static_cast<T*>(data.get()), shape);
    }

    // recursive definition to loop through and test all the type pairs (thank you StackOverflow)
    // we'll embed most conversion functions here so that we don't need to duplicate the boiler plate
    template <typename...>
    class FFIConversionManager;
    template<>
    class FFIConversionManager<> {
    public:
        static std::shared_ptr<void> from_python(FFIType type, PyObject* py_obj, std::vector<size_t>& shape) {
            std::string garb = "unhandled type specifier in converting from python: " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
        static std::shared_ptr<void> from_python_attr(FFIType type, PyObject* py_obj, const char* attr, std::vector<size_t>& shape) {
            std::string garb = "unhandled type specifier in converting from python: " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
        static PyObject* as_python(FFIType type,  std::shared_ptr<void>& data, std::vector<size_t>& shape) {
            std::string garb = "unhandled type specifier in converting to python: " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
    };
    template <typename T, typename... Args> // expects FFITypePair objects
    class FFIConversionManager<T, Args...> {
    public:
        static std::shared_ptr<void> from_python_direct(FFIType type, PyObject* py_obj, std::vector<size_t>& shape) {
            // determine if our type is a pointer type
            auto pos = std::find(FFIPointerTypes.begin(), FFIPointerTypes.end(), type);
            if (pos < FFIPointerTypes.end() || !shape.empty()) {
                if (debug_print()) {
                    auto garb = mcutils::python::get_python_repr(py_obj);
                    mcutils::python::py_printf("Converting PyObject %s with pointer type FFIType %i\n", garb.c_str(), type);
                }
                // not a pointer to data so we want to extract with a deleter
                return std::shared_ptr<void>(
                        mcutils::python::from_python<typename T::type *>(py_obj),
                        [](typename T::type *val) { delete val; }
                );
            } else {
                if (debug_print()) {
                    auto garb = mcutils::python::get_python_repr(py_obj);
                    mcutils::python::py_printf("Converting PyObject %s with non-pointer type FFIType %i\n", garb.c_str(), type);
                }
                // is a pointer so no deletion
                return std::make_shared<typename T::type>(
                        mcutils::python::from_python<typename T::type>(py_obj)
                );
            }
        }
        static std::shared_ptr<void> from_python(FFIType type, PyObject* py_obj, std::vector<size_t>& shape) {
            if (type == T::value) { // the type matches up with our stored type
                return from_python_direct(type, py_obj, shape);
            } else {
                return FFIConversionManager<Args...>::from_python(type, py_obj, shape);
            }
        }

        static std::shared_ptr<void> from_python_attr_direct(FFIType type, PyObject* py_obj, const char* attr, std::vector<size_t>& shape) {
            auto pos = std::find(FFIPointerTypes.begin(), FFIPointerTypes.end(), type);

            if (pos < FFIPointerTypes.end() || !shape.empty()) {
                // not a pointer to data so we want to extract with a deleter
                if (debug_print()) {
                    auto garb = mcutils::python::get_python_repr(py_obj);
                    mcutils::python::py_printf("Converting PyObject %s attr %s with pointer type FFIType %i\n", garb.c_str(), attr, type);
                }
                return std::shared_ptr<void>(
                        mcutils::python::get_python_attr_ptr<typename T::type>(py_obj, attr),
                        [](typename T::type* val){delete val;}
                );
            } else {
                // new managed instance so no deleter
                if (debug_print()) {
                    auto garb = mcutils::python::get_python_repr(py_obj);
                    mcutils::python::py_printf("Converting PyObject %s attr %s with pointer type FFIType %i\n", garb.c_str(), attr, type);
                }
                return std::make_shared<typename T::type>(
                        mcutils::python::get_python_attr<typename T::type>(py_obj, attr)
                );
            }
        }
        static std::shared_ptr<void> from_python_attr(FFIType type, PyObject* py_obj, const char* attr, std::vector<size_t>& shape) {
            if (type == T::value) { // the type matches up with our stored type
                return from_python_attr_direct(type, py_obj, attr, shape);
            } else {
                return FFIConversionManager<Args...>::from_python_attr(type, py_obj, attr, shape);
            }
        }
        static PyObject* as_python_direct(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
            // determine if our type is a pointer type
            auto pos = std::find(FFIPointerTypes.begin(), FFIPointerTypes.end(), type);
            if (pos < FFIPointerTypes.end()) {
                // is a pointer type so we convert regularly
                return FFITypeHandler<typename T::type>::as_python(type, data, shape);
            } else {
                // not a pointer type so we extract data from shared_ptr as a pointer first
                return FFITypeHandler<typename T::type*>::as_python(type, data, shape);
            }
        }
        static PyObject* as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
            if (type == T::value) {
                return as_python(type, data, shape);
            } else {
                return FFIConversionManager<Args...>::as_python(type, data, shape);
            }
        }
    };

//    // A specialization for PyObject*
//    template <typename... Args> // expects FFITypePair objects
//    class FFIConversionManager<FFITypePair<FFIType::PyObject, PyObject*>, Args...> {
//    public:
//        static std::shared_ptr<void> genericize(FFIType type, PyObject* data, std::vector<size_t>& shape) {
//            if (type == FFIType::PyObject) { // the type matches up with our stored type
//                // not a pointer to data so we want to extract with a deleter
//                return std::shared_ptr<void>(
//                        data,
//                        [](PyObject* val){delete val;}
//                );
//            } else {
//                return FFIConversionManager<Args...>::genericize(type, data, shape);
//            }
//        }
//        static std::shared_ptr<void> from_python(FFIType type, PyObject *py_obj, std::vector<size_t> &shape) {
//            if (type == FFIType::PyObject) { // the type matches up with our stored type
//                return std::shared_ptr<void>(
//                        py_obj,
//                        [](PyObject *val) { delete val; }
//                );
//            } else {
//                return FFIConversionManager<Args...>::from_python(type, py_obj, shape);
//            }
//        }
//        static std::shared_ptr<void> from_python_attr(FFIType type, PyObject* py_obj, const char* attr, std::vector<size_t>& shape) {
//            if (type == FFIType::PyObject) { // the type matches up with our stored type
//                return std::shared_ptr<void>(
//                        mcutils::python::get_python_attr_ptr<PyObject>(py_obj, attr),
//                        [](PyObject* val){delete val;}
//                );
//            } else {
//                return FFIConversionManager<Args...>::from_python_attr(type, py_obj, attr, shape);
//            }
//        }
//        static PyObject* as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
//            if (type == FFIType::PyObject) {
//                return FFITypeHandler<PyObject*>().as_python(type, data, shape);
//            } else {
//                // is a pointer type so we convert as a pointer
//                return FFIConversionManager<Args...>::as_python(type, data, shape);
//            }
//        }
//    };

    template <typename T>
    inline PyObject* ffi_to_python(std::shared_ptr<void>&data, std::vector<size_t>& shape) {
        return FFIConversionManager<T>::as_python_direct(FFITypeHandler<T>::typecode(), data, shape);
    }
    template <FFIType F>
    inline PyObject* ffi_to_python(std::shared_ptr<void>&data, std::vector<size_t>& shape) {
        using T = typename FFITypeMap::find_type<F>;
        return FFIConversionManager<T>::as_python_direct(F, data, shape);
    }
    template <size_t... Idx>
    inline PyObject* ffi_to_python(FFIType type, std::shared_ptr<void>&data, std::vector<size_t>& shape,
                                   std::index_sequence<Idx...> inds) {
        return FFIConversionManager<std::tuple_element_t<Idx, FFITypePairs>...>::as_python(type, data, shape);
    }
    inline PyObject* ffi_to_python(const FFIType type, std::shared_ptr<void>&data, std::vector<size_t>& shape) {
        return ffi_to_python(
                type, data, shape,
                std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{}
                );
    };

    template <typename T>
    inline std::shared_ptr<void> ffi_from_python_attr(PyObject* obj, const char* attr, std::vector<size_t>& shape) {
        return FFIConversionManager<T>::from_python_attr_direct(FFITypeHandler<T>::typecode(), obj, attr, shape);
    }
    template <FFIType F>
    inline std::shared_ptr<void> ffi_from_python_attr(PyObject* obj, const char* attr, std::vector<size_t>& shape) {
        using T = typename FFITypeMap::find_type<F>;
        return FFIConversionManager<T>::from_python_attr_direct(F, obj, attr, shape);
    }
    template <size_t... Idx>
    inline std::shared_ptr<void> ffi_from_python_attr(const FFIType type, PyObject* obj, const char* attr, std::vector<size_t>& shape,
                                   std::index_sequence<Idx...> inds) {
        return FFIConversionManager<std::tuple_element_t<Idx, FFITypePairs>...>::from_python_attr(type, obj, attr, shape);
    }
    inline std::shared_ptr<void> ffi_from_python_attr(const FFIType type, PyObject* obj, const char* attr, std::vector<size_t>& shape) {
        return ffi_from_python_attr(
                type, obj, attr, shape,
                std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{}
        );
    };

    template <typename T>
    inline std::shared_ptr<void> ffi_from_python(PyObject* obj, std::vector<size_t>& shape) {
        return FFIConversionManager<T>::from_python_direct(FFITypeHandler<T>::typecode(), obj, shape);
    }
    template <FFIType F>
    inline std::shared_ptr<void> ffi_from_python_attr(PyObject* obj, std::vector<size_t>& shape) {
        using T = typename FFITypeMap::find_type<F>;
        return FFIConversionManager<T>::from_python_direct(F, obj, shape);
    }
    template <size_t... Idx>
    inline std::shared_ptr<void> ffi_from_python(const FFIType type, PyObject* obj, std::vector<size_t>& shape,
                                                      std::index_sequence<Idx...> inds) {
        return FFIConversionManager<std::tuple_element_t<Idx, FFITypePairs>...>::from_python(type, obj, shape);
    }
    inline std::shared_ptr<void> ffi_from_python(const FFIType type, PyObject* obj, std::vector<size_t>& shape) {
        return ffi_from_python(
                type, obj, shape,
                std::make_index_sequence<std::tuple_size<FFITypePairs>{}>{}
        );
    };

    template <typename T>
    inline std::shared_ptr<void> genericize(T data) {
        return FFITypeHandler<T>::genericize(data);
    }

    class FFIArgument {
        std::string param_key;
        std::vector<size_t> shape_vec; // for holding NumPy data
        FFIType type_char;
    public:
        FFIArgument(
                std::string &name,
                FFIType type,
                std::vector<size_t> &shape
        ) : param_key(name), shape_vec(shape), type_char(type) {}
        FFIArgument(
                const char* name,
                FFIType type,
                std::vector<int> shape
        ) : param_key(name), type_char(type) {
//            for (auto s : shape) { shape_vec}
            shape_vec = std::vector<size_t>(shape.begin(), shape.end());
        }
        // default trivial constructory
        FFIArgument() = default;
//        // default copy constructor
//        FFIArgument(const FFIArgument& arg)  : param_key(arg.param_key), shape_vec(arg.shape_vec), type_char(arg.type_char) {
//            printf("Copying FFIArgument %s\n", repr().c_str());
//        };
//        // trivial destructor should be safe?
//        ~FFIArgument() = default;


        std::string name() {return param_key;}
        std::vector<size_t> shape() {return shape_vec;}
        FFIType type() {return type_char;}

        PyObject * as_tuple();
        std::string repr();

    };

    size_t FFICompoundReturn::key_index(std::string& key) {
        size_t i;
//        py_printf("  > looking through ");
//        py_printf("%lu poopies\n", params.size());
        for ( i=0; i < type.keys.size(); i++) {
            auto p = type.keys[i];
//            py_printf("  > this is not my mom (%lu) ", i);
//            py_printf("%s\n", p.name().c_str());
            if (p == key) break;
        };
        if ( i == type.keys.size()) throw std::runtime_error("key \"" + key + "\" not found");
        return i;
    }

    template <typename T>
    T FFICompoundReturn::get(std::string key) {
        auto idx = key_index(key);
        return FFITypeHandler<T>::cast(type.types[idx], data[idx]);
    }
    template <typename T>
    void FFICompoundReturn::set(std::string key, T value) {
        auto idx = key_index(key);
        validate_type<T>(type.types[idx]);
        data[idx] = genericize<T>(value);
    }
    PyObject* FFICompoundReturn::as_python() {
        std::vector<PyObject*> objects;
        objects.reserve(data.size());
        for (size_t i=0; i < data.size(); i++){
            objects[i] = ffi_to_python(type.types[i], data[i], type.shapes[i]);
        }
        return mcutils::python::as_python_dict(
                type.keys,
                objects
                );
    }


    class FFIParameter {
        // object that maps onto the python FFI stuff...
        PyObject *py_obj;
        FFIArgument arg_spec;
        std::shared_ptr<void> param_data; // we void pointer this to make it easier to handle
    public:
        FFIParameter(
                PyObject *obj,
                FFIArgument& arg
                ) : py_obj(obj), arg_spec(arg), param_data() {};

        FFIParameter(
                std::shared_ptr<void>& data,
                FFIArgument& arg
        ) : py_obj(NULL), arg_spec(arg), param_data(data) {};

        template <typename T>
        FFIParameter(
                T data,
                FFIArgument& arg
        ) : py_obj(NULL), arg_spec(arg), param_data(data) {};

        explicit FFIParameter(PyObject *obj) : py_obj(obj) { init(); }
        // default trivial constructor
        FFIParameter() = default;
//        // default copy constructor
//        FFIParameter(const FFIParameter& arg) : py_obj(arg.py_obj), arg_spec(arg.arg_spec), param_data(arg.param_data) {
//                printf("Copying FFIParameter with value %s\n", repr().c_str());
//        };
//        // trivial destructor should be safe?
//        ~FFIParameter() = default;
//        // ...?

        void init();

        std::string name() { return arg_spec.name(); }
        std::vector<size_t> shape() { return arg_spec.shape(); }
        FFIType type() { return arg_spec.type(); }

        template <typename T>
        T value() {
            return FFITypeHandler<T>::cast(type(), param_data);
        }

        std::shared_ptr<void> _raw_ptr() { return param_data; } // I put this out there so people smarter than I can use it

        PyObject* as_python();
        std::string repr();
    };

    class FFIParameters {
        // object that maps onto the python FFI stuff...
        PyObject *py_obj;
        std::vector<FFIParameter> params;
    public:
        FFIParameters() : py_obj(), params() {};
        explicit FFIParameters(PyObject* param_obj) : py_obj(param_obj) {
            params = {};
            init();
        }
        void init();

        size_t param_index(std::string& key);
        FFIParameter get_parameter(std::string& key);
        FFIParameter get_parameter(const char* key);
        void set_parameter(std::string& key, FFIParameter& param);
        void set_parameter(const char* key, FFIParameter& param);

        template <typename T>
        T value(std::string& key) { return get_parameter(key).value<T>(); }
        template <typename T>
        T value(const char* key) { return get_parameter(key).value<T>(); }
        std::vector<size_t> shape(std::string& key);
        std::vector<size_t> shape(const char* key);
        FFIType typecode(std::string& key);
        FFIType typecode(const char* key);

    };



    // This used to be in FFIParameters.cpp but I'm going header only

    using mcutils::python::py_printf;

    using namespace mcutils::python;

    bool DEBUG_PRINT=false;
    bool debug_print() {
        return DEBUG_PRINT;
    }
    void set_debug_print(bool db) {
        DEBUG_PRINT=db;
        pyadeeb.set_debug_print(db); // because of bad design choices I gotta do this multiple places...
    }

//    // weirdness with constexpr... (https://stackoverflow.com/a/8016853/5720002)
//    template <typename T, typename... Args>
//    constexpr const FFIType FFIConversionManager<T, Args...>::pointer_types[];

    // defines a compiler map between FFIType and proper types
    PyObject * FFIArgument::as_tuple() {
        return Py_BuildValue("(NNN)",
                             mcutils::python::as_python<std::string>(param_key),
                             mcutils::python::as_python<FFIType>(type_char),
                             mcutils::python::as_python_tuple<size_t>(shape_vec)
        );
    }
    std::string FFIArgument::repr() {
        auto pp = as_tuple();
        auto repr = get_python_repr(pp);
        Py_XDECREF(pp);
        return repr;
    }

    void FFIParameter::init() {
        if (debug_print()) {
            auto garb = get_python_repr(py_obj);
            py_printf("Destructuring PyObject %s\n", garb.c_str());
        }
        if (debug_print()) py_printf("  > getting arg_type\n");
        auto type_char = get_python_attr<FFIType>(py_obj, "arg_type");
        if (debug_print()) py_printf("    > got %d\n", static_cast<int>(type_char));
        if (debug_print()) py_printf("  > getting arg_name\n");
        auto name = get_python_attr<std::string>(py_obj, "arg_name");
        if (debug_print()) py_printf("  > getting arg_shape\n");
        auto shape = get_python_attr_iterable<size_t>(py_obj, "arg_shape");
        if (debug_print()) py_printf("  > getting arg_val\n");
//        auto val_obj = get_python_attr<PyObject*>(py_obj, "arg_value");
//        if (debug_print()) py_printf("  converting to voidptr...\n");
        param_data = ffi_from_python_attr(type_char, py_obj, "arg_value", shape); // pulls arg_value by default...

        if (debug_print()) py_printf("  constructing FFIArgument...\n");

        arg_spec = FFIArgument(name, type_char, shape);

    }

    PyObject* FFIParameter::as_python() {
        auto shp = shape();
        return ffi_to_python(type(), param_data, shp);
    }

    std::string FFIParameter::repr() {
        auto pp = as_python();
        auto repr = get_python_repr(pp);
        Py_XDECREF(pp);
        return repr;
    }

    void FFIParameters::init() {
        params = get_python_attr_iterable<FFIParameter>(py_obj, "ffi_parameters");
    }

    size_t FFIParameters::param_index(std::string& param_name) {
        size_t i;
//        py_printf("  > looking through ");
//        py_printf("%lu poopies\n", params.size());
        for ( i=0; i < params.size(); i++) {
            auto p = params[i];
//            py_printf("  > this is not my mom (%lu) ", i);
//            py_printf("%s\n", p.name().c_str());
            if (p.name() == param_name) break;
        };
        if ( i == params.size()) throw std::runtime_error("parameter \"" + param_name + "\" not found");
        return i;
    }
    FFIParameter FFIParameters::get_parameter(std::string& param_name) {
        auto i = param_index(param_name);
        return params[i];
    }
    FFIParameter FFIParameters::get_parameter(const char* param_name) {
        std::string key = param_name;
        return get_parameter(key);
    }
    void FFIParameters::set_parameter(std::string& param_name, FFIParameter& param) {
        try {
            auto i = param_index(param_name);
            params[i] = param;
        } catch (std::exception& e) {
            params.push_back(param);
        }
    }
    void FFIParameters::set_parameter(const char *param_name, FFIParameter &param) {
        std::string key = param_name;
        set_parameter(key, param);
    }

    std::vector<size_t> FFIParameters::shape(std::string &key) {
        return get_parameter(key).shape();
    }
    std::vector<size_t> FFIParameters::shape(const char *key) {
        return get_parameter(key).shape();
    }
    FFIType FFIParameters::typecode(std::string &key) {
        return get_parameter(key).type();
    }
    FFIType FFIParameters::typecode(const char *key) {
        return get_parameter(key).type();
    }


}

// register a conversion for FFIType
namespace mcutils::python {
    template<>
    inline PyObject *as_python<plzffi::FFIParameter>(plzffi::FFIParameter data) {
        if (plzffi::debug_print()) printf("Converting FFIParameter to PyObject...\n");
        return data.as_python();
    }

    template<>
    inline plzffi::FFIParameter from_python<plzffi::FFIParameter>(PyObject *data) {
        if (plzffi::debug_print()) printf("Converting PyObject to FFIParameter...\n");
        return plzffi::FFIParameter(data);
    }
}

#endif //RYNLIB_FFIPARAMETERS_HPP
