
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

    using debug = mcutils::python::pyadeeb;
    using DebugLevel = mcutils::python::DebugLevel;
    using mcutils::python::pyobj;
    using mcutils::python::py_printf;

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

    class FFICompoundType {
        std::vector<std::string> _keys;
        std::vector<FFIType> _types;
        std::vector<std::vector<size_t> > _shapes;

    public:
        FFICompoundType() = default;
        FFICompoundType(
                std::vector<std::string> key_vec,
                std::vector<FFIType> type_vec,
                std::vector<std::vector<size_t> > shape_vec
        ) : _keys(key_vec), _types(type_vec), _shapes(shape_vec) {
            assert(_keys.size() == _types.size());
            assert(_keys.size() == _shapes.size());
        };
        FFICompoundType(
                std::vector<std::string> key_vec,
                std::vector<FFIType> type_vec
        ) : _keys(key_vec), _types(type_vec), _shapes(std::vector<std::vector<size_t>>(key_vec.size())) {
            assert(_types.size() == _types.size());
        };
        std::vector<std::string> keys() { return _keys; }
        std::vector<FFIType> types() { return _types; }
        std::vector<std::vector<size_t> > shapes() { return _shapes; }
        pyobj as_python() {
            return pyobj(Py_BuildValue(
                    "{s:N,s:N,s:N}",
                    "keys", mcutils::python::as_python_tuple_object(_keys),
                    "types", mcutils::python::as_python_tuple_object(_types),
                    "shapes", mcutils::python::as_python_tuple_object(_shapes)
            ));
        }
    };


    enum FFIContainerType {
        None = 0,
        Raw = 1,
        Vector = 2,
        Array = 3
    };
    class FFICompoundReturn {
        FFICompoundType type;
        std::vector<FFIContainerType> containers;
        std::vector<std::shared_ptr<void> > data;
        std::vector<std::vector<size_t>> res_shapes;
    public:
        FFICompoundReturn() = default;
        FFICompoundReturn(FFICompoundType comp_type, std::vector<std::shared_ptr<void> > init_data) : type(comp_type), data(init_data) {
            auto size = type.keys().size();
            containers.resize(size);
            data.resize(size);
            res_shapes.resize(size);
        };
        explicit FFICompoundReturn(FFICompoundType comp_type) : type(comp_type) {
            auto size = type.keys().size();
            containers.resize(size);
            data.resize(size);
            res_shapes.resize(size);
        }

//        ~FFICompoundReturn() // I. destructor
//        {
//            // dunno what I do with my shared ptrs...
//        }
//
//        FFICompoundReturn(const FFICompoundReturn& other) // II. copy constructor
//                : FFICompoundReturn(other.type, other.data) {}
//
//        FFICompoundReturn& operator=(const FFICompoundReturn& other) // III. copy assignment
//        {
//            if (this != &other) {
//                type = other.type;
//                data = other.data;
//            }
//
//            return *this;
//        }


        size_t key_index(std::string& key);

        template <typename T>
        class value_getter;
        template <typename T>
        value_getter<T> get(std::string key);
        template <typename T>
        value_getter<T> get_idx(size_t idx);
        template <typename T>
        void set(std::string key, T value);
        template <typename T>
        void set(std::string key, T* value);
        template <typename T>
        void set(std::string key, std::vector<T> value);
        void set_shape(std::string key, std::vector<size_t> shape);

//        template <typename T>
//        class setter{
//            FFICompoundReturn* parent;
//        public:
//            void set(size_t idx, std::string key, T value);
//            void set(size_t idx, std::string key, T* value);
//            void set(size_t idx, std::string key, std::vector<T> value);
//        };

        FFICompoundType types() {return type;}
        std::vector<FFIContainerType> container_types() {return containers;}
        std::vector<std::vector<size_t>> result_shapes() {return res_shapes;};

        pyobj as_python();
    };

}
// register a conversion for FFIType
namespace mcutils::python {
        template<>
         PyObject* as_python_object<plzffi::FFIType>(plzffi::FFIType data) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("Converting FFIType\n");
            return as_python_object<int>(static_cast<int>(data));
        }
        template<>
         plzffi::FFIType from_python<plzffi::FFIType>(pyobj data) {
            return static_cast<plzffi::FFIType>(data.getattr<int>("value"));
        }

        template<>
         pyobj as_python<plzffi::FFICompoundType>(plzffi::FFICompoundType data) {
            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("Converting FFICompoundType\n");
            return data.as_python();
        }
        template<>
         PyObject* as_python_object<plzffi::FFICompoundType>(plzffi::FFICompoundType data) {
            return as_python<plzffi::FFICompoundType>(data).obj();
        }

        template<>
         plzffi::FFICompoundType from_python<plzffi::FFICompoundType>(pyobj dict) {

            auto keys = dict.get_key<pyobj>("keys");
            auto types = dict.get_key<pyobj>("types");
            auto shapes = dict.get_key<pyobj>("shapes");

            return plzffi::FFICompoundType {
                    from_python<std::vector<std::string> >(keys),
                    from_python<std::vector<plzffi::FFIType> >(types),
                    from_python<std::vector<std::vector<size_t> > >(shapes)
            };

        }

//    template<>
//    PyObject* numpy_object_from_data<plzffi::FFICompoundReturn>(
//            plzffi::FFICompoundReturn* buffer,
//            NPY_TYPES dtype,
//            std::vector<size_t>& shape,
//            bool copy
//    ); // Implemented later

    template<>
     pyobj as_python<plzffi::FFICompoundReturn>(plzffi::FFICompoundReturn data) {
        if (pyadeeb::debug_print(DebugLevel::All)) {
            py_printf("Constructing compound return value\n");
        }
        return data.as_python();
    }
    template<>
     PyObject* as_python_object<plzffi::FFICompoundReturn>(plzffi::FFICompoundReturn data) {
        return as_python<plzffi::FFICompoundReturn>(data).obj();
    }
//        template<>
//         plzffi::FFICompoundReturn convert<plzffi::FFICompoundReturn>(pyobj data) {
//            return data.convert();
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
//         pyobj as_python_object<plzffi::FFIType>(plzffi::FFIType data) {
//            return as_python_object<int>(static_cast<int>(data));
//        }
//        template<>
//         plzffi::FFIType convert<plzffi::FFIType>(pyobj data) {
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
            FFITypePair<FFIType::PyObject, pyobj>
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
                        char msg[60] = "";
                        sprintf(msg, "can't resolve type %s to FFIType...", typeid(T).name());
                        throw std::runtime_error(msg);
                    } else {
                        return ffi_type_resolver<D, Args...>::typecode();
                    }
                }
            }
        };
        template <typename D, typename T, typename... Args>
        struct ffi_type_resolver<D, std::tuple<T, Args...>> {
            static constexpr FFIType typecode() { return ffi_type_resolver<D, T, Args...>::typecode(); }
        };


        template <typename T>
        static constexpr FFIType ffi_typecode() {
            return ffi_type_resolver<T, pairs>::typecode();
        }

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


    template <typename pairs>
    struct FFITypeValidatorBase {
        template<typename, typename...>
        struct resolver;
        template<typename D>
        struct resolver<D> {
            static void value(FFIType type) {
                std::string msg = "ERROR: unhandled typename/FFIType pair ";
                std::string tname = mcutils::type_name<D>::value;
                msg += "(" + tname + "/" + std::to_string(static_cast<int>(type)) + ")";
                printf("%s\n", msg.c_str());
                throw std::runtime_error(msg);
            }
        };
        template<typename D, typename T, typename... Args>
        struct resolver<D, T, Args...> {
            static void value(FFIType type) {
                if (std::is_same<D, typename T::type>::value) {
//                auto valids = T::valid;
//                auto nels = std::tuple_size_v<decltype(T::valid)>;
                    // https://stackoverflow.com/a/40233852/5720002
                    // Functional-programming loop over tuple elements
                    // to check if the type is in the held tuple
                    // This is basically a bunch of template fuckery
                    // just to avoid a huge switch...I think it's worth it?
                    bool found = std::apply(
                            [type](auto &&... args) {
                                return ((decltype(args)(args) == type) || ...);
                            },
                            T::valid
                    );
                    if (!found) {
                        std::string msg = "typename/FFIType mismatch: (";
                        msg += mcutils::type_name<D>::value;
                        msg += "/" + std::to_string(static_cast<int>(type)) + ")";
                        printf("%s\n", msg.c_str());
                        throw std::runtime_error(msg);
                    }
                } else {
                    resolver<D, Args...>::value(type);
                }
            };
        };
        template<typename D, typename T, typename... Args> // expects FFITypeset objects
        struct resolver<std::vector<D>, T, Args...> {
            static void value(FFIType type) {
                resolver<D, T, Args...>::value(type);
            }
        };
        template<typename D, typename T, typename... Args> // expects FFITypeset objects
        struct resolver<D *, T, Args...> {
            static void value(FFIType type) {
                resolver<D, T, Args...>::value(type);
            }
        };
        template<typename D, typename T, typename... Args> // Tuple-based spec
        struct resolver<D, std::tuple<T, Args...>> {
            static void value(FFIType type) {
                resolver<D, T, Args...>::value(type);
            }
        };

        template<typename D>
        static void validate(FFIType type) {
            return resolver<D, pairs>::value(type);
        }
    };
    using FFITypeValidator = FFITypeValidatorBase<FFITypesets>;


    template <typename T>
     void validate_type(FFIType type) {
        return FFITypeValidator::validate<T>(type);
    }
    template <typename T>
     constexpr FFIType ffi_typecode() {
        return FFITypeMap::ffi_typecode<T>();
    }

    template <typename T>
    struct FFITypeHandler {
        // class that helps us maintain a map between type codes & proper types
        static constexpr FFIType ffi_type() { return ffi_typecode<T>(); }
        static void validate(FFIType type_code) { validate_type<T>(type_code); }
        static T cast(FFIType type_code, std::shared_ptr<void> &data) {
            validate(type_code);
            return *static_cast<T *>(data.get());
        }
        static std::shared_ptr<void> genericize(T data) {
//            validate(type_code);

            auto ptr = std::make_shared<T>(data);

            if (debug::debug_print(DebugLevel::All)) {
                py_printf("Stored generic version of object of type %s at %p\n", mcutils::type_name<T>::c_str(), ptr.get());
            }

            return ptr;

        }
        static pyobj as_python(FFIType type_code, std::shared_ptr<void> &data, std::vector<size_t> &shape) {
            if (!shape.empty()) {
                return FFITypeHandler<T*>().as_python(type_code, data, shape);
            }
            return mcutils::python::as_python<T>(*static_cast<T *>(data.get()));
        }
    };

    template <typename T>
    struct FFITypeHandler<std::vector<T> > {
        // specialization to handle vector types
        static constexpr FFIType ffi_type() {return ffi_typecode<T>();}
        static void validate(FFIType type_code) { validate_type<T>(type_code); }
        static std::vector<T> cast(FFIType type_code, std::shared_ptr<void>& data) {
            return *static_cast<std::vector<T>*>(data.get());
        }
        static std::shared_ptr<void> genericize(std::vector<T> data) {
            return std::make_shared<std::vector<T>>(data);
        }
        static pyobj as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
            auto vals = *static_cast<std::vector<T>*>(data.get());
            return mcutils::python::numpy_from_data<T>(vals, shape);
        }
    };
    template <typename T>
    struct FFITypeHandler<T*> {
        // specialization to handle pointer types
        static constexpr FFIType ffi_type() {return ffi_typecode<T>();}
        static void validate(FFIType type_code) { validate_type<T>(type_code); }
        static T* cast(FFIType type_code, std::shared_ptr<void>& data) {
            return static_cast<T*>(data.get());
        }
        static std::shared_ptr<void> genericize(T* data) {
            return std::shared_ptr<void>(
                    data
                    , [](T* val){} //  // delete val;}
            );
        }
        static pyobj as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
            return mcutils::python::numpy_from_data<T>(static_cast<T*>(data.get()), shape);
        }
    };

    // recursive definition to loop through and test all the type pairs (thank you StackOverflow)
    // we'll embed most conversion functions here so that we don't need to duplicate the boiler plate
    template <typename...>
    class FFIConversionManager;
    template<>
    class FFIConversionManager<> {
    public:
        static std::shared_ptr<void> from_python(FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            std::string garb = "unhandled type specifier in converting from python: " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
        static std::shared_ptr<void> from_python_attr(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            std::string garb = "unhandled type specifier in converting from python: " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
        static pyobj as_python(FFIType type,  std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            std::string garb = "unhandled type specifier in converting to python: " + std::to_string(static_cast<int>(type));
            throw std::runtime_error(garb.c_str());
        }
    };
    template <typename T, typename... Args> // expects FFITypePair objects
    class FFIConversionManager<T, Args...> {
    public:
        static std::shared_ptr<void> from_python_direct(FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            // determine if our type is a pointer type
            auto pos = std::find(FFIPointerTypes.begin(), FFIPointerTypes.end(), type);
            if (pos < FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
                if (debug::debug_print(DebugLevel::All)) {
                    auto garb = py_obj.repr();
                    py_printf("Converting PyObject %s with pointer type FFIType %i\n", garb.c_str(), type);
                }
                // extract with a deleter if raw pointer
                switch(ctype) {
                    case(FFIContainerType::Vector):
                        return std::make_shared<std::vector<typename T::type>>(
                                mcutils::python::from_python<std::vector<typename T::type>>(py_obj)
                        );
                    default:
                        return std::shared_ptr<void>(
                                mcutils::python::from_python<typename T::type *>(py_obj)
                                , [](typename T::type *val) {}// This came from python and will be managed by python // delete val; }
                        );
                }
            } else {
                if (debug::debug_print(DebugLevel::All)) {
                    auto garb = py_obj.repr();
                    py_printf("Converting PyObject %s with non-pointer type FFIType %i\n", garb.c_str(), type);
                }
                // is a pointer so no deletion
                return std::make_shared<typename T::type>(
                        mcutils::python::from_python<typename T::type>(py_obj)
                );
            }
        }
        static std::shared_ptr<void> from_python(FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            if (type == T::value) { // the type matches up with our stored type
                return from_python_direct(type, py_obj, shape, ctype);
            } else {
                return FFIConversionManager<Args...>::from_python(type, py_obj, shape, ctype);
            }
        }

        static std::shared_ptr<void> from_python_attr_direct(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            using D = typename T::type;

            auto pos = std::find(FFIPointerTypes.begin(), FFIPointerTypes.end(), type);
            if (pos < FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
                if (debug::debug_print(DebugLevel::All)) {
                    auto garb = py_obj.repr();
                    py_printf("Converting PyObject %s attr %s with pointer type FFIType %i\n", garb.c_str(), attr, type);
                }
                switch(ctype) {
                    case(FFIContainerType::Vector):
                        return std::make_shared<std::vector<D>>(py_obj.getattr<std::vector<D>>(attr));
                    default:
                        return std::shared_ptr<void>(
                                py_obj.getattr<D*>(attr)
                                , [](D* val){}// This came from python and will be managed by python //delete val;}
                        );
                }
            } else {
                // new managed instance so no deleter
                if (debug::debug_print(DebugLevel::All)) {
                    auto garb = py_obj.repr();
                    py_printf("Converting PyObject %s attr %s with pointer type FFIType %i\n", garb.c_str(), attr, type);
                }
                return std::make_shared<D>(py_obj.getattr<D>(attr));
            }
        }
        static std::shared_ptr<void> from_python_attr(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            if (type == T::value) { // the type matches up with our stored type
                return from_python_attr_direct(type, py_obj, attr, shape);
            } else {
                return FFIConversionManager<Args...>::from_python_attr(type, py_obj, attr, shape);
            }
        }

        static pyobj as_python_direct(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            using D = typename T::type;

            // determine if our type is a pointer type
            auto pos = std::find(FFIPointerTypes.begin(), FFIPointerTypes.end(), type);
            if (pos < FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
                // is a pointer type so we convert regularly

                switch(ctype) {
                    case(FFIContainerType::Vector): {
                        if (debug::debug_print(DebugLevel::All)) {
                            py_printf("Converting type std::vector<%s> to python with FFIType %i\n", mcutils::type_name<D>::c_str(), type);
                        }
                        return FFITypeHandler<std::vector<D>>::as_python(type, data, shape);
                    }
                    default: {
                        if (debug::debug_print(DebugLevel::All)) {
                            py_printf("Converting type %s* to python with FFIType %i\n", mcutils::type_name<D>::c_str(), type);
                        }
                        return FFITypeHandler<D*>::as_python(type, data, shape);
                    }
                }
            } else {

                if (debug::debug_print(DebugLevel::All)) {
                    py_printf("Converting type %s to python with non-pointer type FFIType %i\n", mcutils::type_name<D>::c_str(), type);
                }
                // not a pointer type so we extract data from shared_ptr as a pointer first
                return FFITypeHandler<D>::as_python(type, data, shape);
            }
        }
        static pyobj as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            if (type == T::value) {
                return as_python_direct(type, data, shape, ctype);
            } else {
                return FFIConversionManager<Args...>::as_python(type, data, shape, ctype);
            }
        }
    };
    template <typename T, typename... Args> // expects FFITypePair objects
    class FFIConversionManager<std::tuple<T, Args...>> {
    public:

        static std::shared_ptr<void> from_python(FFIType type, pyobj py_obj, std::vector<size_t> &shape, FFIContainerType ctype = None) {
            return FFIConversionManager<T, Args...>::from_python(type, py_obj, shape, ctype);
        }

        static std::shared_ptr<void> from_python_attr(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            return FFIConversionManager<T, Args...>::from_python_attr(type, py_obj, attr, shape);
        }

        static pyobj as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            return FFIConversionManager<T, Args...>::as_python(type, data, shape, ctype);
        }

    };
    using FFIConverter = FFIConversionManager<FFITypePairs>;

    template <typename T>
     pyobj ffi_to_python(std::shared_ptr<void>&data, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        return FFIConversionManager<T>::as_python_direct(FFITypeHandler<T>::typecode(), data, shape, ctype);
    }
    template <FFIType F>
     pyobj ffi_to_python(std::shared_ptr<void>&data, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        using T = typename FFITypeMap::find_type<F>;
        return FFIConversionManager<T>::as_python_direct(F, data, shape, ctype);
    }
     pyobj ffi_to_python(FFIType type, std::shared_ptr<void>&data, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        return FFIConverter::as_python(type, data, shape, ctype);
    }

    template <typename T>
     std::shared_ptr<void> ffi_from_python_attr(pyobj obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        return FFIConversionManager<T>::from_python_attr_direct(FFITypeHandler<T>::typecode(), obj, attr, shape, ctype);
    }
    template <FFIType F>
     std::shared_ptr<void> ffi_from_python_attr(pyobj obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        using T = typename FFITypeMap::find_type<F>;
        return FFIConversionManager<T>::from_python_attr_direct(F, obj, attr, shape, ctype);
    }
     std::shared_ptr<void> ffi_from_python_attr(const FFIType type, pyobj obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        return FFIConverter::from_python_attr(type, obj, attr, shape, ctype);
    }

    template <typename T>
     std::shared_ptr<void> ffi_from_python(pyobj obj, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        return FFIConversionManager<T>::from_python_direct(FFITypeHandler<T>::typecode(), obj, shape, ctype);
    }
    template <FFIType F>
     std::shared_ptr<void> ffi_from_python_attr(pyobj obj, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        using T = typename FFITypeMap::find_type<F>;
        return FFIConversionManager<T>::from_python_direct(F, obj, shape, ctype);
    }
     std::shared_ptr<void> ffi_from_python(const FFIType type, pyobj obj, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
        return FFIConverter::from_python(type, obj, shape, ctype);
    }

    template <typename T>
     std::shared_ptr<void> genericize(T data) {
        auto ptr = FFITypeHandler<T>::genericize(data);
        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Got generic version of object of type %s at %p\n", mcutils::type_name<T>::c_str(), ptr.get());
        }
        return ptr;

    }

    /*
     * FFICompoundReturn defs
     *
     *
     */

    template <typename pairs>
    struct FFICompoundReturnVectorCollator {
        template <typename T>
        class compound_return_vector_extractor {
            FFICompoundReturn* buffer;
            size_t nels;
            size_t idx;
        public:
            compound_return_vector_extractor(FFICompoundReturn* b, size_t n, size_t i) : buffer(b), nels(n), idx(i) {}
            std::vector<T> extract_plain() {
                std::vector<T> res(nels);
                for (size_t i = 0; i < nels; i++) {
                    auto r = buffer[i];
                    res[i] = r.template get_idx<T>(idx).extract_plain();
                }
                return res;
            }
            std::vector<std::vector<T>> extract_vector() {
                std::vector<std::vector<T>> res(nels);
                for (size_t i = 0; i < nels; i++) {
                    auto r = buffer[i];
                    res[i] = r.template get_idx<T>(idx).extract_vector();
                }
                return res;
            }
            explicit operator std::vector<T>() {
                return extract_plain();
            }
            explicit operator std::vector<std::vector<T>>() {
                return extract_vector();
            }
        };

        template<typename...>
        struct resolver;
        template<>
        struct resolver<> {
//            static void collate(
//                    FFIType type,
//                    FFICompoundReturn* buffer, size_t nels, size_t idx
//            ) {
//                std::string msg = "ERROR: in collator dispatch: unresolved FFIType " + std::to_string(static_cast<int>(type));
//                printf("%s\n", msg.c_str());
//                throw std::runtime_error(msg);
//            }
            static pyobj extract_ffi(
                    FFIType type, FFIContainerType ctype, std::vector<size_t>& shape,
                    FFICompoundReturn* buffer, size_t nels, size_t idx
            ) {
                std::string msg = "ERROR: in collator dispatch: unresolved FFIType " + std::to_string(static_cast<int>(type));
                if (debug::debug_print(DebugLevel::Normal))py_printf("%s\n", msg.c_str());
                throw std::runtime_error(msg);
            }
        };
        template<typename t, typename... subargs>
        struct resolver<t, subargs...> {
//            static compound_return_vector_extractor<typename t::type> collate(
//                    FFIType type,
//                    FFICompoundReturn* buffer, size_t nels, size_t idx
//            ) {
//                if (type == t::value) {
//                    using D = typename t::type;
//                    return compound_return_vector_extractor<D> (
//                            buffer,
//                            nels,
//                            idx
//                    );
//                } else {
//                    return resolver<subargs...>::collate(
//                            type,
//                            buffer, nels, idx
//                    );
//                }
//            };
            static pyobj extract_ffi(
                    FFIType type, FFIContainerType ctype, std::vector<size_t>& shape,
                    FFICompoundReturn* buffer, size_t nels, size_t idx
            ) {
                if (type == t::value) {
                    using D = typename t::type;
                    auto collator = compound_return_vector_extractor<D>(
                            buffer,
                            nels,
                            idx
                    );
//                    std::vector<size_t> new_shape(1+shape.size());
//                    new_shape[0] = nels;
//                    for (size_t i = 0; i < shape.size(); i++) {
//                        new_shape[1+i] = shape[i];
//                    }

//                    py_printf( "          - old shape: ( ");
//                    for (auto s: shape) py_printf( "%lu ", s);
//                    py_printf(")\n");
//                    py_printf( "          - new shape: ( ");
//                    for (auto s: new_shape) py_printf( "%lu ", s);
//                    py_printf(")\n");


                    switch(ctype) {
                        case FFIContainerType::Vector: {
                            auto val = collator.extract_vector();
                            return mcutils::python::numpy_from_data<D>(val, shape);
                        };
                        default: {
                            auto val = collator.extract_plain();
                            return mcutils::python::numpy_from_data<D>(val, shape);
                        }
                    }

                } else {
                    return resolver<subargs...>::extract_ffi(
                            type, ctype, shape,
                            buffer, nels, idx
                    );
                }
            }
        };
        template<typename t, typename... subargs> // Tuple-based spec
        struct resolver<std::tuple<t, subargs...>> {
//            static compound_return_vector_extractor<typename t::type> collate(
//                    FFIType type,
//                    FFICompoundReturn* buffer, size_t nels, size_t idx
//            ) {
//                return resolver<t, subargs...>::call(
//                        type,
//                        buffer, nels, idx
//                );
//            }
            static pyobj extract_ffi(
                    FFIType type, FFIContainerType ctype, std::vector<size_t>& shape,
                    FFICompoundReturn* buffer, size_t nels, size_t idx
                    ) {

                if (debug::debug_print(DebugLevel::All)) py_printf("     --> collating compound return values\n");
                return resolver<t, subargs...>::extract_ffi(
                        type, ctype, shape,
                        buffer, nels, idx
                );
            }
        };
        using collator_resolver = resolver<pairs>;

        static pyobj collate_to_python(
                FFIType type, FFIContainerType ctype, std::vector<size_t>& shape,
                FFICompoundReturn* buffer, size_t nels, size_t idx
                ) {
            return collator_resolver::extract_ffi(type, ctype, shape, buffer, nels, idx);
        }
    };

    using FFICompoundReturnCollator = FFICompoundReturnVectorCollator<FFITypePairs>;

    size_t FFICompoundReturn::key_index(std::string& key) {
        size_t i;
//        py_printf("  > looking through ");
//        py_printf("%lu poopies\n", params.size());
        auto keys = type.keys();
        for ( i=0; i < keys.size(); i++) {
            auto p = keys[i];
//            py_printf("  > this is not my mom (%lu) ", i);
//            py_printf("%s\n", p.name().c_str());
            if (p == key) break;
        };
        if ( i == keys.size()) throw std::runtime_error("key \"" + key + "\" not found");
        return i;
    }

    template <typename T>
    class FFICompoundReturn::value_getter{
        std::shared_ptr<void> data;
        FFIType type;
        FFIContainerType ctype;
    public:
        value_getter(
                std::shared_ptr<void> dat,
                FFIType typ,
                FFIContainerType ctyp
        ) : data(std::move(dat)), type(typ), ctype(ctyp) {}
        T extract_plain() {
            return FFITypeHandler<T>::cast(type, data);
        }
        std::vector<T> extract_vector() {
            return FFITypeHandler<std::vector<T>>::cast(type, data);
        }

        explicit operator T() {
            if (ctype == FFIContainerType::Vector) throw std::runtime_error("requested plain type but container type is Vector");
            return extract_plain();
        }
        explicit operator std::vector<T>() {
            if (ctype != FFIContainerType::Vector) throw std::runtime_error("requested std::vector but container type is not Vector");
            return extract_vector();
        }
    };
    template <typename T>
    FFICompoundReturn::value_getter<T> FFICompoundReturn::get_idx(size_t idx) {
        return value_getter<T>(data[idx], type.types()[idx], container_types()[idx]);
    };
    template <typename T>
    FFICompoundReturn::value_getter<T> FFICompoundReturn::get(std::string key) {
        auto idx = key_index(key);
        return value_getter<T>(data[idx], type.types()[idx], container_types()[idx]);
    };
    template <typename T>
    void FFICompoundReturn::set(std::string key, T value) {
        auto idx = key_index(key);
        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Setting key %s at index %d\n", key.c_str(), idx);
        }
        validate_type<T>(type.types()[idx]);
        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Type pair validated...\n", key.c_str(), idx);
        }
        containers[idx] = FFIContainerType::None;
        data[idx] = genericize<T>(value);
    }
    template <typename T>
    void FFICompoundReturn::set(std::string key, T* value) {
        auto idx = key_index(key);
        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Setting key %s at index %d\n", key.c_str(), idx);
        }
        validate_type<T>(type.types()[idx]);
        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Type pair validated...\n", key.c_str(), idx);
        }
        containers[idx] = FFIContainerType::Raw;
        data[idx] = genericize<T*>(value);
    }
    template <typename T>
    void FFICompoundReturn::set(std::string key, std::vector<T> value) {
        auto idx = key_index(key);
        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Setting key %s at index %d\n", key.c_str(), idx);
        }
        validate_type<T>(type.types()[idx]);
        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Type pair validated...\n", key.c_str(), idx);
        }
        containers[idx] = FFIContainerType::Vector;
        data[idx] = genericize<std::vector<T>>(value);
    }
    void FFICompoundReturn::set_shape(std::string key, std::vector<size_t> shape) {
        auto idx = key_index(key);
        res_shapes[idx].assign(shape.begin(), shape.end()); // copy the data in
    }

    pyobj FFICompoundReturn::as_python() {
//        return mcutils::python::as_python_object(1);
//        return mcutils::python::as_python_dict(
//                {},
//                {}
//        );

        auto keys = type.keys();
        auto types = type.types();
        auto shapes = type.shapes();
        auto rs = res_shapes;

        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Building compound return type with keys: ");
            for (auto k:keys) {
                py_printf("%s ", k.c_str());
            }
            py_printf("\n");
        }

        if (data.size() != keys.size()) {
            std::string msg = "Mismatch between number of keys and number of stored values ";
            msg += "(" + std::to_string(keys.size()) + " and " + std::to_string(data.size()) + ")";
            throw std::runtime_error(msg.c_str());
        }
        if (containers.size() != keys.size()) {
            std::string msg = "Mismatch between number of keys and number of container types ";
            msg += "(" + std::to_string(keys.size()) + " and " + std::to_string(containers.size()) + ")";
            throw std::runtime_error(msg.c_str());
        }

        std::vector<pyobj> objects;
        objects.resize(data.size());
        for (size_t i=0; i < data.size(); i++){
            if (debug::debug_print(DebugLevel::All)) {
                py_printf("casting value to python for key %s\n", keys[i].c_str());
            }
            objects[i] = ffi_to_python(types[i], data[i], rs[i].empty()?shapes[i]:rs[i], containers[i]);

            if (debug::debug_print(DebugLevel::All)) {
                py_printf("got cast value %s\n", objects[i].repr().c_str());
            }
        }

        return mcutils::python::as_python_dict(keys, objects);
    }

    /*
     * FFIArgument defs
     *
     */

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

        pyobj as_tuple();
        std::string repr();

    };


    /*
     * FFIParameter defs
     *
     */

    class FFIParameter {
        // object that maps onto the python FFI stuff...
        pyobj py_obj;
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

        explicit FFIParameter(pyobj obj) : py_obj(obj) { init(); }
        // default trivial constructor
        FFIParameter() = default;
//        // default copy constructor
//        FFIParameter(const FFIParameter& arg) : py_obj(arg.py_obj), arg_spec(arg.arg_spec), param_data(arg.param_data) {
//                printf("Copying FFIParameter with value %s\n", repr().c_str());
//        };
//        // trivial destructor should be safe?
//        ~FFIParameter() = default;

        void init();

        std::string name() { return arg_spec.name(); }
        std::vector<size_t> shape() { return arg_spec.shape(); }
        FFIType type() { return arg_spec.type(); }

        template <typename T>
        T value() {
            return FFITypeHandler<T>::cast(type(), param_data);
        }

        std::shared_ptr<void> _raw_ptr() { return param_data; } // I put this out there so people smarter than I can use it

        PyObject* as_python_object();
        pyobj as_python();
        std::string repr();
    };

    class FFIParameters {
        // object that maps onto the python FFI stuff...
        pyobj py_obj;
        std::vector<FFIParameter> params;
    public:
        FFIParameters() : py_obj(), params() {};
        explicit FFIParameters(pyobj param_obj) : py_obj(param_obj) {
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
    
    using namespace mcutils::python;

//    // weirdness with constexpr... (https://stackoverflow.com/a/8016853/5720002)
//    template <typename T, typename... Args>
//    constexpr const FFIType FFIConversionManager<T, Args...>::pointer_types[];

    // defines a compiler map between FFIType and proper types
    pyobj FFIArgument::as_tuple() {
        return pyobj(Py_BuildValue("(NNN)",
                             mcutils::python::as_python_object<std::string>(param_key),
                             mcutils::python::as_python_object<FFIType>(type_char),
                             mcutils::python::as_python_tuple_object<size_t>(shape_vec)
        ));
    }
    std::string FFIArgument::repr() {
        return as_tuple().repr();
    }

    void FFIParameter::init() {
        if (debug::debug_print(DebugLevel::All)) {
            py_printf("Destructuring PyObject %s\n", py_obj.repr().c_str());
        }
        if (debug::debug_print(DebugLevel::All)) py_printf("  > getting arg_type\n");
        auto type_char = py_obj.getattr<FFIType>("arg_type");
        if (debug::debug_print(DebugLevel::All)) py_printf("    > got %d\n", static_cast<int>(type_char));
        if (debug::debug_print(DebugLevel::All)) py_printf("  > getting arg_name\n");
        auto name = py_obj.getattr<std::string>("arg_name");
        if (debug::debug_print(DebugLevel::All)) py_printf("  > getting arg_shape\n");
        auto shape = py_obj.getattr<std::vector<size_t>>("arg_shape");
        if (debug::debug_print(DebugLevel::All)) py_printf("  > getting arg_val\n");
        param_data = ffi_from_python_attr(type_char, py_obj, "arg_value", shape); // pulls arg_value by default...

        if (debug::debug_print(DebugLevel::All)) py_printf("  constructing FFIArgument...\n");

        arg_spec = FFIArgument(name, type_char, shape);

    }

    pyobj FFIParameter::as_python() {
        auto shp = shape();
        return ffi_to_python(type(), param_data, shp);
    }
    PyObject* FFIParameter::as_python_object() {
        return as_python().obj();
    }

    std::string FFIParameter::repr() { return as_python().repr(); }

    void FFIParameters::init() {
        if (pyadeeb::debug_print(DebugLevel::All)) py_printf("initializing parameters object from %s\n", py_obj.repr().c_str());
        params = py_obj.getattr<std::vector<FFIParameter>>("ffi_parameters");
    }

    size_t FFIParameters::param_index(std::string& key) {
        size_t i;
        for ( i=0; i < params.size(); i++) {
            auto p = params[i];
            if (p.name() == key) break;
        };
        if ( i == params.size()) throw std::runtime_error("parameter \"" + key + "\" not found");
        return i;
    }
    FFIParameter FFIParameters::get_parameter(std::string& key) {
        auto i = param_index(key);
        return params[i];
    }
    FFIParameter FFIParameters::get_parameter(const char* key) {
        std::string k = key;
        return get_parameter(k);
    }
    void FFIParameters::set_parameter(std::string& key, FFIParameter& param) {
        try {
            auto i = param_index(key);
            params[i] = param;
        } catch (std::exception& e) {
            params.push_back(param);
        }
    }
    void FFIParameters::set_parameter(const char *key, FFIParameter &param) {
        std::string k = key;
        set_parameter(k, param);
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

// register a conversion for FFIParamter
namespace mcutils::python {

    template<>
    PyObject* numpy_object_from_data<plzffi::FFICompoundReturn>(
            plzffi::FFICompoundReturn* buffer,
            std::vector<size_t>& shape,
            bool copy
    ) {

        if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting FFICompoundReturn* to dict of numpy arrays\n");
        auto rep = buffer[0];

        auto types = rep.types().types();
        auto containers = rep.container_types();
        auto shapes = rep.types().shapes();
        auto rs = rep.result_shapes();

        size_t nels = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

        auto keys = rep.types().keys();
        std::vector<pyobj> objs(keys.size());
        for (size_t i=0; i < keys.size(); i++) {

            if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting key %s\n", keys[i].c_str());

            // we need to concatenate the base data
            // shape with the stored return shape
            auto shp = rs[i].empty()?shapes[i]:rs[i];
            std::vector<size_t> new_shape(shape.size() + shp.size());
            std::copy(shape.begin(), shape.end(), new_shape.begin());
            std::copy(shp.begin(), shp.end(), new_shape.begin() + shape.size());

//            py_printf( "          - concat shape: ( ");
//            for (auto s: new_shape) py_printf( "%lu ", s);
//            py_printf(")\n");
//            py_printf( "          - passed shape: ( ");
//            for (auto s: shape) py_printf( "%lu ", s);
//            py_printf(")\n");

            objs[i] = plzffi::FFICompoundReturnCollator::collate_to_python(
                    types[i], containers[i], new_shape,
                    buffer, nels, i
                    );

        }
        return as_python_dict_object(
                keys,
                objs
                );
    }

    template<>
     PyObject *as_python_object<plzffi::FFIParameter>(plzffi::FFIParameter data) {
        if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> Converting FFIParameter to PyObject...\n");
        return data.as_python_object();
    }

    template<>
     plzffi::FFIParameter from_python<plzffi::FFIParameter>(pyobj data) {
        if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> Converting PyObject to FFIParameter...\n");
        return plzffi::FFIParameter(data);
    }
}

#endif //RYNLIB_FFIPARAMETERS_HPP
