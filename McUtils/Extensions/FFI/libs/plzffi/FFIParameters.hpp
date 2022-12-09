
#ifndef RYNLIB_FFIPARAMETERS_HPP
#define RYNLIB_FFIPARAMETERS_HPP

#include "PyAllUp.hpp"
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <type_traits>
#include <algorithm>
#include <unordered_map>
#include <unordered_set> 


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

    class ffiobj {
        // a little helper class for passing around shared_ptrs with associated conversion info
        // to provide an abstraction on the other versions of this idea I have laying around
        std::shared_ptr<void> data;
        FFIType _type;
        std::vector<size_t> _shape;
        FFIContainerType _ctype;
        pyobj base;
    public:
        ffiobj(std::shared_ptr<void>& dat, FFIType ffi_type, std::vector<size_t> shp, FFIContainerType container_type):
            data(dat), _type(ffi_type), _shape(shp), _ctype(container_type) {};
        ffiobj(std::shared_ptr<void>& dat, FFIType ffi_type, std::vector<size_t> shp):
            // we default to assuming a None container type
            data(dat), _type(ffi_type), _shape(shp), _ctype(FFIContainerType::None) {};
        ffiobj(std::shared_ptr<void>& dat, FFIType ffi_type):
            // for like ints and things to be passed as parameters
            data(dat), _type(ffi_type), _shape({}), _ctype(FFIContainerType::None) {};
        ffiobj(FFIType ffi_type, std::vector<size_t> shp) :
            // for like ints and things to be passed as parameters
            _type(ffi_type), _shape(shp), _ctype(FFIContainerType::None) {};
        ffiobj() = default;
        
        void set_base(pyobj& obj) { base=obj; }

        template <typename T>
        T convert();

        template <typename T>
        void assign(T value);

        static ffiobj cast(pyobj py_obj, FFIType type, std::vector<size_t>& shape, FFIContainerType ctype=None);
        template <typename T>
        static ffiobj cast(T value, std::vector<size_t>& shape, FFIContainerType ctype=None);
        template <typename T>
        static ffiobj cast(T value);

        // static ffiobj cast(pyobj py_obj, FFIType type, std::vector<size_t>& shape, FFIContainerType ctype=None) {
        //     auto dat = FFIConverter::from_python(type, py_obj, shape, ctype);
        //     ffiobj obj(dat, type, shape, ctype);
        //     obj.base = py_obj;
        // }

        std::shared_ptr<void> ptr() {return data;}
        FFIType type()  {return _type;}
        std::vector<size_t> shape()  {return _shape;}
        FFIContainerType container_type()  {return _ctype;}

    };

    class FFICompoundReturn {
        FFICompoundType type;
        std::vector<ffiobj> objects;
        // std::vector<FFIContainerType> containers;
        // std::vector<std::shared_ptr<void> > data;
        // std::vector<std::vector<size_t>> res_shapes;
    public:
        FFICompoundReturn() = default;
        // FFICompoundReturn(FFICompoundType comp_type, std::vector<std::shared_ptr<void> > init_data) : type(comp_type), data(init_data) {
        //     auto size = type.keys().size();
        //     containers.resize(size);
        //     data.resize(size);
        //     res_shapes.resize(size);
        // };
        explicit FFICompoundReturn(FFICompoundType comp_type) : type(comp_type) {
            auto size = type.keys().size();
            objects.resize(size);
            // containers.resize(size);
            // data.resize(size);
            // res_shapes.resize(size);
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

        ffiobj get(std::string key);
        ffiobj get_idx(size_t idx);
        template <typename T>
        void set(std::string key, T value, std::vector<size_t> shape = {});
        template <typename T>
        void set(std::string key, T* value, std::vector<size_t> shape = {});
        template <typename T>
        void set(std::string key, std::vector<T> value, std::vector<size_t> shape = {});
        // void set_shape(std::string key, std::vector<size_t> shape);

//        template <typename T>
//        class setter{
//            FFICompoundReturn* parent;
//        public:
//            void set(size_t idx, std::string key, T value);
//            void set(size_t idx, std::string key, T* value);
//            void set(size_t idx, std::string key, std::vector<T> value);
//        };

        FFICompoundType types() {return type;}
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

//    template <typename T>
//    struct get_ffi {
//        static const FFIType value = FFIType::GENERIC;
//    };
    template <FFIType F, typename T>
    struct FFITypePair {
        using type=T;
        static FFITypePair get_pair (std::integral_constant<FFIType, F>){ return {}; };
        static const FFIType value = F;
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

    template <typename...pairs>
    struct FFITypeMapper;
    template <typename...types>
    struct FFITypeTupleIndexer {
        static std::unordered_map<FFIType, size_t> ffi_map;

        template<typename...>
        struct ffi_type_mapper;
        template<>
        struct ffi_type_mapper<> {
            static size_t find(size_t cur, FFIType t) {
                return cur;
            }
        };
        template<typename type, typename... rest>
        struct ffi_type_mapper<type, rest...> {
            static size_t find(size_t cur, FFIType t) {
                ffi_map[t] = cur;
                if (t == type::value) {
                    return cur;
                } else {
                    return ffi_type_mapper<rest...>::find(cur + 1, t);
                };
            }
            static size_t find(FFIType t) {
                if (ffi_map.empty()) ffi_map.reserve(sizeof...(rest));
                auto pos = ffi_map.find(t);
                if (pos == ffi_map.end()) {
                    return find(0, t);
                } else {
                    return pos->second;
                }
            }
        };
        using populator = ffi_type_mapper<types...>;

        static size_t ffi_type_index(FFIType type) {
            return populator::find(type);
        }

        friend struct FFITypeMapper<types...>;
    };

//    template <typename pairs>
//    struct TypeTupleIndexer {
//
//        template<typename...>
//        struct _map_populator;
//
//        template<>
//        struct _map_populator<> {
//            static void populate(size_t cur, std::unordered_map<FFIType, size_t>) {}
//        };
//
//        template<typename type, typename... rest>
//        struct _map_populator<type, rest...> {
//            static void populate(size_t cur, std::unordered_map<FFIType, size_t> map) {
//                map[type::value] = cur;
//                _map_populator<rest...>::populate(cur + 1, map);
//            }
//
//            static void populate(std::unordered_map<FFIType, size_t> map) {
//                map.reserve(sizeof...(rest) + 1);
//                populate(0, map);
//            }
//        };
//
//        template<typename... types> // Tuple-based spec
//        struct _map_populator<std::tuple<types...>> : _map_populator<types...> {
//        };
//        using populator = _map_populator<pairs>;
//
//        static void populate() { populator::populate(ffi_map); }
//
//        size_t ffi_type_index(FFIType type) {
//            if (ffi_map.empty()) populate();
//            return ffi_map[type];
//        }
//
//        friend struct FFITypeMapper<pairs>;
//    };

    // thank you again StackOverflow
    template <typename...pairs>
    struct FFITypeMapper : public pairs... {
        using indexer = FFITypeTupleIndexer<pairs...>;
        static size_t ffi_type_index(FFIType type) {
            return indexer::ffi_type_index(type);
        };

        template<typename, typename...>
        struct type_index_resolver;
        template<typename D, typename T, typename... Args>
        struct type_index_resolver<D, T, Args...> {
            static constexpr size_t type_index(size_t cur) {
                if (std::is_same_v<D, typename T::type>) {
                    return cur;
                } else {
//                    static_assert(sizeof...(Args) > 0, "invalid type resolution");
                    if constexpr (sizeof...(Args) == 0) {
                        char msg[60] = "";
                        sprintf(msg, "can't find type tuple index for type %s...", typeid(T).name());
                        throw std::runtime_error(msg);
                    } else {
                        return type_index_resolver<D, Args...>::type_index(cur+1);
                    }
                }
            }
        };
        template<typename D, typename T, typename... Args>
        struct type_index_resolver<D*, T, Args...> : type_index_resolver<D, T, Args...> {};
        template<typename D, typename T, typename... Args>
        struct type_index_resolver<std::vector<D>, T, Args...> : type_index_resolver<D, T, Args...> {};
        template <typename T>
        static constexpr size_t type_index() {
            return type_index_resolver<T, pairs...>::type_index(0);
        }

        // duplication to avoid having to pack things back into a tuple here...
        template<typename, typename...>
        struct ffi_type_resolver;
        template<typename D, typename T, typename... Args>
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
        template<typename D, typename T, typename... Args>
        struct ffi_type_resolver<D*, T, Args...> : ffi_type_resolver<D, T, Args...> {};
        template<typename D, typename T, typename... Args>
        struct ffi_type_resolver<std::vector<D>, T, Args...> : ffi_type_resolver<D, T, Args...> {};
        template<typename T>
        static constexpr FFIType ffi_typecode() {
            return ffi_type_resolver<T, pairs...>::typecode();
        }

        using pairs::get_pair...;
        template<FFIType F>
        using find_type = typename decltype(get_pair(std::integral_constant<FFIType, F>{}))::type;

    };
    template <typename...types>
    struct FFITypeMapper<std::tuple<types...>> : FFITypeMapper<types...> {};
    struct FFITypeMap : FFITypeMapper<FFITypePairs> {
        using pairs = FFITypePairs;

        template <typename T>
        static std::string type_name() {
            return mcutils::type_name<T>::value;
        };
        template <FFIType F>
        static pyobj type_name() {
            using T = typename FFITypeMap::find_type<F>;
            return type_name<T>();
        }
        template <size_t N = 0> // Caller that will step through the pairs till we get where we need to be
        static auto type_name_caller(size_t idx, FFIType type) {
            if (idx >= std::tuple_size_v<pairs>) {
                std::string msg = "FFIType index error in from_python call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
            if (N == idx) {
                using D = typename std::tuple_element_t<N, pairs>::type;
                return type_name<D>();
            }
            if constexpr (N + 1 < std::tuple_size_v<pairs>) {
                return type_name_caller<N + 1>(idx, type);
            } else {
                std::string msg = "Unreachable: FFIType index error in type_name call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
        }
        static auto type_name(FFIType type) {
            return type_name_caller<0>(ffi_type_index(type), type);
        }

    }; // hopefully this helps with the huge error messages...
    template<> std::unordered_map<FFIType, size_t> FFITypeMap::indexer::ffi_map = {};

    struct FFITypeValidator;
    template <typename typesets>
    struct TypeTupleIndexer {
        template<typename, typename...>
        struct type_index_resolver;
        template<typename D, typename T, typename... Args>
        struct type_index_resolver<D, T, Args...> {
            static constexpr size_t type_index(size_t cur) {
                if (std::is_same_v<D, typename T::type>) {
                    return cur;
                } else {
//                    static_assert(sizeof...(Args) > 0, "invalid type resolution");
                    if constexpr (sizeof...(Args) == 0) {
                        return std::tuple_size<typesets>{}; // exhausted everything so return bad index
                    } else {
                        return type_index_resolver<D, Args...>::type_index(cur+1);
                    }
                }
            }
        };
        template<typename D, typename...tests>
        struct type_index_resolver<D, std::tuple<tests...>>: type_index_resolver<D, tests...> {};
        template <typename T>
        static constexpr size_t type_index() {
            return type_index_resolver<T, typesets>::type_index(0);
        }

        friend struct FFITypeValidator;
    };

    std::unordered_set<FFIType> FFIPointerTypes {
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

    struct FFITypeValidator {
        using typesets = FFITypesets;
        using indexer = TypeTupleIndexer<typesets>;
        template<typename T>
        struct resolver {
            static void value(FFIType type) {
                constexpr size_t idx = indexer::template type_index<T>(); // This will throw an error if not valid
                if constexpr (idx < std::tuple_size<typesets>{}) {
                    bool found = std::apply(
                            [type](auto &&... args) {
                                return ((decltype(args)(args) == type) || ...);
                            },
                            std::tuple_element_t<idx, typesets>::valid // the valid FFITypes for T
                    );
                    if (!found) {
                        std::string msg = "typename/FFIType mismatch: (";
                        msg += mcutils::type_name<T>::value;
                        msg += "/" + std::to_string(static_cast<int>(type)) + ")";
                        if (debug::debug_print(DebugLevel::Normal)) py_printf("%s\n", msg.c_str());
                        throw std::runtime_error(msg);
                    }
                } else {
                        std::string msg = "can't find type tuple index for type %s..." + mcutils::type_name<T>::value;
                        throw std::runtime_error(msg);
                }
            }
        };
        template<typename D>
        struct resolver<std::vector<D>> : resolver<D> {};
        template<typename D>
        struct resolver<D*> : resolver<D> {};

        template<typename D>
        static void validate(FFIType type) {
            return resolver<D>::value(type);
        }
    };

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

            if (debug::debug_print(DebugLevel::Excessive)) {
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
            
            auto ptr = std::make_shared<std::vector<T>>(data);
            if (debug::debug_print(DebugLevel::Excessive)) {
                py_printf("Stored generic version of object of type std::vector<%s> at %p\n", mcutils::type_name<T>::c_str(), ptr.get());
            }
            
            return ptr;
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
            auto ptr = std::shared_ptr<void> (
                    data
                    , [](T* val){} //  // delete val;}
            );
            if (debug::debug_print(DebugLevel::Excessive)) {
                py_printf("Stored shared_ptr of (%s*)%p at %p\n", mcutils::type_name<T>::c_str(), data, ptr.get());
            }
            return ptr;
        }
        static pyobj as_python(FFIType type_code, std::shared_ptr<void>& data, std::vector<size_t>& shape) {
            return mcutils::python::numpy_from_data<T>(static_cast<T*>(data.get()), shape);
        }
    };

    struct FFIConverter {
        using map = FFITypeMap;
        using pairs = FFITypePairs;

        template <typename T>
        static std::shared_ptr<void> from_python(FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            // determine if our type is a pointer type
            auto pos = FFIPointerTypes.find(type);
            if (pos != FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
                if (debug::debug_print(DebugLevel::All)) {
                    auto garb = py_obj.repr();
                    py_printf("Converting PyObject %s with pointer type FFIType %i\n", garb.c_str(), type);
                }
                // extract with a deleter if raw pointer
                switch(ctype) {
                    case(FFIContainerType::Vector):
                        return std::make_shared<std::vector<T>>(
                                py_obj.convert<std::vector<T>>()
                        );
                    default:
                        return std::shared_ptr<void>(
                                py_obj.convert<T*>()
                                , [](T*val) {}// This came from python and will be managed by python // delete val; }
                        );
                }
            } else {
                if (debug::debug_print(DebugLevel::All)) {
                    auto garb = py_obj.repr();
                    py_printf("Converting PyObject %s with non-pointer type FFIType %i\n", garb.c_str(), type);
                }
                // is a pointer so no deletion
                return std::make_shared<T>(
                    py_obj.convert<T>()
                );
            }
        }
        template <FFIType F>
        static pyobj from_python(pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
            using T = typename FFITypeMap::find_type<F>;
            return from_python<T>(F, py_obj, shape, ctype);
        }
        template <size_t N = 0> // Caller that will step through the pairs till we get where we need to be
        static auto from_python_caller(size_t idx, FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            if (idx >= std::tuple_size_v<pairs>) {
                std::string msg = "FFIType index error in from_python call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
            if (N == idx) {
                using D = typename std::tuple_element_t<N, pairs>::type;
                return from_python<D>(type, py_obj, shape, ctype);
            }
            if constexpr (N + 1 < std::tuple_size_v<pairs>) {
                return from_python_caller<N + 1>(idx, type, py_obj, shape, ctype);
            } else {
                std::string msg = "Unreachable: FFIType index error in from_python call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
        }
        static auto from_python(FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            return from_python_caller<0>(map::ffi_type_index(type), type, py_obj, shape, ctype);
        }


        template <typename D>
        static std::shared_ptr<void> from_python_attr(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            auto pos = FFIPointerTypes.find(type);
            if (pos != FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
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
        template <FFIType F>
        static pyobj from_python_attr(pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            using T = typename FFITypeMap::find_type<F>;
            return from_python_attr<T>(F, py_obj, attr, shape, ctype);
        }
        template <size_t N = 0> // Caller that will step through the pairs till we get where we need to be
        static auto from_python_attr_caller(size_t idx, FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            if (idx >= std::tuple_size_v<pairs>) {
                std::string msg = "FFIType index error in from_python_attr call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
            if (N == idx) {
                using D = typename std::tuple_element_t<N, pairs>::type;
                return from_python_attr<D>(type, py_obj, attr, shape, ctype);
            } 
            if constexpr (N + 1 < std::tuple_size_v<pairs>) {
                return from_python_attr_caller<N + 1>(idx, type, py_obj, attr, shape, ctype);
            } else {
                std::string msg = "Unreachable: FFIType index error in from_python_attr call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
        }
        static auto from_python_attr(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            return from_python_attr_caller<0>(map::ffi_type_index(type), type, py_obj, attr, shape, ctype);
        }


        template <typename D>
        static pyobj as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {

            // determine if our type is a pointer type
            auto pos = FFIPointerTypes.find(type);
            if (pos != FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
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
        template <FFIType F>
        static pyobj as_python(std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            using T = typename FFITypeMap::find_type<F>;
            return as_python<T>(F, data, shape, ctype);
        }
        template <size_t N = 0> // Caller that will step through the pairs till we get where we need to be
        static auto as_python_caller(size_t idx, FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            if (idx >= std::tuple_size_v<pairs>) {
                std::string msg = "FFIType index error in as_python call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
            if (N == idx) {
                using D = typename std::tuple_element_t<N, pairs>::type;
                return as_python<D>(type, data, shape, ctype);
            } 
            if constexpr (N + 1 < std::tuple_size_v<pairs>) {
                return as_python_caller<N + 1>(idx, type, data, shape, ctype);
            } else {
                std::string msg = "Unreachable: FFIType index error in as_python call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
        }
        static auto as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
            return as_python_caller<0>(map::ffi_type_index(type), type, data, shape, ctype);
        }
    };

    /*
     * ffiobj implementation
     *
     */

    template <typename T>
    T ffiobj::convert() {
        if (_type == FFIType::GENERIC) {
            _type = FFITypeMap::ffi_typecode<T>();
        }
        return FFITypeHandler<T>::cast(_type, data);
    }
    template <>
    pyobj ffiobj::convert<pyobj>() {
        if (base.valid()) {
            return base;
        } else {
            return FFIConverter::as_python(_type, data, _shape, _ctype);
        }
    }

    template <typename T>
    void ffiobj::assign(T value) { // for delayed initialization
        if (_type == FFIType::GENERIC) {
            _type = FFITypeMap::ffi_typecode<T>();
        }
        data = FFITypeHandler<T>::cast(_type, value);
    }
    
    ffiobj ffiobj::cast(pyobj py_obj, FFIType ffi_type, std::vector<size_t>& shp, FFIContainerType ctype) {
        auto dat = FFIConverter::from_python(ffi_type, py_obj, shp, ctype);
        ffiobj obj(dat, ffi_type, shp, ctype);
        obj.base = py_obj;
        return obj;
    }
    template <typename T>
    ffiobj ffiobj::cast(T value, std::vector<size_t>& shp, FFIContainerType ctype) {
        auto dat = FFITypeHandler<T>::genericize(value);
        return {dat, FFITypeMap::ffi_typecode<T>(), shp, ctype};
    }

    template<typename T>
    ffiobj ffiobj::cast(T value) {
        std::vector<size_t> shp;
        return ffiobj::cast<T>(value, shp, FFIContainerType::None);
    }

//    // recursive definition to loop through and test all the type pairs (thank you StackOverflow)
//    // we'll embed most conversion functions here so that we don't need to duplicate the boiler plate
//    template <typename...>
//    class FFIConversionManager;
//    template<>
//    class FFIConversionManager<> {
//    public:
//        static std::shared_ptr<void> from_python(FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            std::string garb = "unhandled type specifier in converting from python: " + std::to_string(static_cast<int>(type));
//            throw std::runtime_error(garb.c_str());
//        }
//        static std::shared_ptr<void> from_python_attr(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            std::string garb = "unhandled type specifier in converting from python: " + std::to_string(static_cast<int>(type));
//            throw std::runtime_error(garb.c_str());
//        }
//        static pyobj as_python(FFIType type,  std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            std::string garb = "unhandled type specifier in converting to python: " + std::to_string(static_cast<int>(type));
//            throw std::runtime_error(garb.c_str());
//        }
//    };
//    template <typename T, typename... Args> // expects FFITypePair objects
//    class FFIConversionManager<T, Args...> {
//    public:
//
//
//
//        static std::shared_ptr<void> from_python_direct(FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            // determine if our type is a pointer type
//            auto pos = FFIPointerTypes.find(type);
//            if (pos != FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
//                if (debug::debug_print(DebugLevel::All)) {
//                    auto garb = py_obj.repr();
//                    py_printf("Converting PyObject %s with pointer type FFIType %i\n", garb.c_str(), type);
//                }
//                // extract with a deleter if raw pointer
//                switch(ctype) {
//                    case(FFIContainerType::Vector):
//                        return std::make_shared<std::vector<typename T::type>>(
//                                mcutils::python::from_python<std::vector<typename T::type>>(py_obj)
//                        );
//                    default:
//                        return std::shared_ptr<void>(
//                                mcutils::python::from_python<typename T::type *>(py_obj)
//                                , [](typename T::type *val) {}// This came from python and will be managed by python // delete val; }
//                        );
//                }
//            } else {
//                if (debug::debug_print(DebugLevel::All)) {
//                    auto garb = py_obj.repr();
//                    py_printf("Converting PyObject %s with non-pointer type FFIType %i\n", garb.c_str(), type);
//                }
//                // is a pointer so no deletion
//                return std::make_shared<typename T::type>(
//                        mcutils::python::from_python<typename T::type>(py_obj)
//                );
//            }
//        }
//        static std::shared_ptr<void> from_python(FFIType type, pyobj py_obj, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            if (type == T::value) { // the type matches up with our stored type
//                return from_python_direct(type, py_obj, shape, ctype);
//            } else {
//                return FFIConversionManager<Args...>::from_python(type, py_obj, shape, ctype);
//            }
//        }
//
//        static std::shared_ptr<void> from_python_attr_direct(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            using D = typename T::type;
//
//            auto pos = FFIPointerTypes.find(type);
//            if (pos != FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
//                if (debug::debug_print(DebugLevel::All)) {
//                    auto garb = py_obj.repr();
//                    py_printf("Converting PyObject %s attr %s with pointer type FFIType %i\n", garb.c_str(), attr, type);
//                }
//                switch(ctype) {
//                    case(FFIContainerType::Vector):
//                        return std::make_shared<std::vector<D>>(py_obj.getattr<std::vector<D>>(attr));
//                    default:
//                        return std::shared_ptr<void>(
//                                py_obj.getattr<D*>(attr)
//                                , [](D* val){}// This came from python and will be managed by python //delete val;}
//                        );
//                }
//            } else {
//                // new managed instance so no deleter
//                if (debug::debug_print(DebugLevel::All)) {
//                    auto garb = py_obj.repr();
//                    py_printf("Converting PyObject %s attr %s with pointer type FFIType %i\n", garb.c_str(), attr, type);
//                }
//                return std::make_shared<D>(py_obj.getattr<D>(attr));
//            }
//        }
//        static std::shared_ptr<void> from_python_attr(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            if (type == T::value) { // the type matches up with our stored type
//                return from_python_attr_direct(type, py_obj, attr, shape);
//            } else {
//                return FFIConversionManager<Args...>::from_python_attr(type, py_obj, attr, shape);
//            }
//        }
//
//        static pyobj as_python_direct(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            using D = typename T::type;
//
//            // determine if our type is a pointer type
//            auto pos = FFIPointerTypes.find(type);
//            if (pos != FFIPointerTypes.end() || !shape.empty() || ctype != FFIContainerType::None) {
//                // is a pointer type so we convert regularly
//
//                switch(ctype) {
//                    case(FFIContainerType::Vector): {
//                        if (debug::debug_print(DebugLevel::All)) {
//                            py_printf("Converting type std::vector<%s> to python with FFIType %i\n", mcutils::type_name<D>::c_str(), type);
//                        }
//                        return FFITypeHandler<std::vector<D>>::as_python(type, data, shape);
//                    }
//                    default: {
//                        if (debug::debug_print(DebugLevel::All)) {
//                            py_printf("Converting type %s* to python with FFIType %i\n", mcutils::type_name<D>::c_str(), type);
//                        }
//                        return FFITypeHandler<D*>::as_python(type, data, shape);
//                    }
//                }
//            } else {
//
//                if (debug::debug_print(DebugLevel::All)) {
//                    py_printf("Converting type %s to python with non-pointer type FFIType %i\n", mcutils::type_name<D>::c_str(), type);
//                }
//                // not a pointer type so we extract data from shared_ptr as a pointer first
//                return FFITypeHandler<D>::as_python(type, data, shape);
//            }
//        }
//        static pyobj as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            if (type == T::value) {
//                return as_python_direct(type, data, shape, ctype);
//            } else {
//                return FFIConversionManager<Args...>::as_python(type, data, shape, ctype);
//            }
//        }
//    };
//    template <typename T, typename... Args> // expects FFITypePair objects
//    class FFIConversionManager<std::tuple<T, Args...>> {
//    public:
//
//        static std::shared_ptr<void> from_python(FFIType type, pyobj py_obj, std::vector<size_t> &shape, FFIContainerType ctype = None) {
//            return FFIConversionManager<T, Args...>::from_python(type, py_obj, shape, ctype);
//        }
//
//        static std::shared_ptr<void> from_python_attr(FFIType type, pyobj py_obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            return FFIConversionManager<T, Args...>::from_python_attr(type, py_obj, attr, shape);
//        }
//
//        static pyobj as_python(FFIType type, std::shared_ptr<void>& data, std::vector<size_t>& shape, FFIContainerType ctype = None) {
//            return FFIConversionManager<T, Args...>::as_python(type, data, shape, ctype);
//        }
//
//    };
//    using FFIConverter = FFIConversionManager<FFITypePairs>;

//    template <typename T>
//     pyobj ffi_to_python(std::shared_ptr<void>&data, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        return FFIConversionManager<T>::as_python_direct(FFITypeHandler<T>::typecode(), data, shape, ctype);
//    }
//    template <FFIType F>
//     pyobj ffi_to_python(std::shared_ptr<void>&data, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        using T = typename FFITypeMap::find_type<F>;
//        return FFIConversionManager<T>::as_python_direct(F, data, shape, ctype);
//    }
//     pyobj ffi_to_python(FFIType type, std::shared_ptr<void>&data, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        return FFIConverter::as_python(type, data, shape, ctype);
//    }
//
//    template <typename T>
//     std::shared_ptr<void> ffi_from_python_attr(pyobj obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        return FFIConversionManager<T>::from_python_attr_direct(FFITypeHandler<T>::typecode(), obj, attr, shape, ctype);
//    }
//    template <FFIType F>
//     std::shared_ptr<void> ffi_from_python_attr(pyobj obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        using T = typename FFITypeMap::find_type<F>;
//        return FFIConversionManager<T>::from_python_attr_direct(F, obj, attr, shape, ctype);
//    }
//     std::shared_ptr<void> ffi_from_python_attr(const FFIType type, pyobj obj, const char* attr, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        return FFIConverter::from_python_attr(type, obj, attr, shape, ctype);
//    }
//
//    template <typename T>
//     std::shared_ptr<void> ffi_from_python(pyobj obj, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        return FFIConversionManager<T>::from_python_direct(FFITypeHandler<T>::typecode(), obj, shape, ctype);
//    }
//    template <FFIType F>
//     std::shared_ptr<void> ffi_from_python_attr(pyobj obj, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        using T = typename FFITypeMap::find_type<F>;
//        return FFIConversionManager<T>::from_python_direct(F, obj, shape, ctype);
//    }
//     std::shared_ptr<void> ffi_from_python(const FFIType type, pyobj obj, std::vector<size_t>& shape, FFIContainerType ctype = FFIContainerType::None) {
//        return FFIConverter::from_python(type, obj, shape, ctype);
//    }

//    template <typename T>
//     std::shared_ptr<void> genericize(T data) {
//        auto ptr = FFITypeHandler<T>::genericize(data);
//        if (debug::debug_print(DebugLevel::All)) {
//            py_printf("Got generic version of object of type %s at %p\n", mcutils::type_name<T>::c_str(), ptr.get());
//        }
//        return ptr;
//    }

   /*
    * FFICompoundReturn defs
    *
    *
    */

    struct FFICompoundReturnCollator {
        using map = FFITypeMap;
        using pairs = FFITypePairs;

//        template <typename T>
//        class compound_return_vector_extractor {
//            FFICompoundReturn* buffer;
//            size_t nels;
//            size_t idx;
//        public:
//            compound_return_vector_extractor(FFICompoundReturn* b, size_t n, size_t i)
//                : buffer(b), nels(n), idx(i) {}
//            std::vector<T> extract_plain() {
//                std::vector<T> res(nels);
//                for (size_t i = 0; i < nels; i++) {
//                    auto r = buffer[i];
//                    res[i] = r.get_idx(idx).template convert<T>();
//                }
//                return res;
//            }
//            std::vector<std::vector<T>> extract_vector() {
//                std::vector<std::vector<T>> res(nels);
//                for (size_t i = 0; i < nels; i++) {
//                    auto r = buffer[i];
//                    res[i] = r.get_idx(idx).template convert<std::vector<T>>();
//                }
//                return res;
//            }
//        };

        template <typename D>
        static pyobj collate_to_python(
                   FFIType type, FFIContainerType ctype, std::vector<size_t>& shape,
                   FFICompoundReturn* buffer, size_t nels, size_t idx
           ) {
//            auto collator = compound_return_vector_extractor<D>(buffer, nels, idx);
            switch(ctype) {
                case FFIContainerType::Vector: {
                    std::vector<std::vector<D>> val(nels);
                    for (size_t i = 0; i < nels; i++) {
                        auto r = buffer[i];
                        val[i] = r.get_idx(idx).convert<std::vector<D>>();
                    }
                    return mcutils::python::numpy_from_data<D>(val, shape);
                };
                default: {
                    std::vector<D> val(nels);
                    for (size_t i = 0; i < nels; i++) {
                        auto r = buffer[i];
                        val[i] = r.get_idx(idx).convert<D>();
                    }
                    return mcutils::python::numpy_from_data<D>(val, shape);
                }
            }
        }
        template <FFIType F>
        static pyobj collate_to_python(
            FFIContainerType ctype, std::vector<size_t>& shape,
            FFICompoundReturn* buffer, size_t nels, size_t idx
        ) {
            using T = typename map::find_type<F>;
            return collate_to_python<T>(F, ctype, shape, buffer, nels, idx);
        }
        template <size_t N = 0> // Caller that will step through the pairs till we get where we need to be
        static auto collate_to_python_caller(size_t target, 
            FFIType type, FFIContainerType ctype, std::vector<size_t>& shape,
            FFICompoundReturn* buffer, size_t nels, size_t idx
        ) {
            if (target >= std::tuple_size_v<pairs>) {
                std::string msg = "FFIType index error in collate_to_python call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
            if (N == target) {
                using D = typename std::tuple_element_t<N, pairs>::type;
                if (debug::debug_print(DebugLevel::Excessive)) 
                    py_printf("    --> collating using dtype %s from base type %d (%s)\n", FFITypeMap::type_name<D>().c_str(), static_cast<int>(type), FFITypeMap::type_name(type).c_str());
                return collate_to_python<D>(type, ctype, shape, buffer, nels, idx);
            } 
            if constexpr (N + 1 < std::tuple_size_v<pairs>) {
                return collate_to_python_caller<N + 1>(target, type, ctype, shape, buffer, nels, idx);
            } else {
                std::string msg = "Unreachable: FFIType index error in collate_to_python call. Got idx " + std::to_string(idx) + " for FFIType " + std::to_string(static_cast<int>(type));
                throw std::runtime_error(msg);
            }
        }
        static auto collate_to_python(
            FFIType type, FFIContainerType ctype, std::vector<size_t>& shape,
            FFICompoundReturn* buffer, size_t nels, size_t idx
            ) {
            return collate_to_python_caller<0>(map::ffi_type_index(type), type, ctype, shape, buffer, nels, idx);
        }
    };

//
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
//
//    template <typename T>
//    class FFICompoundReturn::value_getter{
//        std::shared_ptr<void> data;
//        FFIType type;
//        FFIContainerType ctype;
//    public:
//        value_getter(
//                std::shared_ptr<void> dat,
//                FFIType typ,
//                FFIContainerType ctyp
//        ) : data(std::move(dat)), type(typ), ctype(ctyp) {}
//        T extract_plain() {
//            return FFITypeHandler<T>::cast(type, data);
//        }
//        std::vector<T> extract_vector() {
//            return FFITypeHandler<std::vector<T>>::cast(type, data);
//        }

//        explicit operator T() {
//            if (ctype == FFIContainerType::Vector) throw std::runtime_error("requested plain type but container type is Vector");
//            return extract_plain();
//        }
//        explicit operator std::vector<T>() {
//            if (ctype != FFIContainerType::Vector) throw std::runtime_error("requested std::vector but container type is not Vector");
//            return extract_vector();
//        }
//    };
//    template <typename T>
//    FFICompoundReturn::value_getter<T> FFICompoundReturn::get_idx(size_t idx) {
//        return value_getter<T>(data[idx], type.types()[idx], container_types()[idx]);
//    };
//    template <typename T>
//    FFICompoundReturn::value_getter<T> FFICompoundReturn::get(std::string key) {
//        auto idx = key_index(key);
//        return value_getter<T>(data[idx], type.types()[idx], container_types()[idx]);
//    };

   ffiobj FFICompoundReturn::get(std::string key) {
       auto idx = key_index(key);
       return objects[idx];
   }
   ffiobj FFICompoundReturn::get_idx(size_t idx) {
       return objects[idx];
   }

   template <typename T>
   void FFICompoundReturn::set(std::string key, T value, std::vector<size_t> shape) {
       auto idx = key_index(key);
       if (debug::debug_print(DebugLevel::All)) {
           py_printf("Setting key %s at index %d\n", key.c_str(), idx);
       }
       
       validate_type<T>(type.types()[idx]);
       if (debug::debug_print(DebugLevel::Excessive)) {
           py_printf("Type pair validated...\n", key.c_str(), idx);
       }

       auto shp = shape.empty() ? type.shapes()[idx] : shape;
       if (debug::debug_print(DebugLevel::Excessive)) {
            if (!shape.empty()) {
                py_printf("Using passed shape...\n", key.c_str(), idx);   
            } else {
                py_printf("Using stored shape...\n", key.c_str(), idx);   
            }
       }
       auto val = FFITypeHandler<T>::genericize(value);
       objects[idx] = ffiobj(
           val,
           type.types()[idx],
           shp,
           FFIContainerType::None
           );
   }
   template <typename T>
   void FFICompoundReturn::set(std::string key, T* value, std::vector<size_t> shape) {
       auto idx = key_index(key);
       if (debug::debug_print(DebugLevel::All)) {
           py_printf("Setting key %s at index %d\n", key.c_str(), idx);
       }
       validate_type<T>(type.types()[idx]);
       if (debug::debug_print(DebugLevel::All)) {
           py_printf("Type pair validated...\n", key.c_str(), idx);
       }
       auto shp = shape.empty() ? type.shapes()[idx] : shape;
       auto val = FFITypeHandler<T*>::genericize(value);
       objects[idx] = ffiobj(
           val,
           type.types()[idx],
           shp,
           FFIContainerType::Raw
           );
   }
   template <typename T>
   void FFICompoundReturn::set(std::string key, std::vector<T> value, std::vector<size_t> shape) {
       auto idx = key_index(key);
       if (debug::debug_print(DebugLevel::All)) {
           py_printf("Setting key %s at index %d\n", key.c_str(), idx);
       }
       validate_type<T>(type.types()[idx]);
       if (debug::debug_print(DebugLevel::All)) {
           py_printf("Type pair validated...\n", key.c_str(), idx);
       }
       auto shp = shape.empty() ? type.shapes()[idx] : shape;
       auto val = FFITypeHandler<std::vector<T>>::genericize(value);
       objects[idx] = ffiobj(
           val,
           type.types()[idx],
           shp,
           FFIContainerType::Vector
           );
   }
   
   pyobj FFICompoundReturn::as_python() {

       auto keys = type.keys();

       if (debug::debug_print(DebugLevel::All)) {
           py_printf("Building compound return type with keys: ");
           for (auto k:keys) {
               py_printf("%s ", k.c_str());
           }
           py_printf("\n");
       }

       if (objects.size() != keys.size()) {
           std::string msg = "Mismatch between number of keys and number of stored values ";
           msg += "(" + std::to_string(keys.size()) + " and " + std::to_string(objects.size()) + ")";
           throw std::runtime_error(msg.c_str());
       }

       std::vector<pyobj> res_obj(keys.size());
       for (size_t i=0; i < keys.size(); i++){
           if (debug::debug_print(DebugLevel::All)) {
               py_printf("casting value to python for key %s\n", keys[i].c_str());
           }

           res_obj[i] = objects[i].convert<pyobj>();

           if (debug::debug_print(DebugLevel::All)) {
               py_printf("got cast value %s\n", res_obj[i].repr().c_str());
           }
       }

       return mcutils::python::as_python_dict(keys, res_obj);
   }

//
//    /*
//     * FFIArgument defs
//     *
//     */
//
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

        std::string name() {return param_key;}
        std::vector<size_t> shape() {return shape_vec;}
        FFIType type() {return type_char;}

        pyobj as_tuple() {
            return pyobj(Py_BuildValue("(NNN)",
                                       mcutils::python::as_python_object<std::string>(param_key),
                                       mcutils::python::as_python_object<FFIType>(type_char),
                                       mcutils::python::as_python_tuple_object<size_t>(shape_vec)));
        }
        std::string repr() { return as_tuple().repr(); }
    };
//
//
    /*
     * FFIParameter defs
     *
     */

    class FFIParameter {
        // object that maps onto the python FFI stuff...
        pyobj base; // came from python
        ffiobj fobj; // stored C++ object
        std::string key; // always want named parameters
    public:
        FFIParameter(PyObject *obj, FFIArgument& arg) : base(obj), fobj(ffiobj(arg.type(), arg.shape())), key(arg.name()) {};
        FFIParameter(
                std::shared_ptr<void>& data,
                FFIArgument& arg
        ) : base(NULL), fobj(ffiobj(data, arg.type(), arg.shape())), key(arg.name()) {};
        explicit FFIParameter(
                FFIArgument& arg
        ) : base(NULL), fobj(ffiobj(arg.type(), arg.shape())), key(arg.name()) {};
        explicit FFIParameter(pyobj obj) : base(obj) { 
            if (debug::debug_print(DebugLevel::All)) {
                py_printf("Destructuring PyObject %s\n", base.repr().c_str());
            }

            if (debug::debug_print(DebugLevel::All)) py_printf("  > getting arg_name\n");
            key = base.getattr<std::string>("arg_name");
            
            if (debug::debug_print(DebugLevel::All)) py_printf("  > getting arg_type\n");
            auto type = base.getattr<FFIType>("arg_type");
            if (debug::debug_print(DebugLevel::Excessive)) py_printf("    > got %d (%s)\n", static_cast<int>(type), FFITypeMap::type_name(type).c_str());
            if (debug::debug_print(DebugLevel::All)) py_printf("  > getting arg_shape\n");
            auto shape = base.getattr<std::vector<size_t>>("arg_shape");
            if (debug::debug_print(DebugLevel::All)) py_printf("  > getting arg_val\n");
            auto attr = base.getattr<pyobj>("arg_value");
            auto data = FFIConverter::from_python(type, attr, shape); // pulls arg_value by default...
            
            fobj = ffiobj(data, type, shape);
            fobj.set_base(attr);

            // if (debug::debug_print(DebugLevel::All)) py_printf("  constructing FFIArgument...\n");
            // arg_spec = FFIArgument(name, type_char, shape);

         }

        FFIParameter() = default;

        std::string name() { return key; }
        std::vector<size_t> shape() { return fobj.shape(); }
        FFIType type() { return fobj.type(); }

        ffiobj obj() { return fobj; }
        template <typename T>
        T value() { return fobj.convert<T>(); }
        template <typename T>
        void set(T val) { fobj = ffiobj::cast<T>(val); }

        pyobj as_python() { return fobj.convert<pyobj>(); }
        PyObject* as_python_object() { return as_python().obj(); }
        std::string repr() { return as_python().repr(); }
    };

   class FFIParameters {
       // object that maps onto the python FFI stuff...
       pyobj py_obj;
       std::vector<FFIParameter> params;
       std::unordered_map<std::string, size_t> idx_map;
   public:
        FFIParameters()
            : py_obj(), params(), idx_map() {};
        explicit FFIParameters(pyobj param_obj)
            : py_obj(param_obj), idx_map() {
                if (debug::debug_print(DebugLevel::All))
                    py_printf("initializing parameters object from %s\n", py_obj.repr().c_str());
                params = py_obj.getattr<std::vector<FFIParameter>>("ffi_parameters");
                for (size_t i=0; i<params.size(); i++) {
                    idx_map[params[i].name()] = i;
                }
        }

        std::vector<std::string> keys() {
            std::vector<std::string> k(params.size());
            for (size_t i = 0; i < params.size(); i++) {
                auto p = params[i];
                auto key = p.name();
                k[i] = key;
                idx_map[key] = i;
            };
            return k;
        }

        size_t find_param_index(std::string& key) {
            size_t i;
            for (i = 0; i < params.size(); i++) {
                auto p = params[i];
                if (p.name() == key)
                    break;
            };
            return i;
        }
        size_t param_index(std::string& key) {
            auto pos = idx_map.find(key);
            if (pos == idx_map.end()) {
                size_t i = find_param_index(key);
                if (i == params.size())
                    throw std::runtime_error("parameter \"" + key + "\" not found");
                // printf("...setting plop for %s from %p?\n", key.c_str(), this);
                idx_map[key] = i;
                return i;
            } else {
                // printf("...resolving plop for %s from %p?\n", key.c_str(), this);
                auto i = pos->second;
                // printf("...got index %lu (true: %lu, idx_map size: %lu)\n", pos->second, find_param_index(key), idx_map.size());
                if (i >= params.size())
                    throw std::runtime_error("parameter \"" + key + "\" is disabled");
                return i;
            }
        }
        bool contains(std::string& key) {
            auto pos = idx_map.find(key);
            size_t i;
            if (pos == idx_map.end()) {
                i = find_param_index(key);
                if (i == params.size()) {
                    // printf("...no plop found for %s from %p?\n", key.c_str(), this);
                    return false;
                }
                // printf("...setting plop for %s from %p?\n", key.c_str(), this);
                idx_map[key] = i;
            } else {
                i = pos->second;
            }
            return i < params.size();
        }
        
        FFIParameter get_parameter(std::string& key) { return params[param_index(key)]; }
        FFIParameter get_parameter(const char* key) {
                std::string k = key;
                return get_parameter(k);
        }
        void set_parameter(std::string& key, FFIParameter& param) {
            if (!contains(key)) {
                params.push_back(param);
                idx_map[key] = params.size() - 1;
            } else {
                auto i = param_index(key);
                params[i] = param;
            }
        }
        void set_parameter(const char* key, FFIParameter& param) {
                std::string k = key;
                set_parameter(k, param);
        }
        void disable_parameter(std::string& key) {
            idx_map[key] = -1; // underflow
        }
        void disable_parameter(const char* key) {
            std::string k = key;
            disable_parameter(k);
        }
        void enable_parameter(std::string& key) {
            idx_map[key] = find_param_index(key); // reset
        }
        void enable_parameter(const char* key) {
            std::string k = key;
            enable_parameter(k);
        }

       template<typename T>
       T value(std::string &key) { 
            auto param = get_parameter(key);
            // printf("...wat?\n");
            return param.value<T>(); 
            }
       template<typename T>
       T value(const char *key) { 
            auto param = get_parameter(key);
            // printf("...wat?\n");
            return param.value<T>(); 
       }

       std::vector<size_t> shape(std::string &key) { return get_parameter(key).shape(); }
       std::vector<size_t> shape(const char *key) { return get_parameter(key).shape(); }

       FFIType typecode(std::string &key) { return get_parameter(key).type(); }
       FFIType typecode(const char *key) { return get_parameter(key).type(); }

    //    size_t param_index(std::string& key);
    //    FFIParameter get_parameter(std::string& key);
    //    FFIParameter get_parameter(const char* key);
    //    void set_parameter(std::string& key, FFIParameter& param);
    //    void set_parameter(const char* key, FFIParameter& param);

    //    template <typename T>
    //    T value(std::string& key) { return get_parameter(key).value<T>(); }
    //    template <typename T>
    //    T value(const char* key) { return get_parameter(key).value<T>(); }
    //    std::vector<size_t> shape(std::string& key);
    //    std::vector<size_t> shape(const char* key);
    //    FFIType typecode(std::string& key);
    //    FFIType typecode(const char* key);

   };

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
       auto shapes = rep.types().shapes();
       
       size_t nels = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

       auto keys = rep.types().keys();
       std::vector<pyobj> objs(keys.size());
       for (size_t i=0; i < keys.size(); i++) {
            auto obj = rep.get_idx(i);

           if (pyadeeb::debug_print(DebugLevel::All)) py_printf("     --> converting key %s\n", keys[i].c_str());

           // we need to concatenate the base data
           // shape with the stored return shape
           auto shp = obj.shape().empty()?shapes[i]:obj.shape();
           std::vector<size_t> new_shape(shape.size() + shp.size());
           std::copy(shape.begin(), shape.end(), new_shape.begin());
           std::copy(shp.begin(), shp.end(), new_shape.begin() + shape.size());

//            py_printf( "          - concat shape: ( ");
//            for (auto s: new_shape) py_printf( "%lu ", s);
//            py_printf(")\n");
//            py_printf( "          - passed shape: ( ");
//            for (auto s: shape) py_printf( "%lu ", s);
//            py_printf(")\n");
            
        //    if (pyadeeb::debug_print(DebugLevel::All)) py_printf("       >  FFITypes: %d (%s) and %d (%s)\n", 
        //     static_cast<int>(types[i]), plzffi::FFITypeMap::type_name(types[i]).c_str(),
        //     static_cast<int>(obj.type()), plzffi::FFITypeMap::type_name(obj.type()).c_str()
        //     );
           objs[i] = plzffi::FFICompoundReturnCollator::collate_to_python(
                   obj.type(), obj.container_type(), new_shape,
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
