### <a id="McUtils.McUtils.Extensions.ArgumentSignature.<PrimitiveType Instance>"><PrimitiveType Instance></a>
Defines a general purpose ArgumentType so that we can easily manage complicated type specs
    The basic idea is to define a hierarchy of types that can then convert themselves down to
    a `ctypes`-style spec as well as a C++ argument spec so that we can enable `SharedLibraryFunction`
    to use either the basic `ctypes` FFI or a more efficient, but fragile system based off of extension modules

