import uuid

from .CoordinateSystem import CoordinateSystem
from .CoordinateSystemConverter import CoordinateSystemConverter
import weakref

__all__ = [
    "CompositeCoordinateSystem",
    "CompositeCoordinateSystemConverter"
]

#TODO: these should all be metaclasses but :shrag:
class CompositeCoordinateSystem(CoordinateSystem):
    """
    Defines a coordinate system that comes from applying a transformation
    to another coordinate system
    """

    _register_cache = weakref.WeakValueDictionary()
    def __init__(self, base_system, conversion, batched=True, inverse_conversion=None, name=None, **opts):
        self.base_system = base_system
        self.conversion = conversion
        self.inverse_conversion = inverse_conversion
        self.batched = batched
        super().__init__(**opts)
        self.name = self.canonical_name(name, conversion)
    @classmethod
    def canonical_name(cls, name, conversion):
        if name is None:
            if hasattr(conversion, 'name'):
                name = conversion.name
            else:
                name = str(uuid.uuid4()).replace("-", "")
        return name
        #
        #     return type('CompositeCoordinateSystem' + name, (cls,),
        #                 {'base_system': base_system, 'conversion': conversion, 'inverse': inverse})
        # # if self.base_system is None:
        # #     raise ValueError('{name} is a factory class and {name}.{method} should be used to register coordinate systems'.format(
        # #         name=type(self).__name__,
        # #         method='register'
        # #     ))
        # # super().__init__()

    @classmethod
    def register(cls, base_system, conversion, batched=True, inverse_conversion=None, name=None):
        if (base_system, conversion) not in cls._register_cache:
            system_class = cls(base_system, conversion, batched=batched, inverse_conversion=inverse_conversion, name=name)
            CompositeCoordinateSystemConverter(system_class).register()
            if system_class.inverse_conversion is not None:
                CompositeCoordinateSystemConverter(system_class, direction='inverse').register()
            cls._register_cache[(base_system, conversion)] = system_class
        return cls._register_cache[(base_system, conversion)]
    def unregister(self):
        raise NotImplementedError("destructor not here yet")
    def __repr__(self):
        return "{}({}, {})".format(
            type(self).__name__,
            self.base_system,
            self.name
        )

class CompositeCoordinateSystemConverter(CoordinateSystemConverter):
    def __init__(self, system, direction='forward'):
        self.system = system
        self.direction = direction
    @property
    def types(self):
        if self.direction == 'inverse':
            return (self.system, self.system.base_system)
        else:
            return (self.system.base_system, self.system)
    def convert(self, coords, **kw):
        if self.direction == 'forward':
            convertser = self.system.conversion
        elif self.direction == 'inverse':
            convertser = self.system.inverse_conversion
        else:
            raise NotImplementedError("bad value for '{}': {}".format('direction', self.direction))
        return convertser(coords, **kw)
    def convert_many(self, coords, **kw):
        if self.system.batched:
            return self.convert(coords, **kw)
        else:
            return super().convert_many(coords, **kw)
