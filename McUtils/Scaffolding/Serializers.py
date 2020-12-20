"""
Provides scaffolding for creating serializers that dump data to a reloadable format.
Light-weight and unsophisticated, but that's what makes this useful..
"""

import abc, numpy as np, json, io
from collections import OrderedDict

__all__= [
    "BaseSerializer",
    "JSONSerializer",
    "NumPySerializer",
    "HDF5Serializer",
    "YAMLSerializer",
    "ModuleSerializer"
]

class ConvertedData:
    """
    Wrapper class for holding serialized data so we can be sure it's clean
    """
    def __init__(self, data, serializer):
        self.data = data
        self.serializer = serializer
class BaseSerializer(metaclass=abc.ABCMeta):
    """
    Serializer base class to define the interface
    """
    @abc.abstractmethod
    def convert(self, data):
        """
        Converts data into a serializable format
        :param data:
        :type data:
        :return:
        :rtype:
        """
        raise NotImplementedError("BaseSerializer is an abstract class")
    @abc.abstractmethod
    def deconvert(self, data):
        """
        Converts data from a serialized format into a python format
        :param data:
        :type data:
        :return:
        :rtype:
        """
        raise NotImplementedError("BaseSerializer is an abstract class")
    @abc.abstractmethod
    def serialize(self, file, data, **kwargs):
        """
        Writes the data
        :param file:
        :type file:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        raise NotImplementedError("BaseSerializer is an abstract class")
    @abc.abstractmethod
    def deserialize(self, file, **kwargs):
        """
        Loads data from a file
        :param file:
        :type file:
        :return:
        :rtype:
        """
        raise NotImplementedError("BaseSerializer is an abstract class")

class JSONSerializer(BaseSerializer):
    """
    A serializer that makes dumping data to JSON simpler
    """
    class BaseEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return json.JSONEncoder.default(self, obj)
    def __init__(self, encoder=None):
        if encoder is None:
            self.encoder = self.BaseEncoder()
    def convert(self, data):
        return ConvertedData(self.encoder.encode(data), self)
    def deconvert(self, data):
        return data
    def serialize(self, file, data, **kwargs):
        if isinstance(data, ConvertedData):
            file.write(data.data)
        else:
            json.dump(data, file, cls=self.BaseEncoder, **kwargs)
    def deserialize(self, file, key=None, **kwargs):
        dat = json.load(file)
        dat = self.deconvert(dat)
        if key is not None:
            if '/' in key:
                key = key.split("/")
                for k in key:
                    dat = dat[k]
            else:
                return dat[key]
        else:
            return dat

class YAMLSerializer(BaseSerializer):
    """
    A serializer that makes dumping data to JSON simpler
    """
    def __init__(self):
        # just checks that we do really have YAML support...
        import yaml as api
        self.api = api

    def convert(self, data):
        return ConvertedData(data, self)
    def deconvert(self, data):
        return data
    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        data = data.data
        self.api.dump(data, file, **kwargs)
    def deserialize(self, file, key=None, **kwargs):
        dat = self.api.load(file)
        dat = self.deconvert(dat)
        if key is not None:
            if '/' in key:
                key = key.split("/")
                for k in key:
                    dat = dat[k]
            else:
                return dat[key]
        else:
            return dat

class HDF5Serializer(BaseSerializer):
    """
    Defines a serializer that can prep/dump python data to HDF5.
    To minimize complexity, we always use NumPy as an interface layer.
    This restricts what we can serialize, but generally in insignificant ways.
    """

    def __init__(self):
        import h5py as api
        self.api = api

    # we define a converter layer that will coerce everything to NumPy arrays
    atomic_types = (str, int, float)
    converter_dispatch = OrderedDict((
        ((np.ndarray,), lambda data, cls: data),
        ('asarray', lambda data, cls: data.asarray()),
        (atomic_types, lambda x, cls: cls._literal_to_numpy(x)),
        ((dict,), lambda data, cls: cls._dict_to_numpy(data)),
        ((list, tuple), lambda data, cls: cls._iterable_to_numpy(data))
    ))

    @classmethod
    def _literal_to_numpy(cls, data):
        return np.array([data]).reshape(())

    @classmethod
    def _dict_to_numpy(cls, data):
        return {k:cls._convert(v) for k,v in data.items()}

    @classmethod
    def _iterable_to_numpy(cls, data):
        arr = np.array(data)
        if arr.dtype == np.dtype(object):
            # map iterable into a stack of datasets :|
            return dict({'_list_item_'+str(i):cls._convert(v) for i,v in enumerate(data)}, _list_numitems=cls._convert(len(data)))
        else:
            return arr

    @classmethod
    def _convert(cls, data):
        """
        Recursively loop through, test data, make sure HDF5 compatible
        :param data:
        :type data:
        :return:
        :rtype:
        """
        converter = None
        for k, f in cls.converter_dispatch.items():
            if isinstance(k, tuple):  # check if we're dispatching based on type
                if isinstance(data, k):
                    converter = f
            elif isinstance(k, str):  # check if we're duck typing based on attributes
                if hasattr(data, k):
                    converter = f
            elif k(data):  # assume dispatch key is a callable that tells us if data is compatible
                converter = f

        if converter is None:
            raise TypeError("no registered converter to coerce {} into HDF5 compatible format".format(data))

        return converter(data, cls)

    def convert(self, data):
        """
        Converts data into format that can be serialized easily

        :param data:
        :type data:
        :return:
        :rtype:
        """
        return ConvertedData(self._convert(data), self)

    def _write_data(self, h5_obj, key, data):
        """
        Writes a numpy array into a group
        :param h5_group:
        :type h5_group: h5py.Group
        :param key:
        :type key: str
        :param data:
        :type data: np.ndarray
        :return:
        :rtype:
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('trying to write non-numpy data {} to key "{}"'.format(data, key))

        dtype_name = str(data.dtype)
        if '<U' in dtype_name:
            # If so, convert the array to one with bytes
            data = data.astype(dtype=dtype_name.replace('<U', '|S'))

        try:
            ds = h5_obj[key] #type: h5py.Dataset
        except KeyError:
            ds = h5_obj.create_dataset(key, data=data)
        else:
            ds[...] = data
        # no need to return stuff, since we're just serializing
    def _write_dict(self, h5_obj, data):
        """
        Writes a dict into a group
        :param h5_group:
        :type h5_group: h5py.Group
        :param data:
        :type data: dict
        :return:
        :rtype:
        """
        for k,v in data.items():
            # we want to either make a new Group or write the array to the key
            if isinstance(v, dict):
                try:
                    new_grp = h5_obj[k]
                except KeyError:
                    new_grp = h5_obj.create_group(k)
                self._write_dict(new_grp, v)
            else:
                self._write_data(h5_obj, k, v)
    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        if not isinstance(file, (self.api.File, self.api.Group)):
            file = self.api.File(file, "a")
        data = data.data
        if isinstance(data, np.ndarray):
            key = "_data"
            self._write_data(file, key, data)
        else:
            self._write_dict(file, data)

    def deconvert(self, data):
        """
        Converts an HDF5 Dataset into a NumPy array or Group into a dict
        :param data:
        :type data:
        :return:
        :rtype:
        """
        if isinstance(data, self.api.Dataset):
            res = np.empty(data.shape, dtype=data.dtype)
            data.read_direct(res)
        else:
            # we loop through the keys and recursively build up a dict
            res = {}
            for k, v in data.items():
                res[k] = self.deconvert(v)
            if '_list_numitems' in res:
                # actually an iterable but with inconsistent shape
                n_items = res['_list_numitems']
                res = [ res['_list_item_' + str(i)] for i in range(n_items) ]
            elif list(res.keys()) == ['_data']: # special case for if we just saved a single array to file
                res = res['_data']
        return res

    def deserialize(self, file, key=None, **kwargs):
        if not isinstance(file, (self.api.File, self.api.Group)):
            file = self.api.File(file, "r")
        if key is not None:
            file = file[key]
        return self.deconvert(file)

class NumPySerializer(BaseSerializer):
    """
    A serializer that makes implements NPZ dumps
    """

    # we define a converter layer that will coerce everything to NumPy arrays
    atomic_types = (str, int, float)
    converter_dispatch = OrderedDict((
        ((np.ndarray,), lambda data, cls: data),
        ('asarray', lambda data, cls: data.asarray()),
        (atomic_types, lambda x, cls: cls._literal_to_numpy(x)),
        ((dict,), lambda data, cls: cls._dict_to_numpy(data)),
        ((list, tuple), lambda data, cls: cls._iterable_to_numpy(data))
    ))

    @classmethod
    def _literal_to_numpy(cls, data):
        return np.array([data]).reshape(())

    @classmethod
    def _dict_to_numpy(cls, data):
        return {k: cls._convert(v) for k, v in data.items()}

    @classmethod
    def _iterable_to_numpy(cls, data):
        arr = np.array(data)
        if arr.dtype == np.dtype(object):
            # map iterable into a stack of datasets :|
            return dict({'_list_item_' + str(i): cls._convert(v) for i, v in enumerate(data)},
                        _list_numitems=cls._convert(len(data)))
        else:
            return arr

    @classmethod
    def _convert(cls, data):
        """
        Recursively loop through, test data, make sure NumPy compatible
        :param data:
        :type data:
        :return:
        :rtype:
        """
        converter = None
        for k, f in cls.converter_dispatch.items():
            if isinstance(k, tuple):  # check if we're dispatching based on type
                if isinstance(data, k):
                    converter = f
            elif isinstance(k, str):  # check if we're duck typing based on attributes
                if hasattr(data, k):
                    converter = f
            elif k(data):  # assume dispatch key is a callable that tells us if data is compatible
                converter = f

        if converter is None:
            raise TypeError("no registered converter to coerce {} into HDF5 compatible format".format(data))

        return converter(data, cls)

    dict_key_sep="::>|<::" # we pick a somewhat goofy separator to avoid conflicts in most cases
    def _flatten_dict(self, d, sep=None):
        """
        :param d:
        :type d: dict
        :param sep:
        :type sep: str | None
        :return:
        :rtype:
        """
        if sep is None:
            sep = self.dict_key_sep
        new = {}
        for k, v in d.items():
            if isinstance(v, dict):
                d2 = self._flatten_dict(v, sep=sep)
                new.update({k+sep+k2:v2 for k2,v2 in d2.items()})
            else:
                new[k] = v
        return new

    def convert(self, data):
        first_pass = self._convert(data)
        if isinstance(first_pass, dict):
            # we need to flatten out nested dictionaries
            # we're hoping that we won't have massively nested structures
            # since those will totally wreck performance
            data = self._flatten_dict(first_pass)
        else:
            data = first_pass
        return ConvertedData(data, self)

    def _deconvert_val(self, data):
        if isinstance(data, dict):
            # we check to make sure we don't have an implicitly encoded mixed-type list
            if '_list_numitems' in data:
                # actually an iterable but with inconsistent shape
                n_items = data['_list_numitems']
                res = [ data['_list_item_' + str(i)] for i in range(n_items) ]
            else:
                res = {k: self._deconvert_val(v) for k,v in data.items()}
        else:
            res = data
        return res

    def deconvert(self, data, sep=None):
        """
        Unflattens nested dictionary structures so that the original data
        can be recovered
        :param data:
        :type data:
        :param sep:
        :type sep: str | None
        :return:
        :rtype:
        """
        if sep is None:
            sep = self.dict_key_sep
        if isinstance(data, np.ndarray):
            return data
        else:
            data = dict(data.items())
            # now we have to _unflatten_ the flattened dicts -_-
            keys = list(data.keys())
            remapping = [(k.split(sep), k) for k in keys]
            new_data = {}
            for k_list, main_key in remapping:
                where_do_i_go = new_data
                for k in k_list[:-1]:
                    if k not in where_do_i_go:
                        where_do_i_go[k] = {}
                        where_do_i_go = where_do_i_go[k]
                    else:
                        where_do_i_go = where_do_i_go[k]
                where_do_i_go[k_list[-1]] = data[main_key]
            return self._deconvert_val(new_data)

    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        data = data.data
        if isinstance(data, np.ndarray):
            return np.save(file, data)
        else:
            return np.savez(file, **data)
    def deserialize(self, file, key=None, **kwargs):
        dat = np.load(file)
        dat = self.deconvert(dat)
        if isinstance(dat, np.ndarray):
            return dat
        if key is not None:
            if '/' in key:
                key = key.split("/")
                for k in key:
                    dat = dat[k]
            else:
                return dat[key]
        else:
            return dat

class ModuleSerializer(BaseSerializer):
    """
    A somewhat hacky serializer that supports module-based serialization.
    Writes all module parameters to a dict with a given attribute.
    Serialization doesn't support loading arbitrary python code, but deserialization does.
    Use at your own risk.
    """
    default_loader = None
    default_attr = "config"
    def __init__(self, attr=None, loader=None):
        self._loader =loader
        self._attr = attr

    @property
    def loader(self):
        if self._loader is None:
            if self.default_loader is None:
                self.default_loader = self._get_loader()
            return self.default_loader
        else:
            return self._loader
    @property
    def attr(self):
        if self._attr is None:
            return self.default_attr
        else:
            return self._attr
    @classmethod
    def _get_loader(cls):
        from McUtils.Extensions import ModuleLoader
        return ModuleLoader(rootpkg="Configs")

    def convert(self, data):
        return ConvertedData(data, self)
    def deconvert(self, data):
        return data
    def serialize(self, file, data, **kwargs):
        if not isinstance(data, ConvertedData):
            data = self.convert(data)
        data = data.data
        jsoner = JSONSerializer()
        with io.StringIO() as fake:
            jsoner.serialize(fake, data)
            serialized = fake.getvalue()
        print(
            "{} = {}".format(self.attr, serialized),
            file=file
        )
    def deserialize(self, file, key=None, **kwargs):
        module = self.loader.load(file)
        dat = self.deconvert(getattr(module, self.attr))
        if key is not None:
            if '/' in key:
                key = key.split("/")
                for k in key:
                    dat = dat[k]
            else:
                return dat[key]
        else:
            return dat