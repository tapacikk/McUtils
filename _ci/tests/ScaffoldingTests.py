from Peeves.TestUtils import *
from McUtils.Scaffolding import *
from unittest import TestCase
import numpy as np, io, os, tempfile as tmpf, json

class ScaffoldingTests(TestCase):

    @validationTest
    def test_HDF5Serialization(self):
        tmp = io.BytesIO()
        serializer = HDF5Serializer()

        data = [1, 2, 3]
        serializer.serialize(tmp, data)
        loaded = serializer.deserialize(tmp)
        self.assertEquals(loaded.tolist(), data)

        serializer.serialize(tmp, {
            "blebby": {
                "frebby": {
                    "clebby":data
                }
            }
        })
        loaded = serializer.deserialize(tmp, key='blebby')
        self.assertEquals(loaded['frebby']['clebby'].tolist(), data)

        mixed_data = [
            [1, 2, 3],
            "garbage",
            {"temps":[1., 2., 3.]}
            ]
        serializer.serialize(tmp, dict(mixed_data=mixed_data))

        loaded = serializer.deserialize(tmp, key='mixed_data')
        self.assertEquals(mixed_data, [
            loaded[0].tolist(),
            loaded[1].tolist().decode('utf-8'),
            {k:v.tolist() for k,v in loaded[2].items()}
        ])
    @validationTest
    def test_JSONSerialization(self):
        tmp = io.StringIO()
        serializer = JSONSerializer()

        data = [1, 2, 3]
        serializer.serialize(tmp, data)
        tmp.seek(0)
        loaded = serializer.deserialize(tmp)
        self.assertEquals(loaded, data)

        tmp = io.StringIO()
        serializer.serialize(tmp, {
            "blebby": {
                "frebby": {
                    "clebby":data
                }
            }
        })
        tmp.seek(0)
        loaded = serializer.deserialize(tmp, key='blebby')
        self.assertEquals(loaded['frebby']['clebby'], data)

        tmp = io.StringIO()
        mixed_data = [
            [1, 2, 3],
            "garbage",
            {"temps":[1., 2., 3.]}
            ]
        serializer.serialize(tmp, dict(mixed_data=mixed_data))
        tmp.seek(0)
        loaded = serializer.deserialize(tmp, key='mixed_data')
        self.assertEquals(mixed_data, loaded)
    @validationTest
    def test_NumPySerialization(self):
        tmp = io.BytesIO()
        serializer = NumPySerializer()

        data = [1, 2, 3]
        serializer.serialize(tmp, data)
        tmp.seek(0)
        loaded = serializer.deserialize(tmp)
        self.assertEquals(loaded.tolist(), data)

        tmp = io.BytesIO()
        serializer.serialize(tmp, {
            "blebby": {
                "frebby": {
                    "clebby": data
                }
            }
        })
        tmp.seek(0)
        loaded = serializer.deserialize(tmp, key='blebby')
        self.assertEquals(loaded['frebby']['clebby'].tolist(), data)

        tmp = io.BytesIO()
        mixed_data = [
            [1, 2, 3],
            "garbage",
            {"temps": [1., 2., 3.]}
        ]
        serializer.serialize(tmp, dict(mixed_data=mixed_data))
        tmp.seek(0)
        loaded = serializer.deserialize(tmp, key='mixed_data')
        self.assertEquals(mixed_data, [
            loaded[0].tolist(),
            loaded[1].tolist(),
            {k: v.tolist() for k, v in loaded[2].items()}
        ])
    @validationTest
    def test_JSONCheckpointing(self):

        with tmpf.NamedTemporaryFile() as chk_file:
            my_file = chk_file.name
        try:
            with JSONCheckpointer(my_file) as chk:
                # do something
                data = [1, 2, 3]
                chk['step_1'] = data
                # do something else
                chk['step_2_params'] = {
                    'steps': 500,
                    'step_size': .1,
                    'method': 'implicit euler'
                }

                # blobby
                data_2 = np.random.rand(100)
                chk['step_2'] = data_2

            # do some other stuff, maybe need to reload from checkpoint?
            with JSONCheckpointer(my_file) as chk:
                self.assertEquals(len(chk['step_2']), 100)
        finally:
            os.remove(my_file)
    @validationTest
    def test_NumPyCheckpointing(self):

        with tmpf.NamedTemporaryFile() as chk_file:
            my_file = chk_file.name

        try:
            with NumPyCheckpointer(my_file) as chk:
                # do something
                data = [1, 2, 3]
                chk['step_1'] = data
                # do something else
                chk['step_2_params'] = {
                    'steps': 500,
                    'step_size': .1,
                    'method': 'implicit euler'
                }

                # blobby
                data_2 = np.random.rand(100)
                chk['step_2'] = data_2

            # do some other stuff, maybe need to reload from checkpoint?
            with NumPyCheckpointer(my_file) as chk:
                self.assertEquals(len(chk['step_2']), 100)
        finally:
            os.remove(my_file)
    @validationTest
    def test_HDF5Checkpointing(self):

        with tmpf.NamedTemporaryFile(mode="w+b") as chk_file:
            my_file = chk_file.name
        try:
            with HDF5Checkpointer(my_file) as chk:
                # do something
                data = [1, 2, 3]
                chk['step_1'] = data
                # do something else
                chk['step_2_params'] = {
                    'steps': 500,
                    'step_size': .1,
                    'method': 'implicit euler'
                }

                # blobby
                data_2 = np.random.rand(100)
                chk['step_2'] = data_2

            # do some other stuff, maybe need to reload from checkpoint?
            with HDF5Checkpointer(my_file) as chk:
                self.assertEquals(len(chk['step_2']), 100)
        finally:
            os.remove(my_file)

    @debugTest
    def test_Logging(self):
        stdout = io.StringIO()
        logger = Logger(stdout)
        with logger.block(tag='Womp Womp'):
            logger.log_print('wompy dompy domp')
            logger.log_print('Some other useful info?')
            with logger.block(tag="Calling into subprogram"):
                logger.log_print('actually this is fake -_-')
                logger.log_print('took {timing:.5f}s', timing=121.01234)
            logger.log_print('I guess following up on that?')
            with logger.block(tag="Calling into subprogram"):
                logger.log_print('this is also fake! :yay:')
                logger.log_print('took {timing:.5f}s', timing=212.01234)

            logger.log_print('done for now; took {timing:.5f}s', timing=-1)


        with logger.block(tag='Surprise second block!'):
            logger.log_print('just kidding')
            with logger.block(tag="JK on that JK"):
                with logger.block(tag="Doubly nested block!"):
                    logger.log_print('woopy doopy doo bitchez')
                logger.log_print('(all views are entirely my own and do not reflect on my employer in any way)')

            logger.log_print('okay done for real; took {timing:.0f} years', timing=10000)

        # print("\n" + stdout.getvalue())

        with tmpf.NamedTemporaryFile(mode="w+b") as temp:
            log_dump = temp.name
        try:
            with open(log_dump, "w+") as dump:
                dump.write(stdout.getvalue())
            with LogParser(log_dump) as parser:
                blocks = list(parser)
                print(blocks[1].lines[1], "\n", blocks[1].lines[1].lines)
        finally:
            os.remove(log_dump)



