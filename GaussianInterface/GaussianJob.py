import os, numpy as np
from collections import OrderedDict, namedtuple
from ..Data import AtomData

####################################################################################################################
#
#                                               GaussianJobException
#
class GaussianJobException(Exception):
    pass

####################################################################################################################
#
#                                               GJFOptFormatter
#
class GJFOptFormatter:
    @classmethod
    def format_base_opt(self, opt):
        return (str(opt), True)
    @classmethod
    def format_compound_opt(self, opt):
        return ("({})".format(", ".join(*(self.format_opt(o)[0] for o in opt))), True)
    @classmethod
    def format_bool_opt(self, opt):
        return (opt, False) if opt else ("", "ignore")
    @classmethod
    def format_none_opt(self, opt):
        return ("", "ignore")
    @classmethod
    def format_dict_opt(self, opt):
        chunks=[]
        for k, i in opt.items():
            opt_val, opt_tag = self.format_opt(i)
            if opt_tag is True:
                chunks.append("{}={}".format(k, opt_val))
            elif opt_tag is not "ignore":
                chunks.append(opt_val)
        return ("({})".format(", ".join(*chunks)), True)
    @classmethod
    def format_opt(self, opt):
        router = {
            str: self.format_base_opt,
            int: self.format_base_opt,
            tuple: self.format_compound_opt,
            list: self.format_compound_opt,
            bool: self.format_bool_opt,
            dict: self.format_dict_opt,
            OrderedDict: self.format_dict_opt,
            type(None): self.format_none_opt
        }
        meth = router[type(opt)]
        return meth(opt)
    @classmethod
    def format(self, opt_dict, tag):
        chunks=[]
        for k, i in opt_dict.items():
            opt_val, opt_tag = self.format_opt(i)
            if opt_tag is True:
                chunks.append(tag+"{}={}".format(k, opt_val))
            elif opt_tag is not "ignore":
                chunks.append(tag+k)
        return "\n".join(chunks)

####################################################################################################################
#
#                                               GaussianJob
#
class GaussianJob:
    """A class that writes Gaussian .gjf files given a system and config/template options"""

    job_template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Templates")

    def __init__(
            self,
            name,
            *args,
            description = None,
            system = None,
            job = None,
            config = None,
            template = "Template.gjf",
            file = None
            ):

        if not isinstance(name, str):
            raise GaussianJobException("{}: name argument must be string instead of {}".format(
                type(self).__name__,
                name
            ))
        self.name = name

        def throw_bad_field(self, field, must, val):
            raise GaussianJobException("{}: '{}' field must be of type {} (got {})".format(
                type(self).__name__, field, must, val
            ))
        if not isinstance(description, str):
            throw_bad_field(self, 'description', str, description)
        self.desc = description
        if not isinstance(system, self.System):
            throw_bad_field(self, 'system', self.System, system)
        self.system = system
        if not isinstance(job, self.Job):
            throw_bad_field(self, 'job', self.Job, job)
        self.job = job
        if not isinstance(config, self.Config):
            throw_bad_field(self, 'config', self.Config, config)
        self.config = config

        if file is None:
            import tempfile as tf
            f = tf.NamedTemporaryFile()
            dd = os.path.dirname(f.name)
            f.close()
            file = os.path.join(dd, name.replace(" ", "_") + ".gjf")
        if not "Chk" in config:
            config["Chk"] = os.path.splitext(file)[0] + ".chk"
        self.file = file
        self.temp = template

    def format(self):
        temp = self.temp
        if not ('{header}' in temp and '{job}' in temp):
            if not os.path.isfile(temp):
                temp = os.path.join(self.job_template_dir, temp)
            with open(temp) as t:
                temp = t.read()
        return temp.format(
            header = self.config.format(),
            description = self.desc,
            job = self.job.format(),
            system = self.system.format(job_type = self.job.job_type)
        )

    def write(self, file=None):
        import tempfile as tf

        if file is None:
            file = self.file
        if file is None:
            tmp = tf.NamedTemporaryFile()
            file = tmp.name + '.gjf'
            tmp.close()

        with open(file, 'w+') as gjf:
            gjf.write(self.format())

        return file

    def start(self, *cmd, binary = 'g09', **kwargs):
        """Starts a Gaussian job

        :param cmd:
        :type cmd:
        :param binary:
        :type binary:
        :param kwargs:
        :type kwargs:
        :return: started process
        :rtype:
        """
        import subprocess

        args = []
        args.extend(cmd)
        if len(args) == 0:
            args.append(binary)

        for k,i in kwargs.items():
            if i is True:
                args.append("-"+k)
            elif i is not None:
                args.append("-{}={:!r}".format(k, i))

        return subprocess.Popen(args)

    def run(self, *args, **kwargs):
        job = self.start(*args, **kwargs)
        job.wait()

    def __str__(self):
        return "{}({},\n\tfile = {} )".format(
            type(self).__name__,
            self.format(),
            self.file if not isinstance(self.file, str) else "'{}'".format(self.file)
        )

    class Job(OrderedDict):
        """Inner class for handling the main job options"""
        job_types = dict(
            {
                "SinglePoint": "SP",
                "Optimization": "Opt",
                "Frequency": "Freq",
                "ReactionPath": "IRC",
                "TransitionState": "IRCMax",
                "WavefunctionStability": "Stable",
                "BornOppenheimerDynamics": "BOMD",
                "DensityMatrixDynamics": "AOMD"
            },
            **{k:k for k in ["SP","Opt","Freq","IRC","IRCMax","Scan","Polar","ADMP", "BOMD","Force","Stable","Volume"]}
        )
        basis_set_keys = ['mp2', "cc" "aug", "sto"] # common basis set specs...
        def __init__(self, job_type = None, basis_set = None, **kw):
            for k in kw:
                if job_type is None and k in self.job_types:
                    job_type = k
                if basis_set is None:
                    for b in self.basis_set_keys:
                        if k.startswith(b):
                            basis_set = k
                            break
            if job_type is None:
                job_type = "SinglePoint"
            if basis_set is None:
                basis_set = "MP2/aug-cc-pvdz"
            self.job_type = self.job_types[job_type]
            if self.job_type != job_type and job_type in kw: # this is what I get for allowing things like ReactionPath -_-
                jtv = kw[job_type]
                del kw[job_type]
                kw[self.job_type] = jtv
            elif self.job_type not in kw:
                kw[self.job_type] = True

            self.basis_set = basis_set
            if basis_set not in kw:
                kw[basis_set] = True

            super().__init__(kw)

        def format(self):
            """Returns a formatted version of the job setup

            :return:
            :rtype:
            """
            return GJFOptFormatter.format(self, "#")

        def __str__(self):
            return self.format()

    class Config(OrderedDict):
        """Inner class for handling the main config options"""
        def format(self):
            """Returns a formatted version of the job setup

            :return:
            :rtype:
            """
            return GJFOptFormatter.format(self, "%")

        def __str__(self):
            return self.format()

    class System:
        def __init__(self,
                     charge = 0,
                     molecule = None,
                     vars = None,
                     bonds = None
                     ):
            self.charge = charge
            self.mol = molecule
            self.bonds = bonds
            self.vars = vars

        def format(self, job_type="SP"):
            return {
                "molecule":self.format_molecule(job_type),
                "charge": self.format_charges(),
                "bonds": self.format_bonds()
            }
        def format_molecule(self, job_type):
            blocks, vars = self.prep_mol(self.mol)
            if job_type == "Scan":
                blocks.extend(self.format_scan_vars(vars))
            elif job_type == "Opt":
                blocks.extend(self.format_opt_vars(vars))
            else:
                blocks.extend(self.format_const_vars(vars))

            return "\n".join(blocks)
        def format_charges(self):
            try:
                c, mult = self.charge
            except TypeError:
                c = self.charge
                mult = 1
            return "{} {}".format(c, mult)
        def format_bonds(self):
            b = self.bonds
            if b is None:
                return ""
            bond_block = [ None ]*len(b)
            for i,bond in enumerate(b):
                bond = bond if len(bond) > 2 else tuple(bond) + (1,)
                bond_block[i] = " {:d<5} {:d<5} {:d<5}".format(*bond)
            return "\n".join(bond_block)

        @staticmethod
        def format_opt_vars(vars):
            variables = vars["vars"]
            if len(variables) > 0:
                variables_blocks = [" Variables:"]
            else:
                variables_blocks = []
            for k,c in variables.items():
                if c[1] is None:
                    variables_blocks.append("  {:>6} = {:<12f}".format(k, *c))
                else:
                    variables_blocks.append("  {:>6} = {:<12f} s {:<5.0f} {:<12f}".format(k, *c))

            consts = vars["consts"]
            for k,c in consts.items():
                variables_blocks.append("  {:>6} = {:<12f} f".format(k, c))

            return variables_blocks

        @staticmethod
        def format_scan_vars(vars):
            variables = vars["vars"]
            if len(variables) > 0:
                variables_blocks = [" Variables:"]
            else:
                variables_blocks = []
            for k,c in variables.items():
                variables_blocks.append("  {:>6} = {:<12f} {:<5.0f} {:<12f}".format(k, *c))

            consts = vars["consts"]
            if len(consts) > 0:
                constants_blocks = [" Constants:"]
            else:
                constants_blocks = []
            for k,c in consts.items():
                constants_blocks.append("  {:>6} = {:f} 0 0.".format(k, c))

            return variables_blocks + constants_blocks

        @staticmethod
        def format_const_vars(vars):
            variables = vars["vars"]
            consts = vars["consts"]
            if len(consts) + len(variables) > 0:
                constants_blocks = [" Variables:"]
            else:
                constants_blocks = []
            for k,c in variables.items():
                constants_blocks.append("  {:>6} = {:f}".format(k, *c))
            for k,c in consts.items():
                constants_blocks.append("  {:>6} = {:f}".format(k, c))
            return constants_blocks

        @staticmethod
        def get_coord_type(crds):
            crd_type = None
            try:
                csys = crds.system # try to pull the coordinate system directly off the coordinates
            except AttributeError:
                try:
                    spec, zms = crds # if we can't find it try to get the spec and coordinates for a zmat
                    if not isinstance(spec[0], str):
                        raise ValueError("Oops")
                except ValueError:
                    dim0 = len(crds[0]) # if that doesn't work, then we just have a block of coordinates to work with
                    dimn1 = len(crds[-1]) # in that case we simply check lengths to see what we have
                    if dim0 == 1 or dimn1 == 6:
                        crd_type = "zmat"
                    elif dim0 == 3 and dimn1 == 3:
                        crd_type = "cart"
                else:
                    crd_type = "zmatspec"
            else:
                try:
                    from Coordinerds.CoordinateSystems import ZMatrixCoordinates, CartesianCoordinates3D
                    if isinstance(csys, ZMatrixCoordinates):
                        crd_type = "zmat"
                    elif isinstance(csys, CartesianCoordinates3D):
                        crd_type = "cart"
                except ImportError:
                    name = crd_type.__name__
                    if "ZMatrix" in name:
                        crd_type = "zmat"
                    elif "Cartesian" in name:
                        crd_type = "cart"

            return crd_type

        def prep_vars(self, *vars):
            var_map = { # to map variables to coordinates
                "vars"  : OrderedDict(),
                "consts": OrderedDict()
            }
            for var_spec in vars:
                if var_spec is not None:
                    for var in var_spec:
                        if isinstance(var, self.Variable) or len(var) == 4:
                            if not isinstance(var, self.Variable):
                                var = self.Variable(*var)
                            start = var.start
                            num   = var.num
                            step  = var.step
                            if isinstance(num, float):
                                num = np.ceil((num - start)/step)
                            var_map["vars"][var.name] = (start, num, step)
                            try:
                                del var_map["consts"][var.name]
                            except KeyError:
                                pass
                        elif isinstance(var, self.Constant):
                            var_map["consts"][var.name] = var.value
                            try:
                                del var_map["vars"][var.name]
                            except KeyError:
                                pass
                        else:
                            var_map["consts"][var[0]] = var[1]
                            try:
                                del var_map['vars'][var[0]]
                            except KeyError:
                                pass
            return var_map

        def prep_mol(self, mol):
            try:
                ats = mol["atoms"],
                crds = mol["coords"]
            except TypeError:
                try:
                    ats, crds = mol
                except ValueError:
                    ats = mol.atoms
                    crds = mol.crds

            atoms = [None]*len(ats)
            for i, a in enumerate(ats):
                main = AtomData[a, "PrimaryIsotope"]
                if main:
                    atoms[i] = AtomData[a, "Symbol"]
                else:
                    atoms[i] = "{}(Iso={})".format(AtomData[a, "Symbol"], AtomData[a, "MassNumber"])

            # get molspec blocks based on type of coordinates that were fed in
            crd_type = self.get_coord_type(crds)

            var_list = []
            blocks = [None]*len(crds)
            if crd_type == "zmat":
                for zz, a in zip(enumerate(crds), atoms):
                    subblock = [None]*7 # need a new list per line...
                    i,l = zz
                    subblock[0] = a
                    for j,el in enumerate(l):
                        if (j % 2) == 0:
                            subblock[j+1] = int(el)
                        elif isinstance(el, str):
                            # strings get treated as literals
                            subblock[j+1] = el
                        else:
                            var_type = "r" if j == 1 else "a" if j == 3 else "d" if j == 5 else "??"
                            var = var_type + str(i+1)
                            subblock[j+1] = var
                            var_list.append(self.Constant(var, el))

                    blocks[i] = "{:<20} {:<5} {:<12} {:<5} {:<12} {:<5} {:<12}".format(*subblock)

            elif crd_type == "zmatspec":
                # basically like zmat coords but we loop through two things at once
                spec, crds = crds
                for i, csa in enumerate(zip(crds, spec, atoms)):
                    l, s, a = csa
                    subblock = [None]*7 # need a new list per line...
                    subblock[0] = a
                    for j,e in enumerate(s):
                        subblock[1+2*j] = int(e)
                    for j,el in enumerate(l):
                        if isinstance(el, str):
                            # strings get treated as literals
                            subblock[2+2*j] = el
                        else:
                            var_type = "dist" if j == 0 else "angle" if j == 1 else "dihed" if j == 2 else "??"
                            var = var_type + str(i+1)
                            subblock[2+2*j] = var
                            var_list.append(self.Constant(var, el))
                    blocks[i] = "{:<20} {:<5} {:<12} {:<5} {:<12} {:<5} {:<12}".format(*subblock)
            elif crd_type == "cart":
                for i, ca in enumerate(zip(crds, atoms)):
                    c, a = ca
                    subblock = [None]*4
                    subblock[0] = a
                    for j,el in enumerate(c):
                        if isinstance(el, str):
                            # strings get treated as literals
                            subblock[1+j] = el
                        else:
                            var_type = "x" if j == 0 else "y" if j == 1 else "z" if j == 2 else "??"
                            var = var_type + str(i+1)
                            subblock[1+j] = var
                            var_list.append(self.Constant(var, el))
                    blocks[i] = "{:<20} 0 {:<12} {:<12} {:<12}".format(*subblock)

            var_map = self.prep_vars(var_list, self.vars)
            return blocks, var_map

        Variable = namedtuple("Variable", ["name", "start", "num", "step"])
        Variable.__new__.__defaults__ = (None,) * 4 # start and name really shouldn't take defaults but w/e
        Constant = namedtuple("Constant", ["name", "value"])