"""
Graphics primitives module
Provides stuff like Disk, Sphere, etc. and lets them figure out how to plot themselves
"""

__all__ = [
    "GraphicsPrimitive", "Sphere", "Cylinder", "Disk", "Line"
]

import abc
from .Graphics import VTKWindow

class GraphicsPrimitive(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def plot(self, graphics, *args, **kwargs):
        """The one method that needs to be implemented, which takes the graphics and actually puts stuff on its axes

        :param graphics:
        :type graphics:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        pass

class Disk(GraphicsPrimitive):
    def __init__(self, position = [0, 0], radius = 1, **opts):
        self.pos = position
        self.rad = (100*radius)**2
        self.opts = opts
        self.prim = None

    def plot(self, graphics, *args, zdir = None, **kwargs):

        if isinstance(graphics.figure, VTKWindow):
            self.prim = VTKDisk(self.pos, self.rad, **self.opts)
            s = self.prim
            return s.plot(graphics.figure)
        else:
            import numpy as np
            pt = np.array([self.pos]).T
            kw = dict(self.opts, s = [ self.rad ], edgecolors = [ [.95]*3 +[.5] ], **kwargs)
            s = graphics.scatter(*pt, **kw)

        return s

class Line(GraphicsPrimitive):
    def __init__(self, pos1, pos2, radius, **opts):
        self.pos1 = pos1
        self.pos2 = pos2
        self.rad = 72*radius # this can't be configured nearly as cleanly as the circle stuff...
        self.opts = opts
    def plot(self, graphics, *args, **kwargs):
        if isinstance(graphics.figure, VTKWindow):
            self.prim = VTKLine(self.pos1, self.pos2, self.rad, **self.opts)
            s = self.prim
            return s.plot(graphics)
        else:
            import numpy as np
            pos = np.array([self.pos1, self.pos2]).T

            kw = dict(self.opts, linewidth = self.rad, **kwargs)
            line = graphics.plot(*pos, **kw)

            return line

class Sphere(GraphicsPrimitive):
    def __init__(self, position = [0, 0, 0], radius = 1, sphere_points = 48, **opts):
        self.pos = position
        self.rad = radius
        self.opts = opts
        self.sphere_points = sphere_points

    def plot(self, graphics, *args, sphere_points = None, **kwargs):
        if isinstance(graphics.figure, VTKWindow):
            self.prim = VTKSphere(self.pos, self.rad, **self.opts)
            s = self.prim
            return s.plot(graphics.figure)
        else:
            import numpy as np
            # create basic sphere points
            if sphere_points is None:
                sphere_points = self.sphere_points
            u = np.linspace(0, 2 * np.pi, sphere_points)
            v = np.linspace(0, np.pi, sphere_points)
            x = self.rad * np.outer(np.cos(u), np.sin(v))
            y = self.rad * np.outer(np.sin(u), np.sin(v))
            z = self.rad * np.outer(np.ones(np.size(u)), np.cos(v))

            kw = dict(self.opts, **kwargs)
            return graphics.plot_surface(x+self.pos[0], y+self.pos[1], z+self.pos[2], *args, **kw)

class Cylinder(GraphicsPrimitive):
    def __init__(self, p1, p2, radius, circle_points = 32, **opts):
        self.pos1 = p1
        self.pos2 = p2
        self.rad = radius
        self.opts = opts
        self.circle_points = circle_points

    def plot(self, graphics, *args, circle_points = None, **kwargs):
        if isinstance(graphics.figure, VTKWindow):
            self.prim = VTKCylinder(self.pos1, self.pos2, self.rad, **self.opts)
            s = self.prim
            return s.plot(graphics.figure)
        else:
            import numpy as np
            # create basic sphere points
            if circle_points is None:
                circle_points = self.circle_points
            u = np.linspace(0, 2 * np.pi, circle_points)
            v = np.linspace(0, np.pi, circle_points)

            # pulled from SO: https://stackoverflow.com/a/32383775/5720002

            #vector in direction of axis
            v = self.pos2 - self.pos1
            #find magnitude of vector
            mag = np.linalg.norm(v)
            #unit vector in direction of axis
            v = v / mag
            #make some vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            #make vector perpendicular to v
            n1 = np.cross(v, not_v)
            #normalize n1
            n1 /= np.linalg.norm(n1)
            #make unit vector perpendicular to v and n1
            n2 = np.cross(v, n1)
            #surface ranges over t from 0 to length of axis and 0 to 2*pi
            t = np.linspace(0, mag, circle_points)
            theta = np.linspace(0, 2 * np.pi, circle_points)
            #use meshgrid to make 2d arrays
            t, theta = np.meshgrid(t, theta)
            #generate coordinates for surface
            X, Y, Z = [self.pos1[i] + v[i] * t + self.rad * np.sin(theta) * n1[i] + self.rad * np.cos(theta) * n2[i] for i in [0, 1, 2]]

            kw = dict(self.opts, **kwargs)
            return graphics.plot_surface(X, Y, Z, *args, **kw)


#####################################################################################################################
#
#                                           VTKPrimitives
#
class VTKPrimitive:
    def __init__(self,
                 get_mapper,
                 name,
                 color = None,
                 parent = None
                 # we'll add other props as we come along them
                 ):
        self.name = name
        self.parent = parent
        self._mapper = None
        self._actor = None
        self._get_mapper = get_mapper
        self._color = None
        if color is not None:
            self.set_color(color)

    @property
    def actor(self):
        return self.get_actor()

    @staticmethod
    def _setup_actor(mapper):
        import vtk
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor
    @staticmethod
    def _setup_mapper(source):
        import vtk
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        return mapper

    def get_actor(self):
        if self._actor is None:
            self._actor = self._setup_actor(self.get_mapper())
        return self._actor
    def get_mapper(self):
        if self._mapper is None:
            self._mapper = self._get_mapper()
        return self._mapper

    @staticmethod
    def _HTMLColorToRGB(colorString):
        '''
        Convert #RRGGBB to a [R, G, B] list.
        :param: colorString a string in the form: #RRGGBB where RR, GG, BB are hexadecimal.
        The elements of the array rgb are unsigned chars (0..255).
        :return: The red, green and blue components as a list.
        '''
        colorString = colorString.strip()
        if colorString[0] == '#': colorString = colorString[1:]
        if len(colorString) != 6:
            raise ValueError("Input #%s is not in #RRGGBB format" % colorString)
        r, g, b = colorString[:2], colorString[2:4], colorString[4:]
        r, g, b = [int(n, 16) for n in (r, g, b)]

        return [r/255, g/255, b/255]

    def set_color(self, c):
        import vtk

        if isinstance(c, str) and c.startswith("#"):
            c = self._HTMLColorToRGB(c)
        else:
            colors = vtk.vtkNamedColors()
            c = colors.GetColor3d(c)
        self.actor.GetProperty().SetColor(c)

    def plot(self, window):
        window.add_object(self)
        return [self]

class VTKGeometricPrimitive(VTKPrimitive):
    def __init__(self, source, name, **opts):
        self.source = source
        super().__init__(self._get_mapper, name, **opts)
        self.name = name

    def _get_mapper(self):
        return self._setup_mapper(self.source)

class VTKDisk(VTKGeometricPrimitive):
    def __init__(self, pos, rad, **opts):
        import vtk

        super().__init__(vtk.vtkDiskSource(), "Disk", **opts)

        src = self.source
        # src.SetCenter(*pos)
        src.SetRadius(rad)

class VTKLine(VTKGeometricPrimitive):
    def __init__(self, pt1, pt2, **opts):
        import vtk

        super().__init__(vtk.vtkLineSource(), "Line", **opts)

        self.source.SetPoint1(*pt1)
        self.source.SetPoint2(*pt2)


class VTKCylinder(VTKGeometricPrimitive):
    def __init__(self, pt1, pt2, rad, cylinder_points = 24, **opts):
        import vtk

        line = vtk.vtkLineSource()
        line.SetPoint1(*pt1)
        line.SetPoint2(*pt2)

        tf = vtk.vtkTubeFilter()
        tf.SetInputConnection(line.GetOutputPort())
        tf.SetRadius(rad)
        tf.SetNumberOfSides(cylinder_points)
        tf.Update()
        super().__init__(tf, "Cylinder", **opts)

class VTKSphere(VTKGeometricPrimitive):
    def __init__(self, pos, rad, **opts):
        import vtk

        super().__init__(vtk.vtkSphereSource(), "Sphere", **opts)

        sphereSource = self.source
        sphereSource.SetCenter(*pos)
        sphereSource.SetRadius(rad)
        # Make the surface smooth.
        sphereSource.SetPhiResolution(100)
        sphereSource.SetThetaResolution(100)

