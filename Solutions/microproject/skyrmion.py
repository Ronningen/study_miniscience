import vtk
import numpy as np
import gmsh
from numba import njit
import sympy


Size = 1         # geometrical size of film
Triangles_N = 40 # estimated amaunt of triangles in a row
Lc = Size/Triangles_N 
K = 1e-5         # magnitude of interaction


# набор оптимизированных numba математических функций
class optimise:
    @njit
    # векторизованная нормировка 3-вектора
    def normolize(a):
        norms = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
        return a / norms
    
    @njit
    # векторизованное смешнное произведение 3-векторов
    def mixed(a,b,c):
        d=\
        a[0]*(b[1]*c[2]-b[2]*c[1])+\
        a[1]*(b[2]*c[0]-b[0]*c[2])+\
        a[2]*(b[0]*c[1]-b[1]*c[0]) 
        return d

    @njit
    # векторизованное векторное произведение 3-векторов
    def cross(b, c):
        a = np.zeros(shape=(3, len(b[0])), dtype=np.double)
        a[0]=(b[1]*c[2]-b[2]*c[1])
        a[1]=(b[2]*c[0]-b[0]*c[2])
        a[2]=(b[0]*c[1]-b[1]*c[0]) 
        return a

    # @njit
    def density(n, film_neighbors, dipols):
        dx_moment = np.zeros(shape=(3, n), dtype=np.double)
        dy_moment = np.zeros(shape=(3, n), dtype=np.double)
        for i in range(n):
            n = len(film_neighbors[i])
            for neighbor in film_neighbors[i]:
                j = int(neighbor[2])
                dmx = dipols[0][j] - dipols[0][i]
                dmy = dipols[1][j] - dipols[1][i]
                dmz = dipols[2][j] - dipols[2][i]
                dx_moment[0][i] += dmx / neighbor[0] / n
                dx_moment[1][i] += dmy / neighbor[0] / n
                dx_moment[2][i] += dmz / neighbor[0] / n
                dy_moment[0][i] += dmx / neighbor[1] / n
                dy_moment[1][i] += dmy / neighbor[1] / n
                dy_moment[2][i] += dmz / neighbor[1] / n
        return optimise.mixed(dipols, dx_moment, dy_moment)

    # @njit
    def field(nodes, dipols, has_outer_field=False, interact=True, outer_field=[0]):
        n = len(nodes[0])
        field = np.zeros(shape=(3, n), dtype=np.double)
        if has_outer_field:
            field += outer_field
        if interact:
            for i in range(n):
                R = np.zeros(shape=(3, n), dtype=np.double)
                m = np.zeros(n, dtype=np.double)
                for ax in range(3):
                    R[ax] = nodes[ax,i] - nodes[ax]
                    m += R[ax] * dipols[ax]

                Rn = np.linalg.norm(R, axis=0)
                for ax in range(3):
                    field[ax,i] = np.sum(np.divide( (3*R[ax]*m - dipols[ax]*np.power(Rn, 2)), 
                                                    (np.power(Rn, 5)), out = np.zeros(n, dtype=np.double), where = Rn != 0))
        return field

# модельный класс - тонкий слой магнитных диполей
class Film:
    def __init__(self, nodes_coords, trs_points):
        self.N = int(len(nodes_coords) / 3)
        self.nodes = np.array([nodes_coords[0::3],nodes_coords[1::3],nodes_coords[2::3]])
        self.trs = np.array([trs_points[0::3],trs_points[1::3],trs_points[2::3]])
        self.trs -= 1
        self.dipol_moment = np.zeros(shape=(3, self.N), dtype=np.double)
        self.dipol_moment[2] += np.ones(self.N, dtype=np.double)
        self.density = np.zeros(self.N, dtype=np.double)
        self.bc_mask = np.ones(shape=(3, self.N), dtype=np.double)

        # нахождение соседей для рассчета частных пространственных производных
        self.neighbors = []
        for i in range(self.N):
            neighbors = []
            for j in range(self.N):
                dx = self.nodes[0,i] - self.nodes[0,j]
                dy = self.nodes[1,i] - self.nodes[1,j]
                if i != j and dx != 0 and dy != 0 and (dx)**2 + (dy)**2 < 3 * Lc**2:
                    neighbors.append([dx, dy, j])
            self.neighbors.append(neighbors)
        print("film inited")

    # дискретно фиксирует квадратный слой на Size/2 границе толщиной width
    def fix_square_bc(self, width=Size/100):
        self.bc_mask = np.array([ not (
                self.nodes[0,i] < -Size/2+width or self.nodes[0,i] > Size/2 - width or 
                self.nodes[1,i] < -Size/2+width or self.nodes[1,i] > Size/2 - width) for i in range(self.N)]).reshape(-1,1).T

    # дискретно фиксирует круглый слой на Size/2 границе толщиной width
    def fix_circle_bc(self, width=Size/10):
        self.bc_mask = np.array([ not (
                np.power(self.nodes[0,i],2) + np.power(self.nodes[1,i], 2) >= (Size/2-width)**2) for i in range(self.N)]).reshape(-1,1).T

    # функция, постепенно увеличвающая "вязкость" к бесконечности к круглой Size/2 границе с шириной width
    def fix_slight_circle_bc(self, width=Size/100):
        pass
        #self.bc_mask = np.exp( - (np.power(self.nodes[0],2) + np.power(self.nodes[1], 2))/(Size - width)**2)

    # нормирует длину магнитных моментов
    def normolize(self):
        self.dipol_moment = optimise.normolize(self.dipol_moment)

    # обновляет топологическую плотность
    def update_density(self):
        self.density = optimise.density(self.N, self.neighbors, self.dipol_moment)

    # делает шаг эволюции методом рунге-кутты c нормализация по окончанию
    # TODO: изменить галилевы приращения на сферические - сильно увеличит точность и не надо нормализовать, но чуть больше вычислений
    def move(self, h, has_outer_field=False, interact=True, outer_field=[0]):
        def f(y):
            return K * optimise.cross(y, optimise.field(self.nodes, y, has_outer_field, interact, outer_field)) * self.bc_mask
        y = self.dipol_moment
        k1 = f(y)
        k2 = f(y + h/2*k1)
        k3 = f(y + h/2*k2)
        k4 = f(y + h*k3)
        self.dipol_moment += h/6*(k1 + 2*k2 + 2*k3 + k4)
        self.normolize()

    # сохраняет снапщот для vtk
    def snapshot(self, snap_number):
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        moments = vtk.vtkDoubleArray()
        moments.SetNumberOfComponents(3)
        moments.SetName("moments")
        density = vtk.vtkDoubleArray()
        density.SetName("topological density")

        for i in range(0, len(self.nodes[0])):
            points.InsertNextPoint(self.nodes[0,i], self.nodes[1,i], self.nodes[2,i])
            moments.InsertNextTuple((self.dipol_moment[0,i], self.dipol_moment[1,i], self.dipol_moment[2,i]))
            density.InsertNextValue(self.density[i])

        unstructuredGrid.SetPoints(points)
        unstructuredGrid.GetPointData().AddArray(moments)
        unstructuredGrid.GetPointData().AddArray(density)

        for i in range(0, len(self.trs[0])):
            tr = vtk.vtkTriangle()
            for j in range(0, 3):
                tr.GetPointIds().SetId(j, self.trs[j,i])
            unstructuredGrid.InsertNextCell(tr.GetCellType(), tr.GetPointIds())

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputDataObject(unstructuredGrid)
        writer.SetFileName("/Users/samedi/Documents/прога/study_miniscience/Solutions/trash/moments_" + str(snap_number) + ".vtu")
        writer.Write()

    # возвращает скирминное число с некоторым постоянным коэфциентом
    def skyrmions(self):
        return np.sum(self.density)

# набор функций взаимодействия с gmsh
class mesher:
    def create_square_mesh():
        gmsh.model.geo.addPoint(Size/2, Size/2, 0, Lc, 1)
        gmsh.model.geo.addPoint(Size/2, -Size/2, 0, Lc, 2)
        gmsh.model.geo.addPoint(-Size/2, Size/2, 0, Lc, 3)
        gmsh.model.geo.addPoint(-Size/2, -Size/2, 0, Lc, 4)
        gmsh.model.geo.addLine(1,2,1)
        gmsh.model.geo.addLine(2,4,2)
        gmsh.model.geo.addLine(4,3,3)
        gmsh.model.geo.addLine(3,1,4)
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

    def create_circle_mesh():
        gmsh.model.geo.addPoint(0,0,0,Lc,0)
        gmsh.model.geo.addPoint(Size/2, 0, 0, Lc, 1)
        gmsh.model.geo.addPoint(0, -Size/2, 0, Lc, 2)
        gmsh.model.geo.addPoint(-Size/2, 0, 0, Lc, 3)
        gmsh.model.geo.addPoint(0, Size/2, 0, Lc, 4)
        gmsh.model.geo.addCircleArc(1, 0, 2, 1)
        gmsh.model.geo.addCircleArc(2, 0, 3, 2)
        gmsh.model.geo.addCircleArc(3, 0, 4, 3)
        gmsh.model.geo.addCircleArc(4, 0, 1, 4)
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

    def parse_mesh():
        nodeTags, nodesCoord, parametricCoord = gmsh.model.mesh.getNodes()
        triNodesTags = None
        elementTypes, elementTags, elementNodeTags = gmsh.model.mesh.getElements()
        for i in range(0, len(elementTypes)):
            if elementTypes[i] == 2: # находит треугольники
                triNodesTags = elementNodeTags[i]
        if triNodesTags is None:
            print("Can not find triangles data. Exiting.")
            gmsh.finalize()
            exit(-2)
        return nodesCoord, triNodesTags


def set_test_spiral(coords):
    dipol_moment = np.zeros(shape=(3, len(coords[0])), dtype=np.double)
    dipol_moment[1] += (0.03 - np.power(coords[0],2) - np.power(coords[1],2))
    # возмущение для вывода из нейстойчивого равновесия
    np.random.seed(0)
    dipol_moment[0] += (np.random.rand(film.N)-0.5)*Size/1000
    dipol_moment[1] += (np.random.rand(film.N)-0.5)*Size/1000
    return dipol_moment

def set_single_skyrmion(coords, x=0, y=0, w=Size/10):
    dipol_moment = np.zeros(shape=(3, len(coords[0])), dtype=np.double)
    a = 1/w
    d2 = (np.power(coords[0]-x,2) + np.power(coords[1]-y,2))*a**2
    dipol_moment[0] = 2*(film.nodes[0]-x)/(1+d2)*a
    dipol_moment[1] = 2*(film.nodes[1]-y)/(1+d2)*a
    dipol_moment[2] = 2*(1-d2)/(1+d2)
    return dipol_moment


# создание меша и его загрузка в модель
gmsh.initialize()
gmsh.model.add("film")
mesher.create_circle_mesh()
film = Film(*mesher.parse_mesh())
gmsh.finalize()

# Подготовка модели и начальные условия
# film.fix_slight_circle_bc()
film.dipol_moment = set_single_skyrmion(film.nodes, 0.1, 0.1, 0.1)
film.normolize()
film.update_density()
B = np.zeros(shape=(3, film.N), dtype=np.double)
B[2] += -np.ones(film.N, dtype=np.double)*1e5

# Просмотр начального состояния
skN1 = film.skyrmions()
film.snapshot(0)

# Параметры симулции и записи
n = 100         # кол-во кадров
dt = 0.01       # разрешение симуляции во времени
time = 1.5      # полное время симуляции
if (time < dt*n): time = dt*n
snapshorate = int(time/n/dt)

# симуляция
for i in range(int(time/dt)): 
    film.move(dt, interact=True, has_outer_field=False, outer_field=B)
    film.update_density()
    if i%snapshorate == snapshorate-1:
        isn = int((i+1)/snapshorate)
        film.snapshot(isn)
        print(str(int(isn/n*100))+"%")

# Оценка ошибки вычислений по скирмионному числу
skN2 = film.skyrmions()
print("error", 1-skN1/skN2)