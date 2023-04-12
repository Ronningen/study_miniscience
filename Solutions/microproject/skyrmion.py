import vtk
import numpy as np
from numpy import cos, sin
import gmsh
from numba import njit
from datetime import datetime
import matplotlib.pyplot as plt


Size = 1          # geometrical size of film
Triangles_N = 55 # estimated amaunt of triangles in a row
Lc = Size/Triangles_N/2 
K = 1e-5          # magnitude of interaction


# набор оптимизированных numba математических функций
class optimise:
    @njit
    # векторизованная нормировка 3-вектора
    # вовзаращет кортеж из нормированныго вектора и его нормы
    def normolize(a):
        norms = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
        return a / norms, norms
    
    # @njit
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

    @njit
    # вычисляет магнитное поле как сумму внешнего outer_field при has_outer_field=True и
    #                                    полей диполей самой модели при interact=True
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
                R2 = R[0]*R[0] + R[1]*R[1] + R[2]*R[2]
                for ax in range(3):
                    for j in range(n):
                        if i != j: 
                            field[ax][i] += (3*R[ax][j]*m[j] - dipols[ax][j]*R2[j])/(np.power(R2[j], 2.5))
        return field

    @njit
    # поворачивает вектор v относительно оси n на угл a
    def rotate(v, n, a):
        V = np.empty(shape=(3, len(v[0])), dtype=np.double)
        c = cos(a)
        s = sin(a)
        c1 = 1 - c
        V[0] = ( n[0]**2  *c1 +      c )*v[0] + ( n[0]*n[1]*c1 + n[2]*s )*v[1] + ( n[0]*n[2]*c1 + n[1]*s )*v[2]
        V[1] = ( n[0]*n[1]*c1 - n[2]*s )*v[0] + ( n[1]**2  *c1 +      c )*v[1] + ( n[1]*n[2]*c1 - n[0]*s )*v[2]
        V[2] = ( n[0]*n[2]*c1 - n[1]*s )*v[0] + ( n[1]*n[2]*c1 + n[0]*s )*v[1] + ( n[2]**2  *c1 +      c )*v[2]
        return V

# модельный класс - тонкий слой магнитных диполей
class Film:
    def __init__(self, nodes_coords, trs_points, cut_to=Size/2):
        # self.trs = np.array([trs_points[0::3],trs_points[1::3],trs_points[2::3]])
        # self.trs -= 1

        n = int(len(nodes_coords) / 3)
        nodes = np.array([nodes_coords[0::3],nodes_coords[1::3],nodes_coords[2::3]]).T
        #регуляризация меша
        cut_nodes = []
        for i in range(n):
            if nodes[i][0]**2 + nodes[i][1]**2 < cut_to**2:
                cut_nodes.append(nodes[i])

        self.nodes = np.array(cut_nodes[1:]).T
        self.N = len(self.nodes[0])

        self.field = np.zeros(shape=(3, self.N), dtype=np.double)
        self.dipol_moment = np.zeros(shape=(3, self.N), dtype=np.double)
        self.dipol_moment[2] += np.ones(self.N, dtype=np.double)
        self.density = np.zeros(self.N, dtype=np.double)
        self.bc_mask = np.ones(shape=(3, self.N), dtype=np.double)
        self.energy_m = 0
        self.energy_s = 0

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

    # дискретно фиксирует квадратный слой на Size/2 границе толщиной width
    def fix_square_bc(self, width=Size/100):
        bc_mask = np.array([ float(not (
                self.nodes[0,i] < -Size/2+width or self.nodes[0,i] > Size/2 - width or 
                self.nodes[1,i] < -Size/2+width or self.nodes[1,i] > Size/2 - width)) for i in range(self.N)])
        self.bc_mask = np.array([bc_mask]*3)

    # дискретно фиксирует круглый слой на Size/2 границе толщиной width
    def fix_circle_bc(self, width=Size/10):
        bc_mask = np.array([ float(not (
                np.power(self.nodes[0,i],2) + np.power(self.nodes[1,i], 2) >= (Size/2-width)**2)) for i in range(self.N)])
        self.bc_mask = np.array([bc_mask]*3)

    # функция, постепенно увеличвающая "вязкость" к бесконечности к круглой Size/2 границе
    def fix_slight_circle_bc(self):
        r = np.sqrt(np.power(self.nodes[0], 2) + np.power(self.nodes[1], 2))
        bc_mask = 7.38905 * np.exp(1/(2*r/Size - 1) -1/(2*r/Size + 1), where=r<Size/2, out=np.zeros(self.N))
        self.bc_mask = np.array([bc_mask]*3)

    # нормирует длину магнитных моментов
    def normolize(self):
        self.dipol_moment = optimise.normolize(self.dipol_moment)[0]

    # обновляет топологическую плотность
    def update_density(self):
        dx_moment = np.zeros(shape=(self.N, 3), dtype=np.double)
        dy_moment = np.zeros(shape=(self.N, 3), dtype=np.double)
        for i in range(self.N):
            n = len(self.neighbors[i])
            for neighbor in self.neighbors[i]:
                dm = self.dipol_moment.T[neighbor[2]] - self.dipol_moment.T[i]
                dx_moment[i] += dm / neighbor[0] / n
                dy_moment[i] += dm / neighbor[1] / n
        self.density = optimise.mixed(self.dipol_moment, dx_moment.T, dy_moment.T)
        self.energy_s = np.sum(dx_moment**2 + dy_moment**2)

    # делает шаг эволюции методом рунге-кутты c нормализация по окончанию
    def move_rk(self, h, has_outer_field=False, interact=True, outer_field=[0]):
        self.field = optimise.field(self.nodes, self.dipol_moment, has_outer_field, interact, outer_field)
        def f(y):
            return K * optimise.cross(y, optimise.field(self.nodes, y, has_outer_field, interact, outer_field)) * self.bc_mask
        y = self.dipol_moment
        k1 = f(y)
        k2 = f(y + h/2*k1)
        k3 = f(y + h/2*k2)
        k4 = f(y + h*k3)
        self.dipol_moment += h/6*(k1 + 2*k2 + 2*k3 + k4)
        self.normolize()

    # делает шаг эволюции на сфере
    def move_s(self, t, has_outer_field=False, interact=True, outer_field=[0]):
        self.field = optimise.field(self.nodes, self.dipol_moment, has_outer_field, interact, outer_field)
        self.energy_m = -np.sum(np.sum(self.dipol_moment * self.field))
        n, A = optimise.normolize(self.field)
        self.dipol_moment = optimise.rotate(v=self.dipol_moment, 
                                            n=n,
                                            a=K*A*self.bc_mask[0]*t)

    # сохраняет снапщот для vtk
    def snapshot(self, snap_number):
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        moments = vtk.vtkDoubleArray()
        moments.SetNumberOfComponents(3)
        moments.SetName("moments")
        field = vtk.vtkDoubleArray()
        field.SetNumberOfComponents(3)
        field.SetName("field")
        density = vtk.vtkDoubleArray()
        density.SetName("topological density")

        for i in range(0, len(self.nodes[0])):
            points.InsertNextPoint(self.nodes[0,i], self.nodes[1,i], self.nodes[2,i])
            moments.InsertNextTuple((self.dipol_moment[0,i], self.dipol_moment[1,i], self.dipol_moment[2,i]))
            field.InsertNextTuple((self.field[0,i], self.field[1,i], self.field[2,i]))
            density.InsertNextValue(self.density[i])

        unstructuredGrid.SetPoints(points)
        unstructuredGrid.GetPointData().AddArray(moments)
        unstructuredGrid.GetPointData().AddArray(field)
        unstructuredGrid.GetPointData().AddArray(density)

        # for i in range(0, len(self.trs[0])):
        #     tr = vtk.vtkTriangle()
        #     for j in range(0, 3):
        #         tr.GetPointIds().SetId(j, self.trs[j,i])
        #     unstructuredGrid.InsertNextCell(tr.GetCellType(), tr.GetPointIds())

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputDataObject(unstructuredGrid)
        writer.SetFileName("/Users/samedi/Documents/прога/study_miniscience/Solutions/trash/moments_2_" + str(snap_number) + ".vtu")
        writer.Write()

    # возвращает скирминное число с некоторым постоянным коэфциентом
    def skyrmions(self):
        return np.sum(self.density)

# набор функций взаимодействия с gmsh
class mesher:
    def create_square_mesh(s=Size/2):
        gmsh.model.geo.addPoint(s, s, 0, Lc, 1)
        gmsh.model.geo.addPoint(s, -s, 0, Lc, 2)
        gmsh.model.geo.addPoint(-s, s, 0, Lc, 3)
        gmsh.model.geo.addPoint(-s, -s, 0, Lc, 4)
        gmsh.model.geo.addLine(1,2,1)
        gmsh.model.geo.addLine(2,4,2)
        gmsh.model.geo.addLine(4,3,3)
        gmsh.model.geo.addLine(3,1,4)
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

    def create_circle_mesh(s=Size):
        gmsh.model.geo.addPoint(0,0,0,Lc,0)
        gmsh.model.geo.addPoint(s, 0, 0, Lc, 1)
        gmsh.model.geo.addPoint(0, -s, 0, Lc, 2)
        gmsh.model.geo.addPoint(-s, 0, 0, Lc, 3)
        gmsh.model.geo.addPoint(0, s, 0, Lc, 4)
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


def skyrmion(coords, x=0, y=0, width=Size/10):
    dipol_moment = np.zeros(shape=(3, len(coords[0])), dtype=np.double)
    a = 1/width
    d2 = (np.power(coords[0]-x,2) + np.power(coords[1]-y,2))*a**2
    dipol_moment[0] = 2*(film.nodes[1]-y)/(1+d2)*a
    dipol_moment[1] = 2*(film.nodes[0]-x)/(1+d2)*a
    dipol_moment[2] = (1-d2)/(1+d2)
    return dipol_moment


# создание меша и его загрузка в модель
gmsh.initialize()
gmsh.model.add("film")
mesher.create_circle_mesh()
film = Film(*mesher.parse_mesh())
gmsh.finalize()
print("model is created")

# Подготовка модели и начальные условия
# film.fix_slight_circle_bc()
film.dipol_moment = skyrmion(film.nodes, 0, 0, 0.13)
# film.normolize()
film.update_density()
B = np.zeros(shape=(3, film.N), dtype=np.double)
B[2] += -np.ones(film.N, dtype=np.double)*1e5
print("model is ready")

# Просмотр начального состояния
skN1 = film.skyrmions()
film.snapshot(0)
Es = []
Em = []
Skn = []

# Параметры симулции и записи
n = 400    # кол-во кадров
dt = 0.001  # разрешение симуляции во времени
time = 1.2   # полное время симуляции
if (time < dt*n): time = dt*n
snapshorate = (time/n/dt)

# симуляция
print("simulation")
start_time = datetime.now()
for i in range(int(time/dt)): 
    film.move_s(dt, False, True, B)
    film.update_density()
    if i%snapshorate == snapshorate-1:
        isn = int((i+1)/snapshorate)
        film.snapshot(isn)
        print(str(int(isn/n*100))+"%")
    Es.append(film.energy_s)
    Em.append(film.energy_m)
    Skn.append(film.skyrmions())

# Оценка ошибки вычислений по скирмионному числу
skN2 = film.skyrmions()
print("estimated error: ", 1-skN1/skN2)
print(datetime.now() - start_time)

t = np.arange(len(Es))
plt.subplot(221)
plt.plot(t,Es)
plt.title("Skyrmion hamiltonian value")
plt.subplot(222)
plt.plot(t,Em)
plt.title("Classical dipols' energy")
plt.subplot(212)
plt.plot(t,Skn)
plt.xlabel("skyrmion number")
plt.show()