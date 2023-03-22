import vtk
import numpy as np
import gmsh
import math
import os

class Film:
    def __init__(self, nodes_coords, trs_points):
        self.nodes = np.array([nodes_coords[0::3],nodes_coords[1::3],nodes_coords[2::3]])
        self.trs = np.array([trs_points[0::3],trs_points[1::3],trs_points[2::3]])
        self.trs -= 1
        self.dipols = np.array([[0,0,1] for i in range(len(nodes_coords))])

    def move(self, tau):
        self.nodes += self.velocity * tau

    def snapshot(self, snap_number):
        unstructuredGrid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()

        # vel = vtk.vtkDoubleArray()
        # vel.SetNumberOfComponents(3)
        # vel.SetName("vel")

        for i in range(0, len(self.nodes[0])):
            points.InsertNextPoint(self.nodes[0,i], self.nodes[1,i], self.nodes[2,i])
            # smth.InsertNextValue(self.smth[i])
            # vel.InsertNextTuple((self.velocity[0,i], self.velocity[1,i], self.velocity[2,i]))

        unstructuredGrid.SetPoints(points)
        # unstructuredGrid.GetPointData().AddArray(vel)

        for i in range(0, len(self.trs[0])):
            tr = vtk.vtkTriangle()
            for j in range(0, 3):
                tr.GetPointIds().SetId(j, self.trs[j,i])
            unstructuredGrid.InsertNextCell(tr.GetCellType(), tr.GetPointIds())

        # Создаём снапшот в файле с заданным именем
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputDataObject(unstructuredGrid)
        writer.SetFileName("" + str(snap_number) + ".vtu")
        writer.Write()

gmsh.initialize()
gmsh.model.add("film")
size = 1
n = 10
lc = size/n
gmsh.model.geo.addPoint(size/2, size/2, 0, lc, 1)
gmsh.model.geo.addPoint(size/2, -size/2, 0, lc, 2)
gmsh.model.geo.addPoint(-size/2, size/2, 0, lc, 3)
gmsh.model.geo.addPoint(-size/2, -size/2, 0, lc, 4)
gmsh.model.geo.addLine(1,2,1)
gmsh.model.geo.addLine(2,4,2)
gmsh.model.geo.addLine(4,3,3)
gmsh.model.geo.addLine(3,1,4)
gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

nodeTags, nodesCoord, parametricCoord = gmsh.model.mesh.getNodes()
triNodesTags = None
elementTypes, elementTags, elementNodeTags = gmsh.model.mesh.getElements()
for i in range(0, len(elementTypes)):
    if elementTypes[i] == 2: #находит треугольники
        triNodesTags = elementNodeTags[i]

if triNodesTags is None:
    print("Can not find triangles data. Exiting.")
    gmsh.finalize()
    exit(-2)

print("The model has %d nodes and %d triangles" % (len(nodeTags), len(triNodesTags) / 3))

film = Film(nodesCoord, triNodesTags)

gmsh.finalize()