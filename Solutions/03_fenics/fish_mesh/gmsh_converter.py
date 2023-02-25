import gmsh

from dolfinx.io import XDMFFile, gmshio

from mpi4py import MPI

gmsh.initialize()

gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.2)

model = gmsh.model()
model.add("fish")
model.setCurrent("fish")

# gmsh.merge("Solutions/trash/01_task_mesh.msh")

body = model.occ.addSphere(0, 0, 0, 1, tag=1)
model.occ.affineTransform([[3, body]], [1.3,0,0,0, 0,0.5,0,0, 0,0,0.9,0])
left = model.occ.addSphere(0, 0.4, -0.2, 0.4, tag=2)
right = model.occ.addSphere(0, -0.4, -0.2, 0.4, tag=3)
model.occ.affineTransform([[3, left],[3, right]], [1,0,0,0, 0,2,0,0, 0,0,0.3,0])
model.occ.rotate([[3, left]],0,0,0, 1,0,0, -0.5)
model.occ.rotate([[3, right]],0,0,0, 1,0,0, +0.5)

model.occ.synchronize()
model.add_physical_group(dim=3, tags=[body, left, right])
model.mesh.generate(dim=3)

msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
msh.name = "fish"
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

with XDMFFile(msh.comm, f"Solutions/trash/fish.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_meshtags(cell_markers)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    file.write_meshtags(facet_markers)