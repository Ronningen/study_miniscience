#include <set>
#include <cmath>
#include <gmsh.h>

int main(int argc, char **argv)
{
    gmsh::initialize();

    gmsh::model::add("stl");

    gmsh::merge("fish.stl");

    gmsh::model::mesh::classifySurfaces(1, true, true);
    gmsh::model::mesh::createGeometry();

    gmsh::vectorpair entities;
    gmsh::model::getEntities(entities, 2);
    std::vector<int> surfs;
    for (auto surf : entities)
        surfs.push_back(surf.second);
    gmsh::model::geo::addVolume({gmsh::model::geo::addSurfaceLoop(surfs)});

    gmsh::model::geo::synchronize();
    // gmsh::option::setNumber("Mesh.MeshSizeFactor", 0.2);
    gmsh::option::setNumber("Mesh.CharacteristicLengthFromCurvature", 0.5);
    gmsh::option::setNumber("Mesh.AngleToleranceFacetOverlap", 0.01);
    gmsh::option::setNumber("Mesh.MinimumElementsPerTwoPi", 3);
    gmsh::model::mesh::generate(3);

    gmsh::write("../trash/01_task_mesh.msh");

    std::set<std::string> args(argv, argv + argc);
    if (!args.count("-nopopup"))
        gmsh::fltk::run();
    gmsh::finalize();
}