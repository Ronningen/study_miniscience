#include <gmsh.h>
#include <cmath>
#include <iostream>
#include <set>

int main(int argc, char **argv)
{
    gmsh::initialize();
    gmsh::model::add("torus");

    double lc = 0.1, r1 = 0.5, r2 = 1.2;
    gmsh::model::geo::addPoint(r2 + r1, 0, 0, lc, 1);
    gmsh::model::geo::addPoint(r2 - r1, 0, 0, lc, 3);
    gmsh::model::geo::addPoint(r2, r1, 0, lc, 2);
    gmsh::model::geo::addPoint(r2, -r1, 0, lc, 4);
    auto center = gmsh::model::geo::addPoint(r2, 0, 0, lc);

    std::vector<int> surfs;
    for (int i = 1; i <= 4; i++)
    {
        auto tag = gmsh::model::geo::addCircleArc(i, center, i % 4 + 1);
        gmsh::vectorpair wall;
        gmsh::model::geo::revolve({{1, tag}}, 0, 0, 0, 0, 1, 0, M_PI / 3 * 2, wall);
        surfs.push_back(wall[0].second);
        for (int j = 1; j <= 2; j++)
        {
            gmsh::vectorpair wall_copy;
            gmsh::model::geo::copy(wall, wall_copy);
            gmsh::model::geo::rotate(wall_copy, 0, 0, 0, 0, 1, 0, M_PI / 3 * 2 * j);
            surfs.push_back(wall_copy[0].second);
        }
    }
    gmsh::model::geo::addSurfaceLoop(surfs, 1);
    gmsh::model::geo::synchronize();

    gmsh::vectorpair entities;
    gmsh::model::getEntities(entities);
    gmsh::vectorpair vol;
    gmsh::model::geo::extrudeBoundaryLayer(entities, vol, {1,1,1,1,1}, {0.01, 0.02, 0.05, 0.06, 0.07}, true);

    gmsh::model::geo::synchronize();
    gmsh::model::mesh::generate(3);

    std::set<std::string> args(argv, argv + argc);
    if (!args.count("-nopopup"))
        gmsh::fltk::run();
    gmsh::finalize();
}