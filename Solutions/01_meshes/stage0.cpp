#include <gmsh.h>
#include <iostream>
#include <set>

int main(int argc, char **argv)
{
    int subtask;
    if (argc <= 1)
    {
        std::cout << "Enter subtask number: ";
        std::cin >> subtask;
    }
    else
        subtask = atoi(argv[1]);
    

    double size = 0.5;
    int N = 2;
    if (argc > 2)
        N = atof(argv[2]);
    double lc = size/N;


    gmsh::initialize();
    gmsh::model::add("s0");

    switch (subtask)
    {
        case 1:
        {
            gmsh::model::geo::addPoint(size, size, size, lc, 1);
            gmsh::model::geo::addPoint(size, size, -size, lc, 2);
            gmsh::model::geo::addPoint(size, -size, size, lc, 4);
            gmsh::model::geo::addPoint(size, -size, -size, lc, 3);
            gmsh::model::geo::addPoint(-size, size, size, lc, 5);
            gmsh::model::geo::addPoint(-size, size, -size, lc, 6);
            gmsh::model::geo::addPoint(-size, -size, size, lc, 8);
            gmsh::model::geo::addPoint(-size, -size, -size, lc, 7);

            for (int i = 1; i <= 4; i++)
                gmsh::model::geo::addLine(i, i % 4 + 1, i);
            for (int i = 5; i <= 8; i++)
                gmsh::model::geo::addLine(i, i % 4 + 5, i);
            for (int i = 9; i <= 12; i++)
                gmsh::model::geo::addLine(i - 8, i - 4, i);

            gmsh::model::geo::addCurveLoop({1, 2, 3, 4}, 1);
            gmsh::model::geo::addCurveLoop({5, 6, 7, 8}, 6);
            for (int i = 1; i <= 4; i++)
                gmsh::model::geo::addCurveLoop({-i, i + 8, i + 4, -(i % 4 + 9)}, i + 1);

            for (int i = 1; i <= 6; i++)
                gmsh::model::geo::addPlaneSurface({i}, i);

            gmsh::model::geo::addSurfaceLoop({1, 2, 3, 4, 5, 6}, 1);
            gmsh::model::geo::addVolume({1});

            gmsh::model::geo::synchronize();
            gmsh::model::mesh::generate(3);
            break;
        }
        case 2:
        {
            gmsh::model::geo::addPoint(size,0,0,lc,1);
            gmsh::model::geo::addPoint(-size,0,0,lc,2);
            auto center = gmsh::model::geo::addPoint(0,0,0,lc);

            gmsh::model::geo::addCircleArc(1, center, 2, 1);
            gmsh::model::geo::addCircleArc(2, center, 1, 2);
            gmsh::model::geo::addCurveLoop({1,2}, 1);
            gmsh::model::geo::addPlaneSurface({1});

            gmsh::model::geo::synchronize();
            gmsh::model::mesh::generate(2);
            break;
        }
    case 3:
        {
            gmsh::model::geo::addPoint(size,0,0,lc,1);
            gmsh::model::geo::addPoint(-size,0,0,lc,3);
            gmsh::model::geo::addPoint(0,size,0,lc,2);
            gmsh::model::geo::addPoint(0,-size,0,lc,4);
            auto center = gmsh::model::geo::addPoint(0,0,0,lc);

            gmsh::model::geo::addCircleArc(1, center, 2, 1);
            gmsh::model::geo::addCircleArc(2, center, 3, 2);
            gmsh::model::geo::addCircleArc(3, center, 4, 3);
            gmsh::model::geo::addCircleArc(4, center, 1, 4);
            gmsh::model::geo::addCurveLoop({1,2,3,4}, 1);
            gmsh::model::geo::addPlaneSurface({1}, 1);

            gmsh::vectorpair vol; 
            gmsh::model::geo::extrude({{2,1}}, 0, 0, size, vol, {}, {}, false);

            gmsh::model::geo::synchronize();
            gmsh::model::mesh::generate(3);
            break;
        }
    default:
        {
            std::cout << "incorrect option" << std::endl;
            gmsh::finalize();
            return 0;
        }
    }

    std::set<std::string> args(argv, argv + argc);
    if (!args.count("-nopopup"))
        gmsh::fltk::run();
    gmsh::finalize();
}