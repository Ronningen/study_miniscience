#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <XDMFFile.h>
#include <utility>
#include <vector>
#include "poisson.h"

using namespace dolfinx;
using T = PetscScalar;

int main(int argc, char* argv[])
{
    dolfinx::init_logging(argc, argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    auto file = dolfinx::io::XDMFFile(MPI_COMM_WORLD, "Solutions/trash/fish.xdmf", "r");
    const auto mesh = std::make_shared<mesh::Mesh>(
        file.read_mesh(fem::CoordinateElement(mesh::CellType::tetrahedron, 1), mesh::GhostMode::shared_facet, "fish"));
    auto cell_markers = file.read_meshtags(mesh, "fish_cells");
    mesh->topology().create_connectivity(3, 2);
    auto facet_markers = file.read_meshtags(mesh, "fish_facets");
    file.close();

    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));
    auto kappa = std::make_shared<fem::Constant<T>>(1.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto g = std::make_shared<fem::Function<T>>(V);

    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}));

    auto facets = mesh::locate_entities_boundary(
        *mesh, 2,
        [](auto x)
        {
          constexpr double eps = 1.0e-1;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            double z0 = x(2, p);
            marker[p] = (std::abs(z0+0.9) < eps or std::abs(z0 - 0.9) < eps);
          }
          return marker;
        });
    const auto bdofs = fem::locate_dofs_topological({*V}, 2, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(1.0, bdofs, V);

    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(pow(2, -(x(0, p)*x(0, p) + x(1, p)*x(1, p))));

          return {f, {f.size()}};
        });
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            double dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            double dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            double dz = (x(2, p) - 0.5) * (x(2, p) - 0.5);
            f.push_back(10 * std::exp(-(dx + dy + dz) / 0.02));
          }

          return {f, {f.size()}};
        });

    fem::Function<T> u(V);
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         *a, {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                         {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting(b.mutable_array(), {a}, {{bc}}, {}, 1.0);
    b.scatter_rev(std::plus<T>());
    fem::set_bc(b.mutable_array(), {bc});

    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "preonly");
    la::petsc::options::set("pc_type", "lu");
    lu.set_from_options();

    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u.x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    lu.solve(_u.vec(), _b.vec());

    io::VTKFile file_vtk(MPI_COMM_WORLD, "u.pvd", "w");
    file_vtk.write<T>({u}, 0.0);

    PetscFinalize();
    return 0;
}