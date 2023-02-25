cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_FLAGS "-std=c++11 -g")
set(CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/qt@5/5.15.8_1/")

project(stage2)
add_executable(stage2 stage2.cpp)

# Найдём VTK
find_package(VTK REQUIRED)
# ... и подключим её со всеми её компонентами её нативными средствами
include(${VTK_USE_FILE})

# gmsh укажем явно по кусочкам (потому что он был просто распакован руками)
add_library(gmsh SHARED IMPORTED)
set_target_properties(gmsh PROPERTIES
    IMPORTED_LOCATION "/opt/homebrew/Cellar/gmsh/4.11.1/lib/libgmsh.4.11.1.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "/opt/homebrew/Cellar/gmsh/4.11.1/include/"
)
set(GMSH_LIBRARY gmsh)

# если бы gmsh был поставлен прилично, то можно было бы примерно вот так
#find_package(gmsh REQUIRED)
#include_directories(${GMSH_INCLUDE_DIR})
#target_link_libraries(tetr3d ${GMSH_LIBRARY})

# Укажем библиотеки для линковки
target_link_libraries(stage2 ${VTK_LIBRARIES})
target_link_libraries(stage2 ${GMSH_LIBRARY})
