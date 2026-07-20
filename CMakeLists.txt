cmake_minimum_required(VERSION 3.18)
project(tensor_bond_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- libtorch (the C++ distribution of PyTorch) -----------------------------
# Point CMake at your unzipped libtorch, e.g.:
#   cmake -DCMAKE_PREFIX_PATH=/abs/path/to/libtorch -B build -S .
# Or set the Torch_DIR / CMAKE_PREFIX_PATH in your IDE's CMake profile.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- core library -----------------------------------------------------------
# Layout: include/tbp/{core,instruments,io} with a mirrored src/ tree.
#   core        curve math (tensor curve, bootstraps) + schedule helpers
#   instruments fixed-rate bond, Treasury FRN
#   io          CSV loaders for the data/curves cache (std-only, no torch)
add_library(tbp
    src/core/curve.cpp
    src/core/curve_store.cpp
    src/instruments/bond.cpp
    src/instruments/frn.cpp
    src/io/csv.cpp
)
target_include_directories(tbp PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tbp PUBLIC ${TORCH_LIBRARIES})

# --- demo executable --------------------------------------------------------
add_executable(price_bonds apps/price_bonds.cpp)
target_link_libraries(price_bonds PRIVATE tbp)

# --- CUSIP pricer: NPV / schedule / accrued / quote check by CUSIP -----------
add_executable(price_cusip apps/price_cusip.cpp)
target_link_libraries(price_cusip PRIVATE tbp)

# --- tests ------------------------------------------------------------------
enable_testing()
add_executable(test_pricing tests/test_pricing.cpp)
target_link_libraries(test_pricing PRIVATE tbp)
add_test(NAME pricing COMMAND test_pricing)

# On Windows, copy the libtorch DLLs next to the binaries.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET price_bonds POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
        $<TARGET_FILE_DIR:price_bonds>)
endif()
