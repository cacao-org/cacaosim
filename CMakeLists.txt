# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.18)
project(milkcacaosim VERSION 1.0.0 LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CACAOSIM_ENABLE_CUDA "Enable CUDA GPU backend (cuFFT + cuBLAS + CUDA Graphs)" ON)
option(CACAOSIM_NATIVE_ARCH "Enable -march=native CPU SIMD optimization" ON)

set(LIBNAME "milkcacaosim")
set(SRCNAME "cacaosim")

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
find_library(FFTW3F_OMP_LIB NAMES fftw3f_omp HINTS ${FFTW3F_LIBRARY_DIRS})

pkg_check_modules(OPENBLAS QUIET openblas)
if(OPENBLAS_FOUND)
    set(CBLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIRS})
    set(CBLAS_LIBRARIES ${OPENBLAS_LINK_LIBRARIES})
    if(NOT CBLAS_LIBRARIES)
        set(CBLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
    endif()
else()
    find_path(CBLAS_INCLUDE_DIRS NAMES cblas.h HINTS /usr/include /usr/include/x86_64-linux-gnu)
    find_library(CBLAS_LIBRARIES NAMES openblas cblas blas REQUIRED)
endif()

set(SOURCEFILES
    src/milkcacaosim.c
    src/common/cacaosim_rng.c
    src/common/cacaosim_detector.c
    src/common/cacaosim_beam.c
    src/common/cacaosim_cuda_stub.c
    src/pupil/cacaosim_pupil.c
    src/pupil/cacaosim_mkpupil_FPS.c
    src/beam/cacaosim_mkbeam_FPS.c
    src/beam/cacaosim_beamsplit_FPS.c
    src/dm/cacaosim_dm_engine.c
    src/dm/cacaosim_dmsim_FPS.c
    src/coro/cacaosim_coro_engine.c
    src/coro/cacaosim_coro_FPS.c
    src/pyrwfs/cacaosim_pyrwfs_engine.c
    src/pyrwfs/cacaosim_pyrwfs_FPS.c
    src/shwfs/cacaosim_shwfs_engine.c
    src/shwfs/cacaosim_shwfs_FPS.c
    src/zwfs/cacaosim_zwfs_engine.c
    src/zwfs/cacaosim_zwfs_FPS.c
    src/fpcam/cacaosim_fpcam_engine.c
    src/fpcam/cacaosim_fpcam_FPS.c
    src/bench/cacaosim_bench_cfg.c
    src/bench/cacaosim_bench_edit.c
    src/bench/cacaosim_bench_dag.c
    src/bench/cacaosim_bench_view.c
    src/bench/cacaosim_tui_ansi.c
    src/bench/cacaosim_tui_shm.c
    src/bench/cacaosim_tui_params.c
    src/bench/cacaosim_tui_render.c
    src/bench/cacaosim_tui_views.c
    src/bench/cacaosim_tui.c
    src/fused/cacaosim_htsim_FPS.c
)

set(INCLUDEFILES
    src/milkcacaosim.h
    src/cacaosim_types.h
    src/common/cacaosim_time.h
    src/common/cacaosim_rng.h
    src/common/cacaosim_detector.h
    src/common/cacaosim_beam.h
    src/common/cacaosim_cuda.h
    src/pupil/cacaosim_pupil.h
    src/pupil/cacaosim_mkpupil_FPS.h
    src/beam/cacaosim_mkbeam_FPS.h
    src/beam/cacaosim_beamsplit_FPS.h
    src/dm/cacaosim_dm.h
    src/dm/cacaosim_dmsim_FPS.h
    src/coro/cacaosim_coro.h
    src/coro/cacaosim_coro_FPS.h
    src/pyrwfs/cacaosim_pyrwfs.h
    src/pyrwfs/cacaosim_pyrwfs_FPS.h
    src/shwfs/cacaosim_shwfs.h
    src/shwfs/cacaosim_shwfs_FPS.h
    src/zwfs/cacaosim_zwfs.h
    src/zwfs/cacaosim_zwfs_FPS.h
    src/fpcam/cacaosim_fpcam.h
    src/fpcam/cacaosim_fpcam_FPS.h
    src/bench/cacaosim_bench.h
    src/bench/cacaosim_bench_view.h
    src/bench/cacaosim_tui.h
    src/fused/cacaosim_htsim_FPS.h
)

set(SCRIPTS
    scripts/cacaosim-bench
)

set(USE_CUDA_ACCEL OFF)
if(CACAOSIM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
                set(CMAKE_CUDA_ARCHITECTURES "native")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
            endif()
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(USE_CUDA_ACCEL ON)
            add_compile_definitions(HAVE_CUDA)
            list(APPEND SOURCEFILES
                src/common/cacaosim_cuda.cu
                src/dm/cacaosim_dm_cuda.cu
                src/coro/cacaosim_coro_cuda.cu
                src/pyrwfs/cacaosim_pyrwfs_cuda.cu
                src/shwfs/cacaosim_shwfs_cuda.cu
                src/zwfs/cacaosim_zwfs_cuda.cu
                src/fpcam/cacaosim_fpcam_cuda.cu
            )
            message(STATUS "milkcacaosim: CUDA GPU acceleration ENABLED (${CMAKE_CUDA_COMPILER})")
        endif()
    endif()
endif()

set(CACAOSIM_C_OPT_FLAGS -O3 -fno-math-errno -Wall -Wextra -Wno-stringop-truncation)
if(CACAOSIM_NATIVE_ARCH)
    list(APPEND CACAOSIM_C_OPT_FLAGS -march=native)
endif()

set(CACAOSIM_MATH_LIBS
    OpenMP::OpenMP_C
    ${FFTW3F_LINK_LIBRARIES}
    ${CBLAS_LIBRARIES}
    m
)
if(NOT FFTW3F_LINK_LIBRARIES)
    list(APPEND CACAOSIM_MATH_LIBS ${FFTW3F_LIBRARIES})
endif()
if(FFTW3F_OMP_LIB)
    list(APPEND CACAOSIM_MATH_LIBS ${FFTW3F_OMP_LIB})
endif()

if(COMMAND milk_add_runnable_module)
    # ── In-tree build inside milk (feat/bash-engine) ─────────────────────────
    milk_add_runnable_module(${LIBNAME}
        SOURCES ${SOURCEFILES}
        LINK_LIBS
            milkCOREMODiofits
            ${CACAOSIM_MATH_LIBS}
    )
    milk_runnable_module_usage(${LIBNAME}
        INCLUDES
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            ${FFTW3F_INCLUDE_DIRS}
            ${CBLAS_INCLUDE_DIRS}
            $<INSTALL_INTERFACE:include>
            $<INSTALL_INTERFACE:include/${SRCNAME}>
    )
    if(USE_CUDA_ACCEL)
        target_link_libraries(${LIBNAME} PRIVATE CUDA::cudart CUDA::cufft CUDA::cublas)
    endif()
    milk_add_fpsexec_stubs(${LIBNAME})

    add_executable(cacaosim-bench-view src/bench/cacaosim_view_main.c)
    target_compile_definitions(cacaosim-bench-view PRIVATE MILK_NO_CLI)
    target_compile_options(cacaosim-bench-view PRIVATE ${CACAOSIM_C_OPT_FLAGS})
    target_include_directories(cacaosim-bench-view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cacaosim-bench-view PRIVATE ${LIBNAME} milkfps m)
    install(TARGETS cacaosim-bench-view DESTINATION bin)

    add_executable(cacaosim-CTRL src/bench/cacaosim_ctrl_main.c)
    target_compile_definitions(cacaosim-CTRL PRIVATE MILK_NO_CLI)
    target_compile_options(cacaosim-CTRL PRIVATE ${CACAOSIM_C_OPT_FLAGS})
    target_include_directories(cacaosim-CTRL PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cacaosim-CTRL PRIVATE ${LIBNAME} milkfps m)
    install(TARGETS cacaosim-CTRL DESTINATION bin)

    install(DIRECTORY src/ DESTINATION include/${SRCNAME} FILES_MATCHING PATTERN "*.h")
    install(
        DIRECTORY examples/
        DESTINATION share/${SRCNAME}/examples
        FILES_MATCHING PATTERN "*.bench"
    )
    install(PROGRAMS ${SCRIPTS} DESTINATION bin)
else()
    # ── Out-of-tree standalone build against milk (feat/bash-engine) ─────────
    set(MILK_SOURCE_CANDIDATES
        ${MILK_SOURCE_DIR}
        $ENV{MILK_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../milk
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src-frameworkdev/milk
        $ENV{MILK_ROOT}
        /usr/local/milk
    )
    set(MILK_SOURCE_FOUND "")
    foreach(cand ${MILK_SOURCE_CANDIDATES})
        if(cand AND EXISTS "${cand}/src/coremods")
            get_filename_component(MILK_SOURCE_FOUND "${cand}" ABSOLUTE)
            message(STATUS "Found MILK source tree: ${MILK_SOURCE_FOUND}")
            break()
        endif()
    endforeach()

    set(_root_cands ${MILK_ROOT} $ENV{MILK_ROOT} $ENV{MILK_INSTALLDIR})
    if(MILK_SOURCE_FOUND)
        file(GLOB _ver_inst "${MILK_SOURCE_FOUND}/_install/milk-*")
        list(SORT _ver_inst ORDER DESCENDING)
        list(APPEND _root_cands ${_ver_inst} "${MILK_SOURCE_FOUND}/_install")
    endif()
    list(APPEND _root_cands ${CMAKE_INSTALL_PREFIX} /usr/local/milk /usr/local)
    foreach(rcand ${_root_cands})
        if(rcand AND EXISTS "${rcand}/include/libcomputeunit/milk_runnable.h")
            get_filename_component(MILK_ROOT "${rcand}" ABSOLUTE)
            break()
        endif()
    endforeach()

    if(NOT MILK_ROOT)
        message(FATAL_ERROR "Could not locate milk (feat/bash-engine) install with milk_runnable.h")
    endif()

    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${MILK_ROOT}" CACHE PATH "Install prefix" FORCE)
    endif()
    message(STATUS "milkcacaosim: MILK_ROOT=${MILK_ROOT}, INSTALL=${CMAKE_INSTALL_PREFIX}")

    set(CMAKE_INSTALL_RPATH
        "$ORIGIN;$ORIGIN/../lib;${CMAKE_INSTALL_PREFIX}/lib;${MILK_ROOT}/lib;/usr/local/milk/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    set(MILK_STANDALONE_INCLUDES
        ${MILK_ROOT}/include
        ${MILK_ROOT}/include/CommandLineInterface
        ${MILK_ROOT}/include/CommandLineInterface/CLIcore
        ${MILK_ROOT}/include/libmilkcommon
        ${MILK_ROOT}/include/libmilkdata
        ${MILK_ROOT}/include/libfps
        ${MILK_ROOT}/include/libprocessinfo
        ${MILK_ROOT}/include/libcomputeunit
        ${MILK_ROOT}/include/ImageStreamIO
    )

    if(MILK_SOURCE_FOUND)
        list(APPEND MILK_STANDALONE_INCLUDES
            ${MILK_SOURCE_FOUND}/src
            ${MILK_SOURCE_FOUND}/src/coremods
            ${MILK_SOURCE_FOUND}/src/cli
            ${MILK_SOURCE_FOUND}/src/cli/CLIcore
            ${MILK_SOURCE_FOUND}/src/engine
            ${MILK_SOURCE_FOUND}/src/engine/libfps
            ${MILK_SOURCE_FOUND}/src/engine/libmilkcommon
            ${MILK_SOURCE_FOUND}/src/engine/libprocessinfo
            ${MILK_SOURCE_FOUND}/src/engine/libmilkdata
            ${MILK_SOURCE_FOUND}/src/engine/libcomputeunit
            ${MILK_SOURCE_FOUND}/src/engine/ImageStreamIO
        )
        if(EXISTS "${MILK_SOURCE_FOUND}/_build/config")
            list(APPEND MILK_STANDALONE_INCLUDES "${MILK_SOURCE_FOUND}/_build/config")
        endif()
    endif()

    set(MILK_LIB_NAMES
        milkcomputeunit
        milkfps
        milkprocessinfo
        milkdata
        ImageStreamIO
        milkCOREMODmemory
        milkCOREMODarith
        milkCOREMODiofits
        milkCOREMODtools
    )

    set(MILK_SEARCH_HINTS ${MILK_ROOT}/lib ${CMAKE_INSTALL_PREFIX}/lib)
    if(MILK_SOURCE_FOUND)
        list(APPEND MILK_SEARCH_HINTS ${MILK_SOURCE_FOUND}/_build/lib)
    endif()
    list(APPEND MILK_SEARCH_HINTS /usr/local/milk/lib /usr/local/lib)

    set(MILK_LIBRARIES "")
    foreach(lib ${MILK_LIB_NAMES})
        find_library(MILK_${lib}_LIB NAMES ${lib} HINTS ${MILK_SEARCH_HINTS} NO_DEFAULT_PATH)
        if(NOT MILK_${lib}_LIB)
            find_library(MILK_${lib}_LIB NAMES ${lib} HINTS ${MILK_SEARCH_HINTS})
        endif()
        if(MILK_${lib}_LIB)
            list(APPEND MILK_LIBRARIES ${MILK_${lib}_LIB})
        else()
            list(APPEND MILK_LIBRARIES ${lib})
        endif()
    endforeach()

    add_library(${LIBNAME} SHARED ${SOURCEFILES})
    set_target_properties(${LIBNAME} PROPERTIES
        LINKER_LANGUAGE C
        INTERPROCEDURAL_OPTIMIZATION OFF
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_INSTALL_RPATH}"
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
    )
    target_compile_definitions(${LIBNAME} PRIVATE MILK_NO_CLI)
    target_compile_options(${LIBNAME} PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${CACAOSIM_C_OPT_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
    target_include_directories(${LIBNAME} SYSTEM PRIVATE
        ${MILK_STANDALONE_INCLUDES}
        ${FFTW3F_INCLUDE_DIRS}
        ${CBLAS_INCLUDE_DIRS}
    )
    target_include_directories(${LIBNAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/${SRCNAME}>
    )
    target_link_libraries(${LIBNAME} PUBLIC
        ${MILK_LIBRARIES}
        ${CACAOSIM_MATH_LIBS}
        rt
        pthread
    )
    if(USE_CUDA_ACCEL)
        target_link_libraries(${LIBNAME} PRIVATE CUDA::cudart CUDA::cufft CUDA::cublas)
    endif()

    install(TARGETS ${LIBNAME} DESTINATION lib)
    install(DIRECTORY src/ DESTINATION include/${SRCNAME} FILES_MATCHING PATTERN "*.h")
    install(
        DIRECTORY examples/
        DESTINATION share/${SRCNAME}/examples
        FILES_MATCHING PATTERN "*.bench"
    )
    install(PROGRAMS ${SCRIPTS} DESTINATION bin)

    # ── Generate standalone milk-fpsexec-cacaosim-* executables (no CLIcore) ─
    set(_FPSEXEC_STUB_TPL [=[
#include "milk_skinny_runnable.h"

extern const MILK_RUNNABLE @UNIT@;

int main(int argc, char *argv[])
{
    MILK_EMBED_BUILD_TAG();
    return MILK_FPSEXEC_MAIN(argc, argv, &@UNIT@);
}
]=])

    string(CONCAT _FPSEXEC_RE
        "EXPORT_FPSEXEC[ \t]*\\([ \t]*([A-Za-z0-9_]+)[ \t]*,[ \t]*"
        "\"([^\"]+)\"[ \t]*,[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\\)"
    )

    foreach(_rel_src IN LISTS SOURCEFILES)
        set(_abs_src "${CMAKE_CURRENT_SOURCE_DIR}/${_rel_src}")
        if(NOT EXISTS "${_abs_src}")
            continue()
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_abs_src}")
        file(READ "${_abs_src}" _src_text)
        string(REGEX MATCHALL "(^|\n)[ \t]*EXPORT_FPSEXEC[ \t]*\\([^\n]*\\)"
               _fpsexec_hits "${_src_text}")
        foreach(_hit IN LISTS _fpsexec_hits)
            if(_hit MATCHES "${_FPSEXEC_RE}")
                set(_exe "${CMAKE_MATCH_1}-fpsexec-${CMAKE_MATCH_2}")
                set(_unit "${CMAKE_MATCH_3}")
                string(REPLACE "@UNIT@" "${_unit}" _stub_src "${_FPSEXEC_STUB_TPL}")
                set(_stub_file "${CMAKE_CURRENT_BINARY_DIR}/fpsexec_stubs/${_exe}.c")
                file(GENERATE OUTPUT "${_stub_file}" CONTENT "${_stub_src}")
                add_executable(${_exe} "${_stub_file}")
                target_compile_definitions(${_exe} PRIVATE
                    MILK_NO_CLI MILK_BUILD_BINNAME="${_exe}"
                )
                target_include_directories(${_exe} SYSTEM PRIVATE ${MILK_STANDALONE_INCLUDES})
                target_link_libraries(${_exe} PRIVATE
                    ${LIBNAME}
                    ${MILK_LIBRARIES}
                    m
                    rt
                    pthread
                    -Wl,--allow-shlib-undefined
                )
                set_target_properties(${_exe} PROPERTIES
                    BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_INSTALL_RPATH}"
                    INSTALL_RPATH "$ORIGIN/../lib;${CMAKE_INSTALL_RPATH}"
                )
                install(TARGETS ${_exe} DESTINATION bin)
            endif()
        endforeach()
    endforeach()

    # Refresh milk module catalog after install when milk-cli is present
    install(CODE "
        set(_milk_cli \"\${CMAKE_INSTALL_PREFIX}/bin/milk-cli\")
        if(EXISTS \"\${_milk_cli}\")
            message(STATUS \"Refreshing milk module catalog via \${_milk_cli} --catalog\")
            execute_process(COMMAND \"\${_milk_cli}\" --catalog RESULT_VARIABLE _cat_rc)
        endif()
    ")

    # ── Standalone Visualizer, Validation & Benchmark Executables ────────────
    add_executable(cacaosim-bench-view src/bench/cacaosim_view_main.c)
    target_compile_definitions(cacaosim-bench-view PRIVATE MILK_NO_CLI)
    target_compile_options(cacaosim-bench-view PRIVATE ${CACAOSIM_C_OPT_FLAGS})
    target_include_directories(cacaosim-bench-view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(cacaosim-bench-view SYSTEM PRIVATE ${MILK_STANDALONE_INCLUDES})
    target_link_libraries(cacaosim-bench-view PRIVATE
        ${LIBNAME}
        ${MILK_LIBRARIES}
        ${CACAOSIM_MATH_LIBS}
    )
    set_target_properties(cacaosim-bench-view PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_INSTALL_RPATH}"
        INSTALL_RPATH "$ORIGIN/../lib;${CMAKE_INSTALL_RPATH}"
    )
    install(TARGETS cacaosim-bench-view DESTINATION bin)

    add_executable(cacaosim-CTRL src/bench/cacaosim_ctrl_main.c)
    target_compile_definitions(cacaosim-CTRL PRIVATE MILK_NO_CLI)
    target_compile_options(cacaosim-CTRL PRIVATE ${CACAOSIM_C_OPT_FLAGS})
    target_include_directories(cacaosim-CTRL PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(cacaosim-CTRL SYSTEM PRIVATE ${MILK_STANDALONE_INCLUDES})
    target_link_libraries(cacaosim-CTRL PRIVATE
        ${LIBNAME}
        ${MILK_LIBRARIES}
        ${CACAOSIM_MATH_LIBS}
    )
    set_target_properties(cacaosim-CTRL PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_INSTALL_RPATH}"
        INSTALL_RPATH "$ORIGIN/../lib;${CMAKE_INSTALL_RPATH}"
    )
    install(TARGETS cacaosim-CTRL DESTINATION bin)

    enable_testing()

    add_executable(cacaosim-validate tests/cacaosim_validate.c)
    target_compile_definitions(cacaosim-validate PRIVATE MILK_NO_CLI)
    target_compile_options(cacaosim-validate PRIVATE ${CACAOSIM_C_OPT_FLAGS})
    target_include_directories(cacaosim-validate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(cacaosim-validate SYSTEM PRIVATE
        ${MILK_STANDALONE_INCLUDES}
        ${CBLAS_INCLUDE_DIRS}
    )
    target_link_libraries(cacaosim-validate PRIVATE
        ${LIBNAME}
        ${MILK_LIBRARIES}
        ${CACAOSIM_MATH_LIBS}
    )
    set_target_properties(cacaosim-validate PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_INSTALL_RPATH}"
        INSTALL_RPATH "$ORIGIN/../lib;${CMAKE_INSTALL_RPATH}"
    )
    add_test(NAME cacaosim-validate COMMAND cacaosim-validate)
    add_test(
        NAME cacaosim-integration
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_closed_loop.sh"
                "${CMAKE_CURRENT_BINARY_DIR}" "${MILK_ROOT}"
    )
    string(CONCAT _CTEST_ENV
        "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{LD_LIBRARY_PATH};"
        "CACAOSIM_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(
        cacaosim-validate cacaosim-integration
        PROPERTIES ENVIRONMENT "${_CTEST_ENV}"
    )
    install(TARGETS cacaosim-validate DESTINATION bin)

    add_executable(cacaosim-benchmark tests/cacaosim_benchmark.c)
    target_compile_definitions(cacaosim-benchmark PRIVATE MILK_NO_CLI)
    target_compile_options(cacaosim-benchmark PRIVATE ${CACAOSIM_C_OPT_FLAGS})
    target_include_directories(cacaosim-benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(cacaosim-benchmark SYSTEM PRIVATE ${MILK_STANDALONE_INCLUDES})
    target_link_libraries(cacaosim-benchmark PRIVATE
        ${LIBNAME}
        ${MILK_LIBRARIES}
        ${CACAOSIM_MATH_LIBS}
    )
    set_target_properties(cacaosim-benchmark PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_INSTALL_RPATH}"
        INSTALL_RPATH "$ORIGIN/../lib;${CMAKE_INSTALL_RPATH}"
    )
    install(TARGETS cacaosim-benchmark DESTINATION bin)
endif()
