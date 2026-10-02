# Tpetra / Trilinos Agent Instructions

## Repo layout
- This is the **Tpetra** package of the **Trilinos** project (version 17.2.0-dev).
- Parent Trilinos repo: `/projects/Trilinos/`.
- Build directory: `/projects/build/` (CMake cache, Makefile, test outputs).
- Source root: `/projects/tpetra/`.
- `core/src/` — main implementation (Tpetra_*.hpp, Tpetra_*.cpp, `Tpetra_Details_*`).
- `core/test/` — tests, one subdirectory per component (CrsMatrix, MultiVector, ImportExport, etc.).
- `core/example/` — example programs. `Finite-Element-Assembly/` is the active example dir.
- `core/compat/` — backward-compatibility layer (Epetra adapter, classic Node wrapper).
- `core/inout/` — I/O helpers (MatrixMarket, TEKO).
- `core/ext/` — external integrations.
- `tsqr/` — TSQR subpackage, separate from `core/`.
- `scripts/PerformanceTesting/` — performance testing scripts.

## Build system
- **TriBITS** (not vanilla CMake). The top-level `CMakeLists.txt` calls `TRIBITS_PACKAGE_DECL`, `TRIBITS_ADD_OPTION_AND_DEFINE`, etc. Do not assume standard CMake conventions.
- Build cache lives in `/projects/build/CMakeCache.txt`. Reconfigure with `cmake ..` from `/projects/build/`.
- Kokkos version is pinned: `Tpetra_SUPPORTED_KOKKOS_VERSION "5.2.0"` in root `CMakeLists.txt:27`. External Kokkos must match unless `Tpetra_IGNORE_KOKKOS_COMPATIBILITY=ON`.

## Execution model (must know)
- Tpetra is **hybrid parallel**: MPI (distributed) + Kokkos (shared-memory).
- Supported Kokkos back-ends: Serial, OpenMP, Threads, CUDA, HIP, SYCL.
- Kokkos execution spaces are selected via CMake options like `Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_CUDA`.
- Kokkos allows OpenMP and Threads together, but **Tpetra forbids both `Tpetra_INST_OPENMP` and `Tpetra_INST_PTHREAD` being ON simultaneously** — the build will error out.
- GPU-aware MPI: detected via `ompi_info` or set explicitly with `Tpetra_ASSUME_GPU_AWARE_MPI`. When in doubt, set it OFF.
- Runtime override: env var `TPETRA_ASSUME_GPU_AWARE_MPI=ON|OFF`.

## Explicit Template Instantiation (ETI)
- ETI controls which template parameter combinations are **pre-instantiated** at build time.
- Macros: `HAVE_TPETRA_INST_<TYPE>` (instantiation + tests when ETI ON; tests only when ETI OFF).
- Template parameters: scalar types (DOUBLE, FLOAT, COMPLEX_DOUBLE, etc.), ordinal pairs (INT_INT, INT_LONG_LONG), Node types (SERIAL, OPENMP, CUDA, etc.).
- Downstream code must check `HAVE_TPETRA_INST_*` availability macros, not assume types are available.
- If you encounter link errors involving Tpetra templates, the most common cause is a missing ETI type, not a bug in the code.

## Code style
- All files are formatted with `.clang-format` (clang-format 14).

## File naming conventions
- **`Tpetra_<Name>_decl.hpp`** — class declarations only (forward declarations and type signatures).
- **`Tpetra_<Name>_def.hpp`** — inline definitions of member functions declared in `_decl.hpp`.
- **`Tpetra_<Name>.hpp`** — CMake-generated convenience header that includes `_decl.hpp`. Do not put implementation in these.
- **`Tpetra_<Name>_fwd.hpp`** — forward declaration only.
- Implementation (non-inline): `Tpetra_<Name>.cpp`.
- Private helpers: `Tpetra_Details_<Name>.hpp` (in `core/src/`).
- ETI instantiation files: `Tpetra_ETI_*.tmpl`.
- Tests in `core/test/<ComponentName>/`.

## Testing
- Tests are registered via `TRIBITS_ADD_TEST_DIRECTORIES(test)` in `core/CMakeLists.txt:200`.
- Test definitions live in subdirectories under `core/test/`, each with its own `CMakeLists.txt`.
- Some tests are performance-oriented (`BasicPerfTest/`, `PerformanceCGSolve/`).
- XML test suites: `core/test/Tpetra_PerformanceTests.xml`.
- Testing utilities: `core/test/Tpetra_TestingUtilities.hpp`, `Tpetra_TestingXMLUtilities.hpp`.

## Important constraints
- `Tpetra_INST_FLOAT` requires `Teuchos_ENABLE_FLOAT=ON` AND a BLAS with float (S) support.
- `Tpetra_INST_COMPLEX_DOUBLE` requires `Teuchos_ENABLE_COMPLEX=ON` AND complex BLAS.
- `Tpetra_INST_COMPLEX_DOUBLE` also requires `Tpetra_INST_DOUBLE=ON`.
- The "classic" Node type (`Tpetra::KokkosCompat::KokkosSerialClassicNodeAPI`) must be OFF; the refactor Node type is used.
- `Tpetra_INST_PTHREAD` and `Tpetra_INST_OPENMP` cannot both be ON.
- When Kokkos execution space is enabled but the corresponding `Tpetra_INST_*` option is OFF, you will get **link errors** at runtime.

## Fundamental classes
- **`Map`** — Defines the distribution of data across MPI ranks. A Map maps global indices to local indices on each rank, and represents the domain/codomain of vectors and matrices.
- **`DistObject`** — Abstract base for distributed data redistribution. Objects that implement it can exchange data between overlapping Maps via Import/Export.
- **`MultiVector`** — Dense vector object with one or multiple columns, with the rows distributed across MPI ranks. Supports operations like dot products, norms, scaling, and imports.
- **`CrsGraph`** — Compressed-row sparse structure defining a sparsity pattern only. Stores row/col indices and offsets without values.
- **`CrsMatrix`** — Compressed-row sparse matrix with values. Uses a `CrsGraph` to define its sparsity pattern, and stores the matrix values alongside it. Can be constructed in several ways. One creates an empty matrix with a certain number of entries allocated for each row, where `insertGlobalEntries` can insert them. Another takes a set of Maps and a local `KokkosSparse::CrsMatrix` defining both the sparsity pattern and values.

