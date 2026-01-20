# Header Files Directory

This directory contains the header files that define the data structures, function prototypes, and core computation kernels for the stencil computation project.

## Main Header

### `stencil.h`
Primary header file for the hybrid MPI+OpenMP parallel implementation. This header provides:

**Data Structures:**
- `vec2_t`: 2D vector type for coordinates and dimensions
- `plane_t`: 2D grid structure for storing temperature values
- `buffers_t`: Communication buffers for MPI halo exchange
- MPI neighbor and communication tag definitions

**Core Functions:**
- **Energy Injection**: `inject_energy()` - Adds energy at source locations
- **Stencil Computation**:
  - `update_plane()` - Simple full-grid update (for serial/OpenMP-only)
  - `update_interior()` - Updates interior points (MPI+OpenMP with overlap)
  - `update_borders()` - Updates border points after halo exchange
- **Energy Computation**: `get_total_energy()` - Parallel sum of grid values
- **Initialization**: Domain decomposition, neighbor discovery, memory allocation
- **Communication**: Halo packing/unpacking, MPI send/receive operations

**Key Design Features:**
- **Communication-Computation Overlap**: Split update strategy (interior/borders) enables overlapping MPI communication with OpenMP computation
- **OpenMP Parallelization**: All computation loops use `#pragma omp parallel for` with static scheduling
- **Thread Safety**: OpenMP reductions for energy summation, no data races in stencil loops
- **MPI Integration**: Non-blocking communication (MPI_Isend/Irecv) with proper synchronization

**Boundary Conditions:**
- **Non-Periodic (Fixed)**: Grid edges maintain constant values, local communication only
- **Periodic (Wrap-around)**: Grid edges connect in toroidal topology, requires additional communication paths

## Template Headers

### `stencil_template_parallel.h`
Header template for the parallel MPI+OpenMP implementation. Contains the basic structure and function prototypes, suitable as a starting point for understanding or implementing parallel stencil computation.

### `stencil_template_serial.h`
Header template for serial implementation. Provides basic data structures and function prototypes for single-node stencil computation without MPI.

## Usage

Include the appropriate header in your source files:
```c
#include "stencil.h"              // For parallel MPI+OpenMP implementation
#include "stencil_template_serial.h"  // For serial template
```

The headers automatically handle MPI and OpenMP includes, so no additional includes are needed for those libraries.

## Implementation Strategy

The header design follows a two-strategy approach for stencil computation:

1. **Simple Strategy** (serial/OpenMP-only):
   - `update_plane()` updates all grid points at once
   - No communication-computation overlap

2. **Split Strategy** (MPI+OpenMP with overlap):
   - `update_interior()` called BEFORE `MPI_Waitall()` (while halo data transfers)
   - `update_borders()` called AFTER `MPI_Waitall()` (using fresh halo data)
   - Enables hiding communication latency behind computation

This design maximizes performance on distributed systems by overlapping communication and computation phases.
