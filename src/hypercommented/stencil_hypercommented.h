/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * =============================================================================
 * STENCIL.H - PARALLEL STENCIL COMPUTATION HEADER (MPI + OPENMP)
 * HYPERCOMMENTED VERSION - EDUCATIONAL PURPOSES
 * =============================================================================
 * 
 * This header file provides the core functions for parallel stencil computation
 * using a hybrid MPI+OpenMP approach.
 * 
 * =============================================================================
 * WHAT IS A STENCIL COMPUTATION?
 * =============================================================================
 * 
 * A stencil computation is a pattern where each element in an array is updated
 * based on its neighbors. In this case, we use a 5-point stencil:
 * 
 *            (i, j+1)
 *                |
 *    (i-1,j) - (i,j) - (i+1,j)
 *                |
 *            (i, j-1)
 * 
 * The new value at (i,j) depends on:
 *   - The current value at (i,j)
 *   - The four neighboring values (north, south, east, west)
 * 
 * This models heat diffusion: heat spreads from hot regions to cooler neighbors.
 * 
 * =============================================================================
 * FUNCTION OVERVIEW:
 * =============================================================================
 * 
 * ENERGY INJECTION:
 *   inject_energy() - Adds energy at source locations
 *     └─ Simulates heat sources (e.g., heaters) on the grid
 * 
 * STENCIL COMPUTATION (TWO STRATEGIES):
 *   
 *   STRATEGY 1: Simple (for serial/OpenMP-only):
 *     update_plane() - Updates ALL grid points at once
 *     ├─ Uses OpenMP to parallelize computation across threads
 *     └─ Simple but no communication-computation overlap
 *     └─ Good for shared memory (single node) systems
 *   
 *   STRATEGY 2: Split (for MPI+OpenMP with overlap):
 *     update_interior() - Updates INTERIOR points (can compute during MPI)
 *     └─ Called BEFORE MPI_Waitall
 *     └─ Interior points don't need data from other MPI processes
 *     
 *     update_borders() - Updates BORDER points (needs halo data)
 *     └─ Called AFTER MPI_Waitall
 *     └─ Border points need data from neighboring MPI processes
 *     
 *     This split enables COMMUNICATION-COMPUTATION OVERLAP:
 *       1. Start MPI_Isend/Irecv (non-blocking communication)
 *       2. Compute interior points (while MPI transfers halo data)
 *       3. MPI_Waitall (wait for halo data to arrive)
 *       4. Unpack halo data into ghost cells
 *       5. Compute borders (using fresh halo data from neighbors)
 *     
 *     This overlap significantly improves performance on clusters!
 * 
 * ENERGY COMPUTATION:
 *   get_total_energy() - Parallel sum of all grid values
 *     └─ Uses OpenMP reduction for thread-safe summation
 *     └─ Used to verify energy conservation
 * 
 * =============================================================================
 * OPENMP USAGE SUMMARY:
 * =============================================================================
 * 
 * OpenMP (Open Multi-Processing) enables shared-memory parallelism within
 * a single node. All threads can access the same memory space.
 * 
 * All stencil computation functions use OpenMP parallelization:
 *   - update_plane(): Full grid parallelized across threads
 *   - update_interior(): Interior points parallelized
 *   - update_borders(): Border points parallelized (2 separate loops)
 *   - get_total_energy(): Reduction for parallel sum
 * 
 * KEY PRAGMA: #pragma omp parallel for schedule(static)
 *   - Distributes loop iterations evenly across threads
 *   - Static scheduling: iterations assigned at compile time
 *   - Good for uniform workload and cache locality
 *   - Thread-safe: no data races (each thread writes to different locations)
 * 
 * WHY OPENMP?
 *   - Modern CPUs have multiple cores (8-64+ cores per socket)
 *   - Without parallelization, only 1 core is used (wasted resources!)
 *   - OpenMP allows us to use all cores efficiently
 *   - Near-linear speedup: 8 cores ≈ 8x faster
 * 
 * =============================================================================
 * MPI INTEGRATION:
 * =============================================================================
 * 
 * MPI (Message Passing Interface) enables distributed-memory parallelism
 * across multiple nodes in a cluster. Each process has its own memory.
 * 
 * The split update strategy (interior/borders) is designed for MPI:
 *   - Interior points: don't need halo data from neighbor processes
 *   - Border points: need halo data from neighbor processes
 * 
 * HALO EXCHANGE (GHOST CELLS):
 *   - Each process owns a subdomain of the full grid
 *   - Border points need values from neighboring subdomains
 *   - These values are stored in "ghost cells" (halo)
 *   - MPI sends/receives ghost cell data between neighbors
 * 
 * WHY SPLIT COMPUTATION?
 *   - Communication is SLOW (network latency, bandwidth limits)
 *   - Computation is FAST (modern CPUs are very fast)
 *   - By computing interior points DURING communication, we hide latency
 *   - This overlap is critical for strong scaling on large clusters
 * 
 * EXAMPLE TIMELINE:
 *   Without overlap:  |--MPI wait--|--compute all--|
 *   With overlap:     |--MPI wait--|
 *                       |--compute interior--|--compute borders--|
 *   
 *   Overlap saves time = (interior computation time)
 * 
 * =============================================================================
 * MEMORY LAYOUT AND INDEXING:
 * =============================================================================
 * 
 * The 2D grid is stored in a 1D array in row-major order.
 * 
 * LOGICAL GRID (what we think of):
 *   
 *   Ghost cells (halo):  [0][  1   2   3   4  ][xsize+1]
 *                        ---------------------------
 *                        [0][ 1 2 3 4 ... xsize ][x+1]  <- j=0 (ghost)
 *                        [0][ 1 2 3 4 ... xsize ][x+1]  <- j=1 (real)
 *                        [0][ 1 2 3 4 ... xsize ][x+1]  <- j=2 (real)
 *                         ...
 *                        [0][ 1 2 3 4 ... xsize ][x+1]  <- j=ysize (real)
 *                        [0][ 1 2 3 4 ... xsize ][x+1]  <- j=ysize+1 (ghost)
 *                        ---------------------------
 * 
 * PHYSICAL MEMORY (1D array):
 *   Index = j * (xsize+2) + i
 *   
 *   Why xsize+2? To accommodate ghost cells at i=0 and i=xsize+1
 *   Why ysize+2? To accommodate ghost cells at j=0 and j=ysize+1
 * 
 * INTERIOR vs BORDER points:
 *   - Interior: 2 <= i <= xsize-1, 2 <= j <= ysize-1
 *   - Borders:  i=1, i=xsize, j=1, j=ysize
 *   - Ghosts:   i=0, i=xsize+1, j=0, j=ysize+1
 * 
 * =============================================================================
 * PERIODIC BOUNDARY CONDITIONS:
 * =============================================================================
 * 
 * Periodic boundaries create a "wrap-around" effect, like a torus.
 * 
 * Without periodic BC:  [ 1 2 3 4 5 ]  (edges are fixed at 0)
 * With periodic BC:     [ 1 2 3 4 5 ]  (left edge connects to right edge)
 *                         ^         ^
 *                         |---------|
 * 
 * Implementation:
 *   - After updating interior points, copy boundary values to ghost cells
 *   - Ghost cell at i=0 copies from i=xsize (right edge wraps to left)
 *   - Ghost cell at i=xsize+1 copies from i=1 (left edge wraps to right)
 *   - Similarly for j direction
 * 
 * Physical meaning:
 *   - Models infinite periodic systems (crystals, repeating patterns)
 *   - Eliminates edge effects (no artificial boundaries)
 *   - Conserves total energy (no sinks at edges)
 * 
 * =============================================================================
 */

// =============================================================================
// STANDARD LIBRARY INCLUDES
// =============================================================================
// These provide essential functions for I/O, memory, math, etc.

#include <stdlib.h>   // malloc, free, exit
#include <stdio.h>    // printf, fprintf
#include <string.h>   // memcpy, memset
#include <unistd.h>   // getopt for command-line parsing
#include <getopt.h>   // getopt_long
#include <time.h>     // time measurement
#include <math.h>     // sqrt, fabs, etc.

// =============================================================================
// PARALLEL PROGRAMMING LIBRARIES
// =============================================================================

#include <omp.h>      // OpenMP for shared-memory parallelism (multi-threading)
#include <mpi.h>      // MPI for distributed-memory parallelism (multi-node)

// =============================================================================
// DIRECTIONAL CONSTANTS
// =============================================================================
// These define the four cardinal directions for halo exchange.
// Used to index arrays of MPI neighbors and communication buffers.

#define NORTH 0   // Neighbor in the +y direction (j increases)
#define SOUTH 1   // Neighbor in the -y direction (j decreases)
#define EAST  2   // Neighbor in the +x direction (i increases)
#define WEST  3   // Neighbor in the -x direction (i decreases)

// =============================================================================
// COMMUNICATION DIRECTION CONSTANTS
// =============================================================================
// Distinguish between sending and receiving buffers

#define SEND 0    // Buffer used for sending data to neighbors
#define RECV 1    // Buffer used for receiving data from neighbors

// =============================================================================
// TIME LEVEL CONSTANTS
// =============================================================================
// For time-stepping algorithms, we maintain two grid states:
//   OLD: the current state at time t
//   NEW: the next state at time t+dt

#define OLD 0     // Index for old/current time level
#define NEW 1     // Index for new/next time level

// =============================================================================
// COORDINATE INDICES
// =============================================================================
// For 2D vectors (size, position), we use these for clarity

#define _x_ 0     // Index for x-coordinate (columns)
#define _y_ 1     // Index for y-coordinate (rows)

// =============================================================================
// MPI MESSAGE TAGS
// =============================================================================
// Each direction gets a unique tag to avoid mixing messages

#define TAG_N 0   // Tag for messages from/to NORTH neighbor
#define TAG_S 1   // Tag for messages from/to SOUTH neighbor
#define TAG_E 2   // Tag for messages from/to EAST neighbor
#define TAG_W 3   // Tag for messages from/to WEST neighbor

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

// Unsigned integer type (shorthand for cleaner code)
typedef unsigned int uint;

// 2D vector type for grid sizes, positions, etc.
// Example: vec2_t size = {100, 200};  // 100 columns, 200 rows
typedef uint vec2_t[2];

// Communication buffers type
// buffers_t[SEND/RECV][NORTH/SOUTH/EAST/WEST]
// This stores 8 pointers: 4 send buffers (one per direction) and 4 recv buffers
typedef double *restrict buffers_t[2][4];

// Grid plane structure
// Stores a 2D grid of double-precision values and its size
typedef struct {
    double   * restrict data;  // Pointer to 1D array storing 2D grid
                              // 'restrict' keyword: compiler optimization hint
                              // (no aliasing - improves vectorization)
    vec2_t     size;          // Grid dimensions: size[_x_] = columns, size[_y_] = rows
                              // Note: actual allocated size is (size[_x_]+2) * (size[_y_]+2)
                              // to account for ghost cells
} plane_t;

// =============================================================================
// FUNCTION PROTOTYPES
// =============================================================================
// These declare functions that will be defined elsewhere (or inline below)

// -----------------------------------------------------------------------------
// inject_energy: Add energy at specified source locations
// -----------------------------------------------------------------------------
// Parameters:
//   periodic: if 1, apply periodic boundary conditions
//   Nsources: number of energy sources
//   Sources: array of source coordinates (x,y pairs)
//   energy: amount of energy to inject at each source
//   plane: grid to inject energy into
//   N: grid of MPI tasks (for periodic BC in parallel)
// Returns: 0 on success
extern int inject_energy ( const int      ,
                          const int      ,
			  const vec2_t  *,
			  const double   ,
                          plane_t *,
                          const vec2_t   );

// -----------------------------------------------------------------------------
// update_plane: Update all grid points using 5-point stencil
// -----------------------------------------------------------------------------
// Parameters:
//   periodic: if 1, apply periodic boundary conditions
//   N: grid of MPI tasks (for periodic BC in parallel)
//   oldplane: current grid state (time t)
//   newplane: output grid state (time t+dt)
// Returns: 0 on success
extern int update_plane ( const int      ,
                         const vec2_t   ,
                         const plane_t *,
                               plane_t * );

// -----------------------------------------------------------------------------
// update_interior: Update interior points only (for MPI overlap)
// -----------------------------------------------------------------------------
// Parameters:
//   oldplane: current grid state
//   newplane: output grid state
// Returns: 0 on success
extern int update_interior( const plane_t *,
                           plane_t * );

// -----------------------------------------------------------------------------
// update_borders: Update border points only (for MPI overlap)
// -----------------------------------------------------------------------------
// Parameters:
//   periodic: if 1, apply periodic boundary conditions
//   N: grid of MPI tasks (for periodic BC in parallel)
//   oldplane: current grid state (with updated halo)
//   newplane: output grid state
// Returns: 0 on success
extern int update_borders( const int      ,
                          const vec2_t   ,
                          const plane_t *,
                          plane_t * );

// -----------------------------------------------------------------------------
// get_total_energy: Compute total energy in the grid
// -----------------------------------------------------------------------------
// Parameters:
//   plane: grid to compute energy from
//   energy: output total energy (sum of all grid values)
// Returns: 0 on success
extern int get_total_energy( plane_t *,
                            double  * );

// -----------------------------------------------------------------------------
// initialize: Set up MPI, parse arguments, allocate memory
// -----------------------------------------------------------------------------
// This function is too complex to document in a prototype - see implementation
int initialize ( MPI_Comm *,
                int       ,
		int       ,
		int       ,
		char    **,
                vec2_t   *,
                vec2_t   *,                 
		int      *,
                int      *,
		int      *,
		int      *,
		int      *,
		int      *,
                vec2_t  **,
                double   *,
                plane_t  *,
                buffers_t *,
		int      * );

// -----------------------------------------------------------------------------
// memory_release: Free all dynamically allocated memory
// -----------------------------------------------------------------------------
int memory_release (plane_t   *, buffers_t * );

// -----------------------------------------------------------------------------
// output_energy_stat: Print energy statistics (for debugging/verification)
// -----------------------------------------------------------------------------
int output_energy_stat ( int      ,
                        plane_t *,
                        double   ,
                        int      ,
                        MPI_Comm *);

// =============================================================================
// INLINE FUNCTION IMPLEMENTATIONS
// =============================================================================
// Inline functions are defined in the header for performance.
// The compiler can inline (copy) the function code at call sites,
// avoiding function call overhead.

/*
 * =============================================================================
 * INJECT_ENERGY: Add energy at source locations
 * =============================================================================
 * 
 * This function simulates heat sources on the grid. At each source location,
 * we add a fixed amount of energy. This models heaters, flames, or other
 * heat sources in a physical system.
 * 
 * WHY DO WE NEED THIS?
 *   - Without energy injection, the system would reach uniform temperature
 *   - Energy injection creates gradients, driving heat diffusion
 *   - It's the "forcing term" in the heat equation
 * 
 * PERIODIC BOUNDARY CONDITIONS:
 *   - If a source is at the edge and we have periodic BC, we must also
 *     inject energy into the corresponding ghost cell
 *   - This ensures continuity across periodic boundaries
 * 
 * PARALLEL CONSIDERATIONS:
 *   - In MPI parallel code, only the process owning the source location
 *     calls this function
 *   - Sources at subdomain boundaries need special handling (handled elsewhere)
 */
inline int inject_energy ( const int      periodic,  // 1 if periodic BC, 0 otherwise
                          const int      Nsources,  // Number of energy sources
			  const vec2_t  *Sources,   // Array of source coordinates
			  const double   energy,    // Amount of energy to inject
                          plane_t *plane,           // Grid to inject into
                          const vec2_t   N          // MPI process grid dimensions
                          )
{
    // Extract grid dimensions for convenience
    const uint register sizex = plane->size[_x_]+2;  // Full width including ghost cells
    double * restrict data = plane->data;            // Pointer to grid data
    
    // Macro for converting 2D index (i,j) to 1D array index
    // This is the standard row-major layout: index = row * width + column
   #define IDX( i, j ) ( (j)*sizex + (i) )
   
    // Loop over all energy sources
    for (int s = 0; s < Nsources; s++)
        {
            // Extract source coordinates
            int x = Sources[s][_x_];  // Column index (0 to size[_x_]+1)
            int y = Sources[s][_y_];  // Row index (0 to size[_y_]+1)
            
            // Add energy at this source location
            // Note: += is used because we might have multiple sources at the same location
            data[ IDX(x,y) ] += energy;
            
            // Handle periodic boundary conditions
            if ( periodic )
                {
                    // If we only have 1 MPI process in x-direction (N[_x_] == 1),
                    // then this process handles the entire x-extent, so we need
                    // to propagate periodic boundaries ourselves
                    if ( N[_x_] == 1 )
                        {
                            // If source is at left edge (x=1), also inject at right ghost cell
                            // This creates wrap-around: left edge connects to right edge
                            if ( x == 1 )
                                data[ IDX(plane->size[_x_]+1, y) ] += energy;
                            
                            // If source is at right edge (x=xsize), also inject at left ghost cell
                            if ( x == plane->size[_x_] )
                                data[ IDX(0, y) ] += energy;
                        }
                    
                    // Similarly for y-direction
                    if ( N[_y_] == 1 )
                        {
                            // If source is at bottom edge (y=1), also inject at top ghost cell
                            if ( y == 1 )
                                data[ IDX(x, plane->size[_y_]+1) ] += energy;
                            
                            // If source is at top edge (y=ysize), also inject at bottom ghost cell
                            if ( y == plane->size[_y_] )
                                data[ IDX(x, 0) ] += energy;
                        }
                }                
        }
   #undef IDX  // Clean up macro (avoid polluting global namespace)
    
  return 0;  // Success
}

/*
 * =============================================================================
 * UPDATE_PLANE: Full 5-point stencil computation with OpenMP
 * =============================================================================
 * 
 * This is the heart of the heat diffusion simulation. It applies the 5-point
 * stencil formula to every grid point, computing the next time step.
 * 
 * THE STENCIL FORMULA:
 *   U_{i,j}^{n+1} = α·U_{i,j}^n + (1-α)/4·(U_{i-1,j} + U_{i+1,j} + U_{i,j-1} + U_{i,j+1})
 * 
 * WHERE:
 *   - U_{i,j}^n is the current value at grid point (i,j) at time n
 *   - U_{i,j}^{n+1} is the new value at the next time step
 *   - α (alpha) is a diffusion parameter (0 < α < 1)
 *   - The four neighbors contribute equally to the update
 * 
 * PHYSICAL INTERPRETATION:
 *   - α = 0.6 means 60% of energy stays at (i,j), 40% diffuses to neighbors
 *   - Each neighbor receives 10% (= 40%/4) of the diffused energy
 *   - This conserves total energy (sum of all grid values remains constant)
 * 
 * WHY THIS FORMULA?
 *   - It's a discretization of the heat equation: ∂u/∂t = κ∇²u
 *   - The stencil approximates the Laplacian (∇²u) using finite differences
 *   - α controls the time step implicitly (larger α = slower diffusion)
 * 
 * WHEN TO USE THIS FUNCTION:
 *   - Serial code (no MPI)
 *   - OpenMP-only parallel code (shared memory, single node)
 *   
 * WHEN NOT TO USE:
 *   - MPI parallel code should use update_interior + update_borders instead
 *   - This allows communication-computation overlap (much faster!)
 * 
 * OPENMP PARALLELIZATION:
 *   - The outer loop (j) is parallelized across threads
 *   - Each thread computes a subset of rows
 *   - No race conditions: each thread writes to different memory locations
 *   - Excellent speedup: typically 90-95% efficiency on multi-core CPUs
 */
inline int update_plane ( const int      periodic,   // 1 if periodic BC, 0 otherwise
                         const vec2_t   N,           // MPI process grid (for periodic BC)
                         const plane_t *oldplane,    // Current grid state (time t)
                               plane_t *newplane     // Output grid state (time t+dt)
                         )
    
{
    // Extract grid dimensions for convenience and compiler optimization
    // 'register' keyword: hint to compiler to keep in CPU register (faster access)
    uint register fxsize = oldplane->size[_x_]+2;  // Full width (with ghost cells)
    uint register fysize = oldplane->size[_y_]+2;  // Full height (with ghost cells)
    
    uint register xsize = oldplane->size[_x_];     // Interior width (without ghosts)
    uint register ysize = oldplane->size[_y_];     // Interior height (without ghosts)
    
    // Macro for 2D to 1D index conversion (row-major layout)
   #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    // Get pointers to old and new grid data
    // 'restrict' keyword: tells compiler these pointers don't alias (enables optimization)
    double * restrict old = oldplane->data;  // Read from old grid
    double * restrict new = newplane->data;  // Write to new grid
    
    // =============================================================
    // OPENMP PARALLELIZATION OF STENCIL COMPUTATION
    // =============================================================
    // #pragma omp parallel for schedule(static)
    //   - Parallelizes outer loop (j) across threads
    //   - Inner loop (i) remains sequential for better cache efficiency
    //   - Static scheduling: iterations divided evenly at compile time
    //     Example: 8 threads, 80 rows → each thread gets 10 consecutive rows
    //
    // WHY PARALLELIZE OUTER LOOP (j) NOT INNER LOOP (i)?
    //   - Better cache locality: each thread processes consecutive rows
    //   - Rows are stored contiguously in memory (row-major layout)
    //   - Reduces cache misses and false sharing between threads
    //
    // PARALLELIZATION CORRECTNESS:
    //   - Each (i,j) updated by exactly one thread (no race conditions)
    //   - All threads READ from old[] (shared, read-only) ✓
    //   - Each thread WRITES to different locations in new[] ✓
    //   - No synchronization needed inside loop (very efficient!)
    //
    // PERFORMANCE:
    //   - Row-major order matches C memory layout (cache-friendly)
    //   - Good spatial locality: consecutive accesses to nearby memory
    //   - Minimal synchronization overhead (only implicit barrier at end)
    //   - Typical speedup: 7-8x on 8 cores (90-100% parallel efficiency)
    //
    // CACHE EFFECTS:
    //   - Modern CPUs have 3 levels of cache: L1 (32 KB), L2 (256 KB), L3 (8 MB)
    //   - Sequential access pattern keeps data in cache
    //   - Each thread operates on ~ysize/num_threads rows, ideally fits in L2/L3
    // =============================================================
    
    // Diffusion parameters (alpha controls how much energy stays vs. diffuses)
    const double alpha = 0.6;                    // 60% stays at current location
    const double constant = (1.0 - alpha) / 4.0; // 10% goes to each neighbor
    
    #pragma omp parallel for schedule(static)
    for (uint j = 1; j <= ysize; j++)       // Loop over rows (excluding ghost cells)
        for ( uint i = 1; i <= xsize; i++)  // Loop over columns (excluding ghost cells)
            {
                // =============================================================
                // FIVE-POINT STENCIL FORMULA
                // =============================================================
                // 
                // Compute the new value at (i,j) based on:
                //   1. Current value at (i,j)
                //   2. Four neighbors: (i-1,j), (i+1,j), (i,j-1), (i,j+1)
                //
                // Mathematical form:
                //   new[i,j] = α·old[i,j] + β·(old[i-1,j] + old[i+1,j] + old[i,j-1] + old[i,j+1])
                //   where β = (1-α)/4
                //
                // Energy conservation:
                //   Total energy in = α + 4β = α + 4(1-α)/4 = α + 1 - α = 1 ✓
                //   So energy is conserved (no creation or destruction)
                //
                // Stability:
                //   For numerical stability, we need α ≥ 0.5 (CFL condition)
                //   α = 0.6 is a safe choice
                // =============================================================
                
                // Compute contribution from current location (60% stays here)
                double center = old[ IDX(i,j) ] * alpha;
                
                // Compute contribution from four neighbors (10% from each)
                double neighbors = ( old[IDX(i-1, j)]    // West neighbor
                                   + old[IDX(i+1, j)]    // East neighbor
                                   + old[IDX(i, j-1)]    // South neighbor
                                   + old[IDX(i, j+1)]    // North neighbor
                                   ) * constant;
                
                // Combine center and neighbor contributions
                double result = center + neighbors;
                
                // Write result to new grid
                // Note: We write to 'new', not 'old', so we don't overwrite input data
                new[ IDX(i,j) ] = result;
            }

    // =============================================================
    // PERIODIC BOUNDARY CONDITIONS
    // =============================================================
    // After computing interior points, we need to update ghost cells
    // for periodic boundaries. Ghost cells at one edge should match
    // real values at the opposite edge.
    //
    // This creates a "wrap-around" effect, like a torus:
    //   - Left edge (i=1) wraps to right edge (i=xsize)
    //   - Right edge (i=xsize) wraps to left edge (i=1)
    //   - Bottom edge (j=1) wraps to top edge (j=ysize)
    //   - Top edge (j=ysize) wraps to bottom edge (j=1)
    //
    // IN PARALLEL (MPI):
    //   - If N[_x_] > 1, multiple processes handle x-direction
    //     → MPI handles x-periodic BC (via halo exchange)
    //   - If N[_x_] == 1, only one process handles x-direction
    //     → This process handles x-periodic BC locally
    // =============================================================
    
    if ( periodic )
        {
            // Handle x-direction periodic BC (if this process covers full x-extent)
            if ( N[_x_] == 1 )
                {
                    // Propagate x-boundaries: copy edges to opposite ghost cells
                    for ( int i = 1; i <= xsize; i++ )
                        {
                            // Bottom ghost cell (j=0) copies from top edge (j=ysize)
                            new[ IDX(i, 0) ] = new[ IDX(i, ysize) ];
                            
                            // Top ghost cell (j=ysize+1) copies from bottom edge (j=1)
                            new[ IDX(i, ysize+1) ] = new[ IDX(i, 1) ];
                        }
                }
  
            // Handle y-direction periodic BC (if this process covers full y-extent)
            if ( N[_y_] == 1 ) 
                {
                    // Propagate y-boundaries: copy edges to opposite ghost cells
                    for ( int j = 1; j <= ysize; j++ )
                        {
                            // Left ghost cell (i=0) copies from right edge (i=xsize)
                            new[ IDX(0, j) ] = new[ IDX(xsize, j) ];
                            
                            // Right ghost cell (i=xsize+1) copies from left edge (i=1)
                            new[ IDX(xsize+1, j) ] = new[ IDX(1, j) ];
                        }
                }
        }

   #undef IDX  // Clean up macro
  return 0;    // Success
}

/*
 * =============================================================================
 * GET_TOTAL_ENERGY: Parallel sum of all grid values with OpenMP reduction
 * =============================================================================
 * 
 * This function computes the total energy in the system by summing all
 * grid point values. It's essential for verifying energy conservation.
 * 
 * WHY IS THIS IMPORTANT?
 *   - The stencil formula is designed to conserve energy
 *   - By tracking total energy over time, we can verify correctness
 *   - If total energy changes (without sources/sinks), there's a bug!
 * 
 * ENERGY CONSERVATION:
 *   - Without sources/sinks: total energy should be constant
 *   - With periodic BC: energy should be exactly conserved
 *   - With non-periodic BC: some energy "leaks" at boundaries
 * 
 * OPENMP REDUCTION:
 *   - Multiple threads sum different parts of the grid
 *   - reduction(+:totenergy) ensures thread-safe summation
 *   - Much faster than serial summation on multi-core CPUs
 * 
 * NUMERICAL PRECISION:
 *   - We use 'double' (64-bit) by default
 *   - For very large grids, we can enable 'long double' (80-bit or 128-bit)
 *   - This reduces roundoff errors when summing millions of values
 */
inline int get_total_energy( plane_t *plane,     // Grid to compute energy from
                            double  *energy )    // Output: total energy
{
    // Extract grid dimensions
    const int register xsize = plane->size[_x_];  // Width (interior points only)
    const int register ysize = plane->size[_y_];  // Height (interior points only)
    const int register fsize = xsize+2;           // Full width (with ghost cells)

    double * restrict data = plane->data;  // Pointer to grid data
    
    // Macro for 2D to 1D index conversion
   #define IDX( i, j ) ( (j)*fsize + (i) )

   // Choose precision: long double for high accuracy, double for speed
   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;  // 80-bit or 128-bit (platform-dependent)
   #else
    double totenergy = 0;       // 64-bit (standard double precision)
   #endif

    // =============================================================
    // OPENMP REDUCTION FOR PARALLEL SUM
    // =============================================================
    // #pragma omp parallel for schedule(static) reduction(+:totenergy)
    //   - Each thread maintains a PRIVATE copy of totenergy
    //   - Each thread sums a subset of grid points independently
    //   - At the end, all private copies are SUMMED together atomically
    //
    // HOW REDUCTION WORKS:
    //   1. Create private copy of totenergy for each thread (initialized to 0)
    //   2. Each thread sums its assigned rows into its private copy
    //   3. At end of parallel region, combine all private copies:
    //      final_totenergy = thread0_totenergy + thread1_totenergy + ...
    //
    // WHY REDUCTION?
    //   - Simple += would create DATA RACE (multiple threads writing same variable)
    //   - Race condition example:
    //       Thread 1 reads totenergy (100), adds 10, writes 110
    //       Thread 2 reads totenergy (100), adds 20, writes 120
    //       Final value: 120 (WRONG! Should be 130)
    //   - Reduction provides thread-safe way to compute aggregate values
    //   - Compiler handles synchronization automatically (no locks needed!)
    //
    // PERFORMANCE:
    //   - Very efficient: each thread works independently (no contention)
    //   - Only synchronizes once at the end (minimal overhead)
    //   - Near-linear speedup for large grids (>95% parallel efficiency)
    //   - Typical speedup: 7.5x on 8 cores
    //
    // ALTERNATIVE APPROACHES (NOT USED):
    //   1. Atomic operations: #pragma omp atomic
    //      - Too slow! Every += needs lock (massive contention)
    //      - 10-100x slower than reduction
    //   2. Critical sections: #pragma omp critical
    //      - Still too slow (serializes all additions)
    //      - No speedup from parallelization
    //   3. Thread-local arrays:
    //      - Manually allocate array of partial sums (one per thread)
    //      - Each thread sums to its array element
    //      - Finally sum array elements
    //      - Similar performance to reduction, but more code
    //   4. SIMD vectorization:
    //      - Can be combined with OpenMP for extra speedup
    //      - Compiler may auto-vectorize inner loop
    //      - Typical additional 2-4x speedup
    // =============================================================
    
    #pragma omp parallel for schedule(static) reduction(+:totenergy)
    for ( int j = 1; j <= ysize; j++ )      // Loop over rows (interior only)
        for ( int i = 1; i <= xsize; i++ )  // Loop over columns (interior only)
            // Accumulate energy (sum all grid values)
            // Note: We only sum interior points, not ghost cells
            totenergy += data[ IDX(i, j) ];

   #undef IDX  // Clean up macro

    // Convert result to double (in case we used long double)
    *energy = (double)totenergy;
    return 0;  // Success
}

/*
 * =============================================================================
 * UPDATE_INTERIOR: Stencil computation for INTERIOR points only
 * =============================================================================
 * 
 * This function updates only the INTERIOR points of the domain, skipping
 * the first and last row/column (the border points).
 * 
 * WHAT ARE INTERIOR POINTS?
 *   - Points that are NOT on the boundary of the local subdomain
 *   - Specifically: 2 <= i <= xsize-1, 2 <= j <= ysize-1
 *   - These points don't need halo data (ghost cells) from neighbors
 *   - All necessary data is already in this process's memory
 * 
 * CRITICAL FOR COMMUNICATION-COMPUTATION OVERLAP:
 *   - Interior points DON'T need halo data from neighboring MPI processes
 *   - Can be computed WHILE halo exchange is in progress (during MPI communication)
 *   - This overlap reduces idle time and improves scalability
 * 
 * TYPICAL USAGE PATTERN IN MPI CODE:
 *   1. Pack boundary data into send buffers
 *   2. Start non-blocking sends/receives (MPI_Isend/Irecv)
 *   3. ***Call update_interior() - compute while MPI transfers data***
 *   4. Wait for sends/receives to complete (MPI_Waitall)
 *   5. Unpack receive buffers into ghost cells
 *   6. Call update_borders() - compute border points with fresh halo data
 * 
 * WHY IS THIS FASTER?
 *   - Without overlap:  |--MPI wait (idle CPU)--|--compute all points--|
 *   - With overlap:     |--MPI transfer--|
 *                         |--compute interior--|--compute borders--|
 *   
 *   Time saved = (interior computation time) ≈ 70-90% of total computation
 *   
 *   For large grids, interior >> borders, so we hide most communication cost!
 * 
 * STRONG SCALING:
 *   - As we use more MPI processes, each subdomain gets smaller
 *   - Communication time stays roughly constant (fixed boundary size)
 *   - Computation time decreases (fewer points to compute)
 *   - Eventually, communication dominates → scaling stops
 *   - Overlap extends strong scaling limit by hiding communication
 * 
 * EXAMPLE WITH NUMBERS:
 *   - 1000x1000 grid, 10x10 = 100 MPI processes
 *   - Each process: 100x100 = 10,000 points
 *   - Interior: 98x98 = 9,604 points (96% of total)
 *   - Borders: 396 points (4% of total)
 *   - If we can hide 96% of communication cost, we get huge speedup!
 */
inline int update_interior( const plane_t *oldplane,  // Current grid state (time t)
                           plane_t *newplane )        // Output grid state (time t+dt)
{
    // Extract grid dimensions
    const uint xsize = oldplane->size[_x_];   // Interior width
    const uint ysize = oldplane->size[_y_];   // Interior height
    const uint fxsize = xsize+2;              // Full width (with ghost cells)

    // Macro for 2D to 1D index conversion
    #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    // Get pointers to grid data
    double * restrict old = oldplane->data;  // Read from old grid
    double * restrict new = newplane->data;  // Write to new grid

    // Diffusion parameters (same as update_plane)
    const double alpha = 0.6;                    // 60% stays at current location
    const double constant = (1-alpha) / 4.0;     // 10% goes to each neighbor
    
    // =============================================================
    // OPENMP PARALLELIZATION OF INTERIOR POINTS
    // =============================================================
    // #pragma omp parallel for schedule(static)
    //   - Parallelizes computation across threads
    //   - Each thread updates a subset of interior rows
    //
    // LOOP BOUNDS (KEY DIFFERENCE FROM update_plane):
    //   - j: from 2 to ysize-1 (skip j=1 and j=ysize, which are borders)
    //   - i: from 2 to xsize-1 (skip i=1 and i=xsize, which are borders)
    //
    // VISUAL EXAMPLE (10x10 grid):
    //   
    //   Ghost: [ 0 ][ 1  2  3  4  5  6  7  8  9  10][11]
    //        ------------------------------------------------
    //      0:  [ ][ G  G  G  G  G  G  G  G  G  G ][ ]  <- Ghost row
    //      1:  [ ][B  I  I  I  I  I  I  I  I  B ][ ]  <- Border row (not computed here)
    //      2:  [ ][B  I  I  I  I  I  I  I  I  B ][ ]  <- Interior rows
    //      3:  [ ][B  I  I  I  I  I  I  I  I  B ][ ]     (computed here)
    //      4:  [ ][B  I  I  I  I  I  I  I  I  B ][ ]
    //      5:  [ ][B  I  I  I  I  I  I  I  I  B ][ ]
    //      6:  [ ][B  I  I  I  I  I  I  I  I  B ][ ]
    //      7:  [ ][B  I  I  I  I  I  I  I  I  B ][ ]
    //      8:  [ ][B  I  I  I  I  I  I  I  B ][ ]
    //      9:  [ ][B  I  I  I  I  I  I  I  I  B ][ ]
    //     10:  [ ][B  B  B  B  B  B  B  B  B  B ][ ]  <- Border row (not computed here)
    //     11:  [ ][ G  G  G  G  G  G  G  G  G  G ][ ]  <- Ghost row
    //        ------------------------------------------------
    //   
    //   Legend: G = Ghost, B = Border, I = Interior
    //   
    //   This function updates only the 'I' points (interior).
    //   Borders ('B') are updated later by update_borders().
    //
    // WHY SKIP BORDERS?
    //   - Border points (j=1, j=ysize, i=1, i=xsize) need halo data from neighbors
    //   - Halo data comes from neighboring MPI processes via MPI communication
    //   - While MPI is transferring halo, we compute interior points
    //   - After MPI completes, update_borders() handles border points
    //
    // COMMUNICATION-COMPUTATION OVERLAP TIMELINE:
    //   
    //   Time 0:    Start MPI_Isend/Irecv (non-blocking)
    //            ↓
    //   Time 1:    Call update_interior() ← WE ARE HERE
    //            ↓ (MPI transfers halo data in background)
    //            ↓ (CPU computes interior points)
    //   Time 2:    Interior computation finishes
    //            ↓
    //   Time 3:    MPI_Waitall (wait for halo data)
    //            ↓
    //   Time 4:    Unpack halo data into ghost cells
    //            ↓
    //   Time 5:    Call update_borders() (use fresh halo data)
    //            ↓
    //   Time 6:    Done!
    //   
    //   If interior computation (Time 1-2) >= MPI transfer (Time 0-3),
    //   then we completely hide the communication cost! Zero idle time!
    //
    // PERFORMANCE IMPACT:
    //   - Reduces wall-clock time by overlapping communication and computation
    //   - Critical for strong scaling (as problem size per process shrinks)
    //   - Can hide most or all communication latency for large enough domains
    //   - Typical speedup: 1.2-2x compared to no overlap (depends on network speed)
    //
    // WHEN OVERLAP WORKS BEST:
    //   - Large subdomains: more interior points → more computation to hide communication
    //   - Slow network: more time to hide
    //   - Fast CPU: finish interior before MPI completes
    //   
    // WHEN OVERLAP DOESN'T HELP MUCH:
    //   - Tiny subdomains: borders >> interior, little to overlap
    //   - Very fast network (InfiniBand): communication finishes too quickly
    //   - Slow CPU: interior computation outlasts MPI anyway
    // =============================================================
    
    #pragma omp parallel for schedule(static)
    for (uint j = 2; j <= ysize-1; j++)      // Skip first and last row (borders)
        for (uint i = 2; i <= xsize-1; i++)  // Skip first and last column (borders)
            {
                // Apply 5-point stencil (same formula as update_plane)
                // Note: All neighbors (i±1, j±1) are guaranteed to be in local memory
                // (not in ghost cells that need MPI data)
                double result = old[ IDX(i,j) ] * alpha
                                + ( old[IDX(i-1, j)]    // West neighbor (interior or border)
                                  + old[IDX(i+1, j)]    // East neighbor (interior or border)
                                  + old[IDX(i, j-1)]    // South neighbor (interior or border)
                                  + old[IDX(i, j+1)]    // North neighbor (interior or border)
                                  ) * constant;
                
                new[ IDX(i,j) ] = result;
            }

    #undef IDX  // Clean up macro
    return 0;   // Success
}

/*
 * =============================================================================
 * UPDATE_BORDERS: Stencil computation for BORDER points only
 * =============================================================================
 * 
 * This function updates only the BORDER points of the domain:
 *   - First and last row (j=1, j=ysize)
 *   - First and last column (i=1, i=xsize)
 * 
 * WHAT ARE BORDER POINTS?
 *   - Points on the boundary of the local subdomain
 *   - These points DO need halo data (ghost cells) from neighboring MPI processes
 *   - Must wait for MPI communication to complete before computing
 * 
 * MUST BE CALLED AFTER:
 *   1. MPI_Isend/Irecv has been initiated (non-blocking communication)
 *   2. update_interior() has been called (interior points computed)
 *   3. MPI_Waitall has completed (halo data received)
 *   4. Halo data has been unpacked into ghost cells
 * 
 * ONLY THEN can we safely compute border points!
 * 
 * CRITICAL FOR COMMUNICATION-COMPUTATION OVERLAP:
 *   - Border points NEED halo data from neighboring processes
 *   - Can only be computed AFTER halo exchange completes
 *   - Called after update_interior() to complete the stencil computation
 * 
 * WHY SEPARATE FUNCTION?
 *   - Enables overlap: compute interior while MPI transfers halo
 *   - Without split: must wait for MPI before computing anything
 *   - With split: compute interior immediately, borders after MPI
 * 
 * LOOP STRUCTURE:
 *   - Two separate loops: horizontal borders (top/bottom) and vertical borders (left/right)
 *   - Horizontal loop excludes corners (handled by vertical loop)
 *   - This avoids duplicate computation of corner points
 * 
 * CORNER HANDLING:
 *   - Corners are at (1,1), (1,ysize), (xsize,1), (xsize,ysize)
 *   - Corners are handled by the vertical border loop (includes j=1 to j=ysize)
 *   - Horizontal border loop explicitly excludes corners (i from 2 to xsize-1)
 */
inline int update_borders( const int periodic,      // 1 if periodic BC, 0 otherwise
                          const vec2_t N,           // MPI process grid (for periodic BC)
                          const plane_t *oldplane,  // Current grid state (with updated halo)
                          plane_t *newplane )       // Output grid state
{
    // Extract grid dimensions
    const uint xsize = oldplane->size[_x_];  // Interior width
    const uint ysize = oldplane->size[_y_];  // Interior height
    const uint fxsize = xsize+2;             // Full width (with ghost cells)

    // Macro for 2D to 1D index conversion
    #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    // Get pointers to grid data
    double * restrict old = oldplane->data;  // Read from old grid (with fresh halo data!)
    double * restrict new = newplane->data;  // Write to new grid

    // Diffusion parameters (same as update_plane and update_interior)
    const double alpha = 0.6;                    // 60% stays at current location
    const double constant = (1-alpha) / 4.0;     // 10% goes to each neighbor
    
    // Temporary variables for optimization (avoid recalculating IDX)
    double center, neighbors;
    uint i, j;
    
    // =============================================================
    // UPDATE TOP AND BOTTOM BORDERS (HORIZONTAL BORDERS)
    // =============================================================
    // #pragma omp parallel for schedule(static)
    //   - Parallelizes border update across threads
    //   - Each thread updates a subset of border points
    //
    // LOOP STRUCTURE:
    //   - Updates top border (j=1) and bottom border (j=ysize)
    //   - Loop from i=2 to xsize-1 (EXCLUDE CORNERS at i=1 and i=xsize)
    //   - Corners are handled by the vertical border loop below
    //   - This avoids computing each corner twice
    //
    // VISUAL EXAMPLE (10x10 grid):
    //   
    //   Ghost: [ 0 ][ 1  2  3  4  5  6  7  8  9  10][11]
    //        ------------------------------------------------
    //      0:  [ ][ G  G  G  G  G  G  G  G  G  G ][ ]  <- Ghost (from SOUTH neighbor)
    //      1:  [ ][C  T  T  T  T  T  T  T  T  C ][ ]  <- Top border (computed here, except corners)
    //      2:  [ ][L  I  I  I  I  I  I  I  I  R ][ ]
    //      3:  [ ][L  I  I  I  I  I  I  I  I  R ][ ]
    //      ...
    //      9:  [ ][L  I  I  I  I  I  I  I  I  R ][ ]
    //     10:  [ ][C  B  B  B  B  B  B  B  B  C ][ ]  <- Bottom border (computed here, except corners)
    //     11:  [ ][ G  G  G  G  G  G  G  G  G  G ][ ]  <- Ghost (from NORTH neighbor)
    //        ------------------------------------------------
    //   
    //   Legend: G = Ghost, T = Top border, B = Bottom border, C = Corner
    //           L = Left border, R = Right border, I = Interior
    //   
    //   This loop updates 'T' and 'B' points (excluding 'C' corners).
    //
    // WHY AFTER MPI_Waitall?
    //   - Top border points (j=1) use ghost cells at j=0
    //     → Ghost at j=0 comes from SOUTH neighbor via MPI
    //   - Bottom border points (j=ysize) use ghost cells at j=ysize+1
    //     → Ghost at j=ysize+1 comes from NORTH neighbor via MPI
    //   - Must wait for MPI to complete before accessing these ghost cells
    //   - If we compute before MPI completes, we'd use stale/uninitialized data!
    //
    // OPTIMIZATION:
    //   - We compute top and bottom borders in the same loop iteration
    //   - This improves cache locality (access similar i values)
    //   - Reduces loop overhead (one loop instead of two)
    // =============================================================
    
    #pragma omp parallel for schedule(static) private(i)
    for ( i = 2; i <= xsize-1; i++ ) {  // Exclude corners (i=1 and i=xsize)
        // ----------------------------------------------------------
        // Update top border (j=1)
        // ----------------------------------------------------------
        // Stencil neighbors:
        //   - (i-1,1): left neighbor (interior or left border)
        //   - (i+1,1): right neighbor (interior or right border)
        //   - (i,0):   south neighbor (GHOST CELL from SOUTH MPI neighbor)
        //   - (i,2):   north neighbor (interior)
        center = old[ IDX(i,1) ];
        neighbors = old[IDX(i-1, 1)] + old[IDX(i+1, 1)] + old[IDX(i, 0)] + old[IDX(i, 2)];
        new[ IDX(i,1) ] = center * alpha + neighbors * constant;

        // ----------------------------------------------------------
        // Update bottom border (j=ysize)
        // ----------------------------------------------------------
        // Stencil neighbors:
        //   - (i-1,ysize): left neighbor (interior or left border)
        //   - (i+1,ysize): right neighbor (interior or right border)
        //   - (i,ysize-1): south neighbor (interior)
        //   - (i,ysize+1): north neighbor (GHOST CELL from NORTH MPI neighbor)
        center = old[ IDX(i,ysize) ];
        neighbors = old[IDX(i-1, ysize)] + old[IDX(i+1, ysize)] + old[IDX(i, ysize-1)] + old[IDX(i, ysize+1)];
        new[ IDX(i,ysize) ] = center * alpha + neighbors * constant;
    }

    // =============================================================
    // UPDATE LEFT AND RIGHT BORDERS (VERTICAL BORDERS)
    // =============================================================
    // #pragma omp parallel for schedule(static)
    //   - Parallelizes border update across threads
    //   - Each thread updates a subset of border points
    //
    // LOOP STRUCTURE:
    //   - Updates left border (i=1) and right border (i=xsize)
    //   - Loop from j=1 to ysize (INCLUDES ALL CORNERS)
    //   - This ensures all 4 corners are computed exactly once
    //   - Corners at (1,1), (1,ysize), (xsize,1), (xsize,ysize)
    //
    // VISUAL EXAMPLE (10x10 grid):
    //   
    //   Ghost: [ 0 ][ 1  2  3  4  5  6  7  8  9  10][11]
    //        ------------------------------------------------
    //      0:  [ ][ G  G  G  G  G  G  G  G  G  G ][ ]  
    //      1:  [ ][C  T  T  T  T  T  T  T  T  C ][ ]  <- Corners computed here (C)
    //      2:  [ ][L  I  I  I  I  I  I  I  I  R ][ ]  <- Left/right borders (L, R)
    //      3:  [ ][L  I  I  I  I  I  I  I  I  R ][ ]
    //      ...
    //      9:  [ ][L  I  I  I  I  I  I  I  I  R ][ ]
    //     10:  [ ][C  B  B  B  B  B  B  B  B  C ][ ]  <- Corners computed here (C)
    //     11:  [ ][ G  G  G  G  G  G  G  G  G  G ][ ]
    //        ------------------------------------------------
    //        |                                    |
    //    Ghost from                          Ghost from
    //   WEST neighbor                       EAST neighbor
    //   
    //   This loop updates 'L', 'R', and all 'C' (corner) points.
    //
    // WHY AFTER MPI_Waitall?
    //   - Left border points (i=1) use ghost cells at i=0
    //     → Ghost at i=0 comes from WEST neighbor via MPI
    //   - Right border points (i=xsize) use ghost cells at i=xsize+1
    //     → Ghost at i=xsize+1 comes from EAST neighbor via MPI
    //   - Must wait for MPI to complete before accessing these ghost cells
    //
    // CORNER GHOST CELLS:
    //   - Corner ghost cells (0,0), (xsize+1,0), (0,ysize+1), (xsize+1,ysize+1)
    //   - These come from diagonal MPI neighbors
    //   - In some implementations, diagonal neighbors don't communicate
    //   - In this code, we assume diagonal ghost cells are filled (either by
    //     direct communication or by copying from edge ghost cells)
    // =============================================================
    
    #pragma omp parallel for schedule(static) private(j)
    for ( j = 1; j <= ysize; j++ ) {  // Include all rows from 1 to ysize (includes corners)
        // ----------------------------------------------------------
        // Update left border (i=1)
        // ----------------------------------------------------------
        // Stencil neighbors:
        //   - (0,j):   west neighbor (GHOST CELL from WEST MPI neighbor)
        //   - (2,j):   east neighbor (interior)
        //   - (1,j-1): south neighbor (border or interior or corner)
        //   - (1,j+1): north neighbor (border or interior or corner)
        center = old[ IDX(1,j) ];
        neighbors = old[IDX(0, j)] + old[IDX(2, j)] + old[IDX(1, j-1)] + old[IDX(1, j+1)];
        new[ IDX(1,j) ] = center * alpha + neighbors * constant;

        // ----------------------------------------------------------
        // Update right border (i=xsize)
        // ----------------------------------------------------------
        // Stencil neighbors:
        //   - (xsize-1,j): west neighbor (interior)
        //   - (xsize+1,j): east neighbor (GHOST CELL from EAST MPI neighbor)
        //   - (xsize,j-1): south neighbor (border or interior or corner)
        //   - (xsize,j+1): north neighbor (border or interior or corner)
        center = old[ IDX(xsize,j) ];
        neighbors = old[IDX(xsize-1, j)] + old[IDX(xsize+1, j)] + old[IDX(xsize, j-1)] + old[IDX(xsize, j+1)];
        new[ IDX(xsize,j) ] = center * alpha + neighbors * constant;
    }

    // =============================================================
    // PERIODIC BOUNDARY CONDITIONS (UPDATE GHOST CELLS)
    // =============================================================
    // After computing all border points, update ghost cells for periodic BC.
    // This is similar to update_plane, but now we copy from 'new' grid.
    //
    // IN PARALLEL (MPI):
    //   - If N[_x_] > 1 or N[_y_] > 1, some directions use MPI for periodic BC
    //   - If N[_x_] == 1, this process handles full x-extent → local periodic BC
    //   - If N[_y_] == 1, this process handles full y-extent → local periodic BC
    // =============================================================
    
    if ( periodic ) {
        // Handle x-direction periodic BC (if this process covers full x-extent)
        if ( N[_x_] == 1 ) {
            for ( j = 1; j <= ysize; j++ ) {
                // Left ghost (i=0) wraps to right edge (i=xsize)
                new[ IDX( 0, j) ]       = new[ IDX(xsize, j) ];
                // Right ghost (i=xsize+1) wraps to left edge (i=1)
                new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ];
            }
        }

        // Handle y-direction periodic BC (if this process covers full y-extent)
        if ( N[_y_] == 1 ) {
            for ( i = 1; i <= xsize; i++ ) {
                // Bottom ghost (j=0) wraps to top edge (j=ysize)
                new[ IDX( i, 0 ) ]       = new[ IDX(i, ysize) ];
                // Top ghost (j=ysize+1) wraps to bottom edge (j=1)
                new[ IDX( i, ysize+1) ] = new[ IDX(i, 1) ];
            }
        }
    }

    #undef IDX  // Clean up macro
    return 0;   // Success
}

