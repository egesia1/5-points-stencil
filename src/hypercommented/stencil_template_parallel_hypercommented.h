/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * =============================================================================
 * STENCIL_TEMPLATE_PARALLEL.H - PARALLEL STENCIL TEMPLATE (MPI + OPENMP)
 * HYPERCOMMENTED VERSION - EDUCATIONAL PURPOSES
 * =============================================================================
 * 
 * This is a TEMPLATE header file for parallel stencil computation.
 * It provides basic structure and hints for implementing MPI+OpenMP parallelization.
 * 
 * KEY DIFFERENCES FROM COMPLETE PARALLEL VERSION (stencil.h):
 *   - Simplified function signatures
 *   - Missing implementation details (marked with comments/hints)
 *   - Designed for students to complete as an exercise
 *   - Shows basic structure without full complexity
 * 
 * COMPARE WITH:
 *   - stencil.h: Complete, production-ready parallel implementation
 *   - stencil_template_serial.h: Serial implementation (no MPI)
 *   
 * THIS FILE: Skeleton/template for learning MPI+OpenMP parallelization
 * 
 * =============================================================================
 * WHAT IS A STENCIL COMPUTATION?
 * =============================================================================
 * 
 * A stencil computation is a pattern where each element in an array is updated
 * based on its neighbors. This is a 5-point stencil:
 * 
 *            (i, j+1)
 *                |
 *    (i-1,j) - (i,j) - (i+1,j)
 *                |
 *            (i, j-1)
 * 
 * In parallel (MPI), the grid is divided among processes:
 *   - Each process owns a subdomain (patch) of the full grid
 *   - Border points need data from neighboring processes (halo exchange)
 *   - OpenMP parallelizes computation within each subdomain
 * 
 * =============================================================================
 * PARALLEL DECOMPOSITION STRATEGY:
 * =============================================================================
 * 
 * DOMAIN DECOMPOSITION:
 *   - Full grid divided into subdomains (one per MPI process)
 *   - Example: 100x100 grid, 4 processes → each gets 50x50 patch
 *   - Each process computes stencil for its patch
 * 
 * HALO EXCHANGE (GHOST CELLS):
 *   - Border points need values from neighboring patches
 *   - These values are stored in "ghost cells" (halo)
 *   - MPI sends/receives ghost cell data between neighbors
 * 
 * HYBRID PARALLELIZATION (MPI + OPENMP):
 *   - MPI: distributed memory (multiple nodes)
 *   - OpenMP: shared memory (multiple cores per node)
 *   - Typical setup: 4-16 MPI processes per node, 8-64 OpenMP threads per process
 * 
 * =============================================================================
 * LEARNING OBJECTIVES:
 * =============================================================================
 * 
 * By completing this template, you will learn:
 *   1. How to decompose a grid across MPI processes
 *   2. How to implement halo exchange (ghost cell communication)
 *   3. How to handle periodic boundary conditions in parallel
 *   4. How to combine MPI with OpenMP for hybrid parallelization
 *   5. How to optimize communication-computation overlap
 * 
 * =============================================================================
 */

// =============================================================================
// STANDARD LIBRARY INCLUDES
// =============================================================================

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

#define NORTH 0   // Neighbor in the +y direction (j increases)
#define SOUTH 1   // Neighbor in the -y direction (j decreases)
#define EAST  2   // Neighbor in the +x direction (i increases)
#define WEST  3   // Neighbor in the -x direction (i decreases)

// =============================================================================
// COMMUNICATION DIRECTION CONSTANTS
// =============================================================================

#define SEND 0    // Buffer used for sending data to neighbors
#define RECV 1    // Buffer used for receiving data from neighbors

// =============================================================================
// TIME LEVEL CONSTANTS
// =============================================================================

#define OLD 0     // Index for old/current time level
#define NEW 1     // Index for new/next time level

// =============================================================================
// COORDINATE INDICES
// =============================================================================

#define _x_ 0     // Index for x-coordinate (columns)
#define _y_ 1     // Index for y-coordinate (rows)

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

// Unsigned integer type (shorthand for cleaner code)
typedef unsigned int uint;

// 2D vector type for grid sizes, positions, etc.
// Example: vec2_t size = {100, 200};  // 100 columns, 200 rows
typedef uint vec2_t[2];

// Communication buffers type
// buffers_t[NORTH/SOUTH/EAST/WEST]
// This stores 4 pointers: one buffer per direction
// Note: Simplified from full version (no separate SEND/RECV dimension)
typedef double *restrict buffers_t[4];

// Grid plane structure
// Stores a 2D grid of double-precision values and its size
typedef struct {
    double   * restrict data;  // Pointer to 1D array storing 2D grid
                              // 'restrict' keyword: compiler optimization hint
    vec2_t     size;          // Grid dimensions: size[_x_] = columns, size[_y_] = rows
                              // Note: actual allocated size is (size[_x_]+2) * (size[_y_]+2)
} plane_t;

// =============================================================================
// FUNCTION PROTOTYPES
// =============================================================================

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
// NOTE: This is a simplified version. In production code, you might split
//       this into update_interior + update_borders for better overlap.
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
// get_total_energy: Compute total energy in the grid
// -----------------------------------------------------------------------------
// NOTE: In parallel, this requires MPI reduction to sum across all processes!
// Parameters:
//   plane: grid to compute energy from
//   energy: output total energy (sum of all grid values)
// Returns: 0 on success
extern int get_total_energy( plane_t *,
                             double  * );

// -----------------------------------------------------------------------------
// initialize: Set up MPI, parse arguments, allocate memory
// -----------------------------------------------------------------------------
int initialize ( MPI_Comm *,  // MPI communicator
                 int       ,  // argc
		 int       ,  // Number of MPI processes
		 int       ,  // MPI rank (process ID)
		 char    **,  // argv
                 vec2_t   *,  // Global grid size
                 vec2_t   *,  // Local grid size (subdomain)
		 int      *,  // Periodic BC flag
                 int      *,  // Number of iterations
		 int      *,  // Number of sources
		 int      *,  // Report frequency
		 int      *,  // ...additional parameters...
		 int      *,
                 vec2_t  **,  // Source coordinates
                 double   *,  // Energy per source
                 plane_t  *,  // Grid plane structure
                 buffers_t * );  // Communication buffers

// -----------------------------------------------------------------------------
// memory_release: Free all dynamically allocated memory
// -----------------------------------------------------------------------------
int memory_release (plane_t   * );

// -----------------------------------------------------------------------------
// output_energy_stat: Print energy statistics
// -----------------------------------------------------------------------------
int output_energy_stat ( int      ,
                         plane_t *,
                         double   ,
                         int      ,
                         MPI_Comm *);

// =============================================================================
// INLINE FUNCTION IMPLEMENTATIONS
// =============================================================================

/*
 * =============================================================================
 * INJECT_ENERGY: Add energy at source locations (PARALLEL VERSION)
 * =============================================================================
 * 
 * In parallel, this is more complex than serial:
 *   - Source might be in this process's subdomain or another process's
 *   - Only inject if source coordinates fall within this process's patch
 *   - Periodic BC requires special handling at subdomain boundaries
 * 
 * EXERCISE FOR STUDENTS:
 *   - Determine if source is in this process's subdomain
 *   - Inject energy only if source is local
 *   - Handle periodic BC when this process owns the edge
 * 
 * HINT: You need to know:
 *   - This process's subdomain coordinates in the global grid
 *   - Whether this process is at the global boundary (for periodic BC)
 */
inline int inject_energy ( const int      periodic,   // 1 if periodic BC, 0 otherwise
                           const int      Nsources,   // Number of energy sources
			   const vec2_t  *Sources,    // Array of source coordinates
			   const double   energy,     // Amount of energy to inject
                                 plane_t *plane,      // Grid to inject into (local subdomain)
                           const vec2_t   N           // MPI process grid dimensions
                           )
{
    // Extract local grid dimensions
    const uint register sizex = plane->size[_x_]+2;  // Full width including ghost cells
    double * restrict data = plane->data;            // Pointer to grid data
    
    // Macro for converting 2D index (i,j) to 1D array index
   #define IDX( i, j ) ( (j)*sizex + (i) )
   
    // Loop over all energy sources
    for (int s = 0; s < Nsources; s++)
        {
            // Extract source coordinates (global coordinates)
            int x = Sources[s][_x_];  // Global column index
            int y = Sources[s][_y_];  // Global row index
            
            // =============================================================
            // EXERCISE: Determine if source is in this process's subdomain
            // =============================================================
            // 
            // You need to:
            //   1. Convert global coordinates (x,y) to local coordinates
            //   2. Check if local coordinates are within this subdomain
            //   3. Only inject if source is local
            //
            // HINT: You need to know this process's position in the process grid
            //       This information should come from the initialize() function
            //       (not shown in this template)
            // =============================================================
            
            // For now, assume source is always local (INCORRECT for multi-process!)
            // TODO: Add check for whether source is in this subdomain
            data[ IDX(x,y) ] += energy;
            
            // Handle periodic boundary conditions
            if ( periodic )
                {
                    // =============================================================
                    // EXERCISE: Handle periodic BC in parallel
                    // =============================================================
                    //
                    // If this process covers the full extent in x-direction (N[_x_] == 1),
                    // then we need to propagate boundaries locally (like serial code).
                    //
                    // If N[_x_] > 1, then periodic BC is handled by MPI halo exchange,
                    // not here.
                    //
                    // HINT: Check the serial version (stencil_template_serial.h) for
                    //       how to propagate boundaries locally.
                    // =============================================================
                    
                    if ( (N[_x_] == 1)  )
                        {
                            // TODO: Propagate the boundaries if needed
                            // Check the serial version for reference
                        }
                    
                    if ( (N[_y_] == 1) )
                        {
                            // TODO: Propagate the boundaries if needed
                            // Check the serial version for reference
                        }
                }                
        }
   #undef IDX
    
  return 0;  // Success
}

/*
 * =============================================================================
 * UPDATE_PLANE: Full 5-point stencil computation (PARALLEL VERSION)
 * =============================================================================
 * 
 * This is the core stencil computation, but now for a local subdomain.
 * 
 * KEY DIFFERENCES FROM SERIAL:
 *   - We only compute points in this process's subdomain
 *   - Ghost cells are filled by MPI halo exchange (done before calling this)
 *   - Periodic BC is handled differently (MPI or local, depending on N)
 * 
 * STENCIL FORMULA (same as serial):
 *   U_{i,j}^{n+1} = α·U_{i,j}^n + (1-α)/4·(sum of 4 neighbors)
 * 
 * HALO EXCHANGE WORKFLOW:
 *   1. Pack boundary data into send buffers
 *   2. MPI_Isend/Irecv to exchange with neighbors
 *   3. MPI_Waitall to ensure data is received
 *   4. Unpack receive buffers into ghost cells
 *   5. **Call this function to compute stencil**
 *   6. Repeat for next time step
 * 
 * OPTIMIZATION OPPORTUNITY:
 *   - For better performance, split this into update_interior + update_borders
 *   - This allows computing interior points during MPI communication
 *   - See stencil.h for the complete implementation
 * 
 * EXERCISE FOR STUDENTS:
 *   - Implement the stencil computation (similar to serial version)
 *   - Add OpenMP parallelization (#pragma omp parallel for)
 *   - Handle periodic BC correctly in parallel context
 */
inline int update_plane ( const int      periodic,    // 1 if periodic BC, 0 otherwise
                          const vec2_t   N,            // MPI process grid
                          const plane_t *oldplane,     // Current grid state
                                plane_t *newplane )    // Output grid state
    
{
    // Extract grid dimensions
    uint register fxsize = oldplane->size[_x_]+2;  // Full width (with ghost cells)
    uint register fysize = oldplane->size[_y_]+2;  // Full height (with ghost cells)
    
    uint register xsize = oldplane->size[_x_];     // Interior width
    uint register ysize = oldplane->size[_y_];     // Interior height
    
    // Macro for 2D to 1D index conversion
   #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    // HINT: You may attempt to:
    //       (i)  Manually unroll the loop
    //       (ii) Ask the compiler to do it
    // For instance:
    // #pragma GCC unroll 4
    //
    // HINT: This loop is a good candidate for OpenMP parallelization!
    //       Add: #pragma omp parallel for schedule(static)

    // Get pointers to grid data
    double * restrict old = oldplane->data;  // Read from old grid
    double * restrict new = newplane->data;  // Write to new grid
    
    // =============================================================
    // EXERCISE: Implement stencil computation
    // =============================================================
    //
    // Loop over all interior points (1 <= i <= xsize, 1 <= j <= ysize)
    // and apply the 5-point stencil formula.
    //
    // HINT: Check the serial version for the formula.
    //       The only difference is that you're computing a subdomain,
    //       not the full grid.
    //
    // HINT: Add OpenMP parallelization for multi-threading:
    //       #pragma omp parallel for schedule(static)
    //
    // TODO: Implement the nested loops and stencil computation here
    // =============================================================
    
    for (uint j = 1; j <= ysize; j++)
        for ( uint i = 1; i <= xsize; i++)
            {
                // =============================================================
                // NOTE: (i-1,j), (i+1,j), (i,j-1) and (i,j+1) always exist even
                //       if this patch is at some border without periodic conditions;
                //       in that case it is assumed that the +-1 points are outside the
                //       plate and always have a value of 0, i.e. they are an
                //       "infinite sink" of heat
                // =============================================================
                
                // Five-points stencil formula
                //
                // HINT: Check the serial version for some optimization
                //
                // TODO: Implement the stencil formula
                //
                // Simplified version (REPLACE THIS WITH OPTIMIZED VERSION):
                new[ IDX(i,j) ] =
                    old[ IDX(i,j) ] / 2.0 + ( old[IDX(i-1, j)] + old[IDX(i+1, j)] +
                                              old[IDX(i, j-1)] + old[IDX(i, j+1)] ) /4.0 / 2.0;
                
            }

    // =============================================================
    // HANDLE PERIODIC BOUNDARY CONDITIONS
    // =============================================================
    //
    // If this process covers the full extent in a direction (N[_x_] == 1 or N[_y_] == 1),
    // then we need to propagate boundaries locally.
    //
    // If N[_x_] > 1 or N[_y_] > 1, periodic BC is handled by MPI halo exchange,
    // so we don't need to do anything here.
    //
    // EXERCISE: Complete the periodic BC handling
    // =============================================================
    
    if ( periodic )
        {
            if ( N[_x_] == 1 )
                {
                    // TODO: Propagate the boundaries as needed
                    // Check the serial version for reference
                }
  
            if ( N[_y_] == 1 ) 
                {
                    // TODO: Propagate the boundaries as needed
                    // Check the serial version for reference
                }
        }

    
   #undef IDX
  return 0;  // Success
}

/*
 * =============================================================================
 * GET_TOTAL_ENERGY: Parallel sum of all grid values (PARALLEL VERSION)
 * =============================================================================
 * 
 * In parallel, this is more complex than serial:
 *   - Each process computes the sum of its local subdomain
 *   - Then we need MPI_Reduce or MPI_Allreduce to sum across all processes
 * 
 * TWO-LEVEL PARALLELIZATION:
 *   1. OpenMP: parallel sum within this process (shared memory)
 *   2. MPI: sum across all processes (distributed memory)
 * 
 * OPENMP REDUCTION:
 *   - Each thread sums a subset of this process's grid
 *   - OpenMP combines thread-local sums into process-local sum
 * 
 * MPI REDUCTION:
 *   - MPI_Allreduce sums process-local sums into global sum
 *   - All processes get the final global sum
 * 
 * EXERCISE FOR STUDENTS:
 *   - Implement OpenMP reduction for local sum (similar to serial)
 *   - Add MPI_Allreduce to compute global sum
 *   - Handle numerical precision issues (long double option)
 */
inline int get_total_energy( plane_t *plane,      // Grid to compute energy from
                             double  *energy )    // Output: total energy
/*
 * NOTE: This routine is a good candidate for OpenMP parallelization
 *       (for the local sum)
 * 
 * NOTE: After computing the local sum, you need MPI reduction to get
 *       the global sum across all processes!
 */
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
    // EXERCISE: Implement local energy sum with OpenMP
    // =============================================================
    //
    // HINT: Use OpenMP reduction, similar to serial version:
    //       #pragma omp parallel for schedule(static) reduction(+:totenergy)
    //
    // HINT: You may also attempt loop unrolling:
    //       #pragma GCC unroll 4
    //
    // TODO: Implement the nested loops with OpenMP parallelization
    // =============================================================
    
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];

    
   #undef IDX

    // =============================================================
    // EXERCISE: Add MPI reduction to compute global sum
    // =============================================================
    //
    // At this point, 'totenergy' contains the sum for THIS process only.
    // We need to sum across all processes to get the global total.
    //
    // HINT: Use MPI_Allreduce:
    //       double local_energy = (double)totenergy;
    //       double global_energy;
    //       MPI_Allreduce(&local_energy, &global_energy, 1, MPI_DOUBLE, 
    //                     MPI_SUM, MPI_COMM_WORLD);
    //       *energy = global_energy;
    //
    // TODO: Add MPI_Allreduce here
    // =============================================================

    // For now, just return local sum (INCORRECT for multi-process!)
    *energy = (double)totenergy;
    return 0;
}

/*
 * =============================================================================
 * ADDITIONAL EXERCISES AND LEARNING OPPORTUNITIES:
 * =============================================================================
 * 
 * 1. HALO EXCHANGE:
 *    - Implement pack/unpack functions for boundary data
 *    - Implement MPI_Isend/Irecv for non-blocking communication
 *    - Handle corner cases (diagonal neighbors, non-periodic boundaries)
 * 
 * 2. COMMUNICATION-COMPUTATION OVERLAP:
 *    - Split update_plane into update_interior + update_borders
 *    - Call update_interior BEFORE MPI_Waitall
 *    - Call update_borders AFTER MPI_Waitall
 *    - Measure speedup from overlap
 * 
 * 3. LOAD BALANCING:
 *    - What happens if grid doesn't divide evenly?
 *    - How to handle non-uniform process grid (e.g., 3x5 processes)?
 *    - Implement adaptive load balancing
 * 
 * 4. SCALABILITY ANALYSIS:
 *    - Strong scaling: fix problem size, increase processes
 *    - Weak scaling: fix problem size per process, increase processes
 *    - Measure parallel efficiency: speedup / num_processes
 *    - Identify bottlenecks (computation vs. communication)
 * 
 * 5. OPTIMIZATION:
 *    - Experiment with different OpenMP schedules (static, dynamic, guided)
 *    - Try different MPI communication patterns (blocking, non-blocking, one-sided)
 *    - Optimize memory layout for cache efficiency
 *    - Use SIMD vectorization (compiler auto-vectorization or intrinsics)
 * 
 * 6. ADVANCED TOPICS:
 *    - Implement overlapping MPI communication with computation
 *    - Use MPI derived datatypes for non-contiguous data
 *    - Implement persistent communication (MPI_Send_init, MPI_Start)
 *    - Try MPI one-sided communication (MPI_Put, MPI_Get, MPI_Win)
 *    - Implement dynamic load balancing with work stealing
 * 
 * =============================================================================
 * DEBUGGING TIPS:
 * =============================================================================
 * 
 * 1. Start with small grids (e.g., 10x10) and few processes (2-4)
 * 2. Print intermediate results to verify correctness
 * 3. Check energy conservation: total energy should be constant (periodic BC)
 * 4. Visualize the grid to spot errors (halo exchange bugs are often visible)
 * 5. Use MPI debugging tools (e.g., TotalView, DDT, GDB with MPI)
 * 6. Check for deadlocks: make sure send/recv are properly matched
 * 7. Verify ghost cells are correctly filled before computing borders
 * 
 * =============================================================================
 * PERFORMANCE PROFILING:
 * =============================================================================
 * 
 * 1. Use timing functions to measure:
 *    - Total time per iteration
 *    - Computation time (stencil)
 *    - Communication time (halo exchange)
 *    - Synchronization overhead (MPI_Waitall)
 * 
 * 2. Calculate metrics:
 *    - FLOPS (floating-point operations per second)
 *    - Bandwidth utilization (bytes transferred / time)
 *    - Latency hiding (how much computation overlaps with communication)
 * 
 * 3. Use profiling tools:
 *    - MPI: mpiP, TAU, Scalasca, Score-P
 *    - OpenMP: Intel VTune, TAU
 *    - General: perf, gprof, valgrind
 * 
 * 4. Identify bottlenecks:
 *    - Is it compute-bound or communication-bound?
 *    - Is there load imbalance?
 *    - Are there synchronization hotspots?
 * 
 * =============================================================================
 * FURTHER READING:
 * =============================================================================
 * 
 * BOOKS:
 *   - "Using MPI" by Gropp, Lusk, Skjellum
 *   - "Parallel Programming in OpenMP" by Chandra et al.
 *   - "Introduction to High Performance Computing for Scientists and Engineers"
 *     by Hager and Wellein
 * 
 * ONLINE RESOURCES:
 *   - MPI Tutorial: https://mpitutorial.com/
 *   - OpenMP Tutorial: https://www.openmp.org/resources/tutorials-articles/
 *   - LLNL HPC Tutorials: https://hpc-tutorials.llnl.gov/
 * 
 * PAPERS:
 *   - "Communication Avoiding Algorithms" by Ballard, Demmel, Holtz, Schwartz
 *   - "Roofline Model" by Williams, Waterman, Patterson
 * 
 * =============================================================================
 */


