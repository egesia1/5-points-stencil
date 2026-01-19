/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * =============================================================================
 * STENCIL.H - PARALLEL STENCIL COMPUTATION HEADER (MPI + OPENMP)
 * =============================================================================
 * 
 * This header file provides the core functions for parallel stencil computation
 * using a hybrid MPI+OpenMP approach.
 * 
 * =============================================================================
 * FUNCTION OVERVIEW:
 * =============================================================================
 * 
 * ENERGY INJECTION:
 *   inject_energy() - Adds energy at source locations
 * 
 * STENCIL COMPUTATION (TWO STRATEGIES):
 *   
 *   STRATEGY 1: Simple (for serial/OpenMP-only):
 *     update_plane() - Updates ALL grid points at once
 *     ├─ Uses OpenMP to parallelize computation
 *     └─ Simple but no communication-computation overlap
 *   
 *   STRATEGY 2: Split (for MPI+OpenMP with overlap):
 *     update_interior() - Updates INTERIOR points (can compute during MPI)
 *     └─ Called BEFORE MPI_Waitall
 *     
 *     update_borders() - Updates BORDER points (needs halo data)
 *     └─ Called AFTER MPI_Waitall
 *     
 *     This split enables COMMUNICATION-COMPUTATION OVERLAP:
 *       1. Start MPI_Isend/Irecv (non-blocking)
 *       2. Compute interior (while MPI transfers halo)
 *       3. MPI_Waitall (wait for halo data)
 *       4. Unpack halo data
 *       5. Compute borders (using fresh halo data)
 * 
 * ENERGY COMPUTATION:
 *   get_total_energy() - Parallel sum of all grid values
 *     └─ Uses OpenMP reduction for thread-safe summation
 * 
 * =============================================================================
 * OPENMP USAGE SUMMARY:
 * =============================================================================
 * 
 * All stencil computation functions use OpenMP parallelization:
 *   - update_plane(): Full grid parallelized
 *   - update_interior(): Interior points parallelized
 *   - update_borders(): Border points parallelized (2 separate loops)
 *   - get_total_energy(): Reduction for parallel sum
 * 
 * KEY PRAGMA: #pragma omp parallel for schedule(static)
 *   - Distributes loop iterations evenly across threads
 *   - Static scheduling for uniform workload and good cache locality
 *   - Thread-safe: no data races (each thread writes to different locations)
 * 
 * =============================================================================
 * MPI INTEGRATION:
 * =============================================================================
 * 
 * The split update strategy (interior/borders) is designed for MPI:
 *   - Interior points: don't need halo data from neighbors
 *   - Border points: need halo data from neighbors
 * 
 * This split enables overlapping MPI communication with OpenMP computation,
 * which is critical for strong scaling on distributed systems.
 * 
 * =============================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>

#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

// MPI Tags
#define TAG_N 0
#define TAG_S 1
#define TAG_E 2
#define TAG_W 3

typedef unsigned int uint;

typedef uint    vec2_t[2];
typedef double *restrict buffers_t[2][4];  // [SEND/RECV][NORTH/SOUTH/EAST/WEST]

typedef struct {
    double   * restrict data;
    vec2_t     size;
} plane_t;

// Function prototypes
extern int inject_energy ( const int      ,
                          const int      ,
			  const vec2_t  *,
			  const double   ,
                          plane_t *,
                          const vec2_t   );

extern int update_plane ( const int      ,
                         const vec2_t   ,
                         const plane_t *,
                               plane_t * );

extern int update_interior( const plane_t *,
                           plane_t * );

extern int update_borders( const int      ,
                          const vec2_t   ,
                          const plane_t *,
                          plane_t * );

extern int get_total_energy( plane_t *,
                            double  * );

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

int memory_release (plane_t   *, buffers_t * );

int output_energy_stat ( int      ,
                        plane_t *,
                        double   ,
                        int      ,
                        MPI_Comm *);

// Inline function implementations
inline int inject_energy ( const int      periodic,
                          const int      Nsources,
			  const vec2_t  *Sources,
			  const double   energy,
                          plane_t *plane,
                          const vec2_t   N
                          )
{
    const uint register sizex = plane->size[_x_]+2;
    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*sizex + (i) )
    for (int s = 0; s < Nsources; s++)
        {
            int x = Sources[s][_x_];
            int y = Sources[s][_y_];
            
            data[ IDX(x,y) ] += energy;
            
            if ( periodic )
                {
                    // Propagate boundaries if needed for periodic conditions
                    if ( N[_x_] == 1 )
                        {
                            if ( x == 1 )
                                data[ IDX(plane->size[_x_]+1, y) ] += energy;
                            if ( x == plane->size[_x_] )
                                data[ IDX(0, y) ] += energy;
                        }
                    
                    if ( N[_y_] == 1 )
                        {
                            if ( y == 1 )
                                data[ IDX(x, plane->size[_y_]+1) ] += energy;
                            if ( y == plane->size[_y_] )
                                data[ IDX(x, 0) ] += energy;
                        }
                }                
        }
 #undef IDX
    
  return 0;
}

/*
 * =============================================================================
 * UPDATE_PLANE: Full 5-point stencil computation with OpenMP
 * =============================================================================
 * 
 * This function updates all grid points using the 5-point stencil.
 * It's used for simple serial/OpenMP implementations.
 * 
 * In the parallel (MPI) version, this is replaced by update_interior() 
 * and update_borders() for communication-computation overlap.
 */
inline int update_plane ( const int      periodic, 
                         const vec2_t   N,         // the grid of MPI tasks
                         const plane_t *oldplane,
                               plane_t *newplane
                         )
    
{
    uint register fxsize = oldplane->size[_x_]+2;
    uint register fysize = oldplane->size[_y_]+2;
    
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];
    
   #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;
    
    // =============================================================
    // OPENMP PARALLELIZATION OF STENCIL COMPUTATION
    // =============================================================
    // #pragma omp parallel for schedule(static)
    //   - Parallelizes outer loop (j) across threads
    //   - Inner loop (i) remains sequential for cache efficiency
    //   - Static scheduling: iterations divided evenly at compile time
    //
    // PARALLELIZATION CORRECTNESS:
    //   - Each (i,j) updated by exactly one thread (no race conditions)
    //   - All threads READ from old[] (shared, read-only)
    //   - Each thread WRITES to different locations in new[]
    //
    // PERFORMANCE:
    //   - Row-major order matches C memory layout
    //   - Good spatial locality for cache utilization
    //   - Minimal synchronization overhead
    // =============================================================
    
    #pragma omp parallel for schedule(static)
    for (uint j = 1; j <= ysize; j++)
        for ( uint i = 1; i <= xsize; i++)
            {
                // Five-points stencil formula (energy-conserving)
                // U_{i,j}^{n+1} = α·U_{i,j}^n + (1-α)/4·(sum of 4 neighbors)
                double alpha = 0.6;
                double constant = (1.0 - alpha) / 4.0;
                double result = old[ IDX(i,j) ] * alpha
                                + ( old[IDX(i-1, j)]
                                  + old[IDX(i+1, j)]
                                  + old[IDX(i, j-1)]
                                  + old[IDX(i, j+1)] ) * constant;
                
                new[ IDX(i,j) ] = result;
            }

    if ( periodic )
        {
            if ( N[_x_] == 1 )
                {
                    // Propagate x-boundaries for periodic conditions
                    for ( int i = 1; i <= xsize; i++ )
                        {
                            new[ IDX(i, 0) ] = new[ IDX(i, ysize) ];
                            new[ IDX(i, ysize+1) ] = new[ IDX(i, 1) ];
                        }
                }
  
            if ( N[_y_] == 1 ) 
                {
                    // Propagate y-boundaries for periodic conditions
                    for ( int j = 1; j <= ysize; j++ )
                        {
                            new[ IDX(0, j) ] = new[ IDX(xsize, j) ];
                            new[ IDX(xsize+1, j) ] = new[ IDX(1, j) ];
                        }
                }
        }

 #undef IDX
  return 0;
}

/*
 * =============================================================================
 * GET_TOTAL_ENERGY: Parallel sum of all grid values with OpenMP reduction
 * =============================================================================
 * 
 * This function computes the total energy in the system by summing all
 * grid point values. It's essential for energy conservation verification.
 * 
 * Uses OpenMP reduction for thread-safe parallel summation.
 */
inline int get_total_energy( plane_t *plane,
                            double  *energy )
{
    const int register xsize = plane->size[_x_];
    const int register ysize = plane->size[_y_];
    const int register fsize = xsize+2;

    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*fsize + (i) )

   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif

    // =============================================================
    // OPENMP REDUCTION FOR PARALLEL SUM
    // =============================================================
    // #pragma omp parallel for schedule(static) reduction(+:totenergy)
    //   - Each thread maintains a PRIVATE copy of totenergy
    //   - Each thread sums a subset of grid points
    //   - At the end, all private copies are SUMMED together
    //
    // WHY REDUCTION?
    //   - Simple += would create DATA RACE (multiple threads writing same variable)
    //   - Reduction provides thread-safe way to compute aggregate values
    //   - Compiler handles synchronization automatically
    //
    // PERFORMANCE:
    //   - Very efficient: each thread works independently
    //   - Only synchronizes once at the end
    //   - Near-linear speedup for large grids
    //
    // ALTERNATIVE APPROACHES (NOT USED):
    //   - Manual atomic operations: #pragma omp atomic (too slow)
    //   - Critical sections: too much synchronization overhead
    //   - Thread-local arrays: more complex, similar performance
    // =============================================================
    
    #pragma omp parallel for schedule(static) reduction(+:totenergy)
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];

   #undef IDX

    *energy = (double)totenergy;
    return 0;
}

/*
 * =============================================================================
 * UPDATE_INTERIOR: Stencil computation for INTERIOR points only
 * =============================================================================
 * 
 * This function updates only the INTERIOR points of the domain, skipping
 * the first and last row/column (the border points).
 * 
 * CRITICAL FOR COMMUNICATION-COMPUTATION OVERLAP:
 * - Interior points DON'T need halo data from neighboring processes
 * - Can be computed WHILE halo exchange is in progress (MPI communication)
 * - This overlap reduces idle time and improves scalability
 * 
 * Used in conjunction with update_borders() for efficient parallel execution.
 */
inline int update_interior( const plane_t *oldplane, plane_t *newplane ) {
    const uint xsize = oldplane->size[_x_];
    const uint ysize = oldplane->size[_y_];
    const uint fxsize = xsize+2;

    #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    const double alpha = 0.6;
    const double constant = (1-alpha) / 4.0;
    
    // =============================================================
    // OPENMP PARALLELIZATION OF INTERIOR POINTS
    // =============================================================
    // #pragma omp parallel for schedule(static)
    //   - Parallelizes computation across threads
    //   - Each thread updates a subset of interior rows
    //
    // LOOP BOUNDS:
    //   - j: from 2 to ysize-1 (skip first/last row = borders)
    //   - i: from 2 to xsize-1 (skip first/last column = borders)
    //
    // WHY SKIP BORDERS?
    //   - Border points (j=1, j=ysize, i=1, i=xsize) need halo data
    //   - Halo data comes from neighboring MPI processes
    //   - While MPI is transferring halo, we compute interior points
    //   - After MPI completes, update_borders() handles border points
    //
    // COMMUNICATION-COMPUTATION OVERLAP:
    //   1. Start non-blocking MPI (Isend/Irecv)
    //   2. Call this function (compute interior)
    //   3. Wait for MPI to complete (Waitall)
    //   4. Call update_borders() (compute borders with fresh halo)
    //
    // PERFORMANCE IMPACT:
    //   - Reduces wall-clock time by overlapping communication and computation
    //   - Critical for strong scaling (as problem size per process shrinks)
    //   - Can hide most or all communication latency for large enough domains
    // =============================================================
    
    #pragma omp parallel for schedule(static)
    for (uint j = 2; j <= ysize-1; j++)  // Skip first and last row (borders)
        for (uint i = 2; i <= xsize-1; i++)  // Skip first and last column (borders)
            {
                double result = old[ IDX(i,j) ] * alpha
                                + ( old[IDX(i-1, j)]
                                  + old[IDX(i+1, j)]
                                  + old[IDX(i, j-1)]
                                  + old[IDX(i, j+1)] ) * constant;
                
                new[ IDX(i,j) ] = result;
            }

    #undef IDX
    return 0;
}

/*
 * =============================================================================
 * UPDATE_BORDERS: Stencil computation for BORDER points only
 * =============================================================================
 * 
 * This function updates only the BORDER points of the domain:
 * - First and last row (j=1, j=ysize)
 * - First and last column (i=1, i=xsize)
 * 
 * MUST BE CALLED AFTER:
 * 1. Halo exchange is complete (MPI_Waitall)
 * 2. Halo data is unpacked into ghost cells
 * 
 * CRITICAL FOR COMMUNICATION-COMPUTATION OVERLAP:
 * - Border points NEED halo data from neighboring processes
 * - Can only be computed AFTER halo exchange completes
 * - Called after update_interior() to complete the stencil computation
 */
inline int update_borders( const int periodic, 
                          const vec2_t N,
                          const plane_t *oldplane,
                          plane_t *newplane ) {
    const uint xsize = oldplane->size[_x_];
    const uint ysize = oldplane->size[_y_];
    const uint fxsize = xsize+2;

    #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    const double alpha = 0.6;
    const double constant = (1-alpha) / 4.0;
    
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
    //   - Loop from i=2 to xsize-1 (EXCLUDE CORNERS)
    //   - Corners are handled by the vertical border loop
    //
    // WHY AFTER MPI_Waitall?
    //   - These points use halo data at j=0 (from NORTH neighbor)
    //   - And halo data at j=ysize+1 (from SOUTH neighbor)
    //   - Must wait for MPI to complete before accessing halo data
    // =============================================================
    
    #pragma omp parallel for schedule(static)
    for ( i = 2; i <= xsize-1; i++ ) { // exclude corners
        // Top border (j=1)
        center = old[ IDX(i,1) ];
        neighbors = old[IDX(i-1, 1)] + old[IDX(i+1, 1)] + old[IDX(i, 0)] + old[IDX(i, 2)];
        new[ IDX(i,1) ] = center * alpha + neighbors * constant;

        // Bottom border (j=ysize)
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
    //   - Loop from j=1 to ysize (INCLUDES CORNERS)
    //   - This ensures all 4 corners are updated
    //
    // WHY AFTER MPI_Waitall?
    //   - These points use halo data at i=0 (from WEST neighbor)
    //   - And halo data at i=xsize+1 (from EAST neighbor)
    //   - Must wait for MPI to complete before accessing halo data
    // =============================================================
    
    #pragma omp parallel for schedule(static)
    for ( j = 1; j <= ysize; j++ ) {
        // Left border (i=1)
        center = old[ IDX(1,j) ];
        neighbors = old[IDX(0, j)] + old[IDX(2, j)] + old[IDX(1, j-1)] + old[IDX(1, j+1)];
        new[ IDX(1,j) ] = center * alpha + neighbors * constant;

        // Right border (i=xsize)
        center = old[ IDX(xsize,j) ];
        neighbors = old[IDX(xsize-1, j)] + old[IDX(xsize+1, j)] + old[IDX(xsize, j-1)] + old[IDX(xsize, j+1)];
        new[ IDX(xsize,j) ] = center * alpha + neighbors * constant;
    }

    // Handle periodic boundaries
    if ( periodic ) {
        if ( N[_x_] == 1 ) {
            for ( j = 1; j <= ysize; j++ ) {
                new[ IDX( 0, j) ]       = new[ IDX(xsize, j) ];
                new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ];
            }
        }

        if ( N[_y_] == 1 ) {
            for ( i = 1; i <= xsize; i++ ) {
                new[ IDX( i, 0 ) ]       = new[ IDX(i, ysize) ];
                new[ IDX( i, ysize+1) ] = new[ IDX(i, 1) ];
            }
        }
    }

    #undef IDX
    return 0;
}
