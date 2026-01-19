/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * =============================================================================
 * STENCIL_TEMPLATE_SERIAL.H - SERIAL STENCIL COMPUTATION HEADER (WITH OPENMP)
 * HYPERCOMMENTED VERSION - EDUCATIONAL PURPOSES
 * =============================================================================
 * 
 * This header file provides the core functions for SERIAL stencil computation
 * with OpenMP parallelization for shared-memory multi-threading.
 * 
 * KEY DIFFERENCES FROM PARALLEL VERSION:
 *   - No MPI (single process, single node)
 *   - Simpler data structures (plain arrays instead of plane_t structs)
 *   - No halo exchange (entire grid is in one memory space)
 *   - Includes border dissipation calculation (for non-periodic boundaries)
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
 * STENCIL COMPUTATION:
 *   update_plane() - Updates ALL grid points at once
 *     └─ Uses OpenMP to parallelize computation
 *     └─ Simple and efficient for shared-memory systems
 * 
 * ENERGY COMPUTATION:
 *   get_total_energy() - Parallel sum of all grid values
 *     └─ Uses OpenMP reduction for thread-safe summation
 *     └─ Verifies energy conservation
 *   
 *   get_border_dissipation() - Calculates energy loss at boundaries
 *     └─ Only relevant for non-periodic boundaries
 *     └─ Explains where energy goes (lost to environment)
 * 
 * =============================================================================
 * MEMORY LAYOUT:
 * =============================================================================
 * 
 * The 2D grid is stored in a 1D array in row-major order.
 * 
 * LOGICAL GRID (2D view):
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
 * GHOST CELLS IN SERIAL CODE:
 *   - Used for boundary conditions (periodic or non-periodic)
 *   - For non-periodic: ghost cells are always 0 (infinite heat sink)
 *   - For periodic: ghost cells copy from opposite edge
 * 
 * =============================================================================
 * OPENMP PARALLELIZATION:
 * =============================================================================
 * 
 * This serial code uses OpenMP for multi-threading:
 *   - update_plane(): Parallelized stencil computation
 *   - get_total_energy(): Parallelized sum with reduction
 * 
 * WHY OPENMP IN "SERIAL" CODE?
 *   - "Serial" means single MPI process (no distributed memory)
 *   - But we still use multiple CPU cores via OpenMP (shared memory)
 *   - Typical speedup: 7-8x on 8-core CPU
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
#include <float.h>    // FLT_MAX, DBL_MAX
#include <math.h>     // sqrt, fabs, etc.

// =============================================================================
// COORDINATE AND DIRECTION CONSTANTS
// =============================================================================

#define NORTH 0   // Neighbor in the +y direction (j increases)
#define SOUTH 1   // Neighbor in the -y direction (j decreases)
#define EAST  2   // Neighbor in the +x direction (i increases)
#define WEST  3   // Neighbor in the -x direction (i decreases)

#define SEND 0    // (Not used in serial, but kept for compatibility)
#define RECV 1    // (Not used in serial, but kept for compatibility)

#define OLD 0     // Index for old/current time level
#define NEW 1     // Index for new/next time level

#define _x_ 0     // Index for x-coordinate (columns)
#define _y_ 1     // Index for y-coordinate (rows)

// =============================================================================
// FUNCTION PROTOTYPES
// =============================================================================

// -----------------------------------------------------------------------------
// initialize: Set up simulation, parse arguments, allocate memory
// -----------------------------------------------------------------------------
// This initializes the entire simulation (serial version)
int initialize ( int      ,      // argc
		 char   **,     // argv
		 int    *,      // size[_x_] (grid width)
		 int     *,     // size[_y_] (grid height)
		 int     *,     // periodic (1 or 0)
		 int     *,     // iterations (number of time steps)
		 int   **,      // sources (array of source coordinates)
		 double  *,     // energy (amount to inject at each source)
		 double **,     // plane data (2D grid as 1D array)
                 int     *,     // nsources (number of sources)
                 int     *      // report_every (output frequency)
		 );

// -----------------------------------------------------------------------------
// memory_release: Free all dynamically allocated memory
// -----------------------------------------------------------------------------
int memory_release ( double *,   // plane data
                    int * );    // sources

// -----------------------------------------------------------------------------
// inject_energy: Add energy at specified source locations
// -----------------------------------------------------------------------------
// Parameters:
//   periodic: if 1, apply periodic boundary conditions
//   Nsources: number of energy sources
//   Sources: array of source coordinates (flat array: x0,y0,x1,y1,...)
//   energy: amount of energy to inject at each source
//   mysize: grid dimensions [width, height]
//   plane: grid to inject energy into
// Returns: 0 on success
extern int inject_energy ( const  int,
                           const int    ,
			   const int   *,
			   const double  ,
			   const int    [2],
                                 double * );

// -----------------------------------------------------------------------------
// update_plane: Update all grid points using 5-point stencil
// -----------------------------------------------------------------------------
// Parameters:
//   periodic: if 1, apply periodic boundary conditions
//   size: grid dimensions [width, height]
//   old: current grid state (time t)
//   new: output grid state (time t+dt)
// Returns: 0 on success
extern int update_plane ( const int       ,
			  const int    [2],
			  const double   *,
		                double   * );

// -----------------------------------------------------------------------------
// get_total_energy: Compute total energy in the grid
// -----------------------------------------------------------------------------
// Parameters:
//   size: grid dimensions [width, height]
//   plane: grid to compute energy from
//   energy: output total energy (sum of all grid values)
// Returns: 0 on success
extern int get_total_energy( const int     [2],
                             const double *,
                             double * );

// -----------------------------------------------------------------------------
// get_border_dissipation: Calculate energy lost at boundaries
// -----------------------------------------------------------------------------
// Parameters:
//   periodic: if 1, no dissipation (periodic BC)
//   size: grid dimensions [width, height]
//   plane: grid to analyze
//   dissipation: output energy dissipated at borders
// Returns: 0 on success
extern int get_border_dissipation( const int     periodic,
                                    const int     size[2],
                                    const double *plane,
                                    double       *dissipation );

// =============================================================================
// INLINE FUNCTION IMPLEMENTATIONS
// =============================================================================

/*
 * =============================================================================
 * INJECT_ENERGY: Add energy at source locations
 * =============================================================================
 * 
 * This function simulates heat sources on the grid. At each source location,
 * we add a fixed amount of energy.
 * 
 * DATA STRUCTURE:
 *   - Sources is a FLAT array: [x0, y0, x1, y1, x2, y2, ...]
 *   - Each source requires 2 integers: x-coordinate and y-coordinate
 *   - Total array size: 2 * Nsources
 * 
 * PERIODIC BOUNDARY CONDITIONS:
 *   - If a source is at the edge and we have periodic BC, we must also
 *     inject energy into the corresponding ghost cell
 *   - This ensures continuity across periodic boundaries
 * 
 * EXAMPLE:
 *   If we have 3 sources at (10,20), (30,40), (50,60):
 *   Sources = [10, 20, 30, 40, 50, 60]
 *   Nsources = 3
 */
inline int inject_energy ( const int     periodic,   // 1 if periodic BC, 0 otherwise
                           const int     Nsources,   // Number of energy sources
			   const int    *Sources,    // Array of source coords (flat: x0,y0,x1,y1,...)
			   const double  energy,     // Amount of energy to inject
			   const int     mysize[2],  // Grid dimensions [width, height]
                           double *plane )           // Grid to inject into
{
    // Macro for converting 2D index (i,j) to 1D array index
    // Note: mysize[_x_]+2 because we have ghost cells at i=0 and i=mysize[_x_]+1
   #define IDX( i, j ) ( (j)*(mysize[_x_]+2) + (i) )
   
    // Loop over all energy sources
    for (int s = 0; s < Nsources; s++) {
        
        // Extract source coordinates from flat array
        // Source s is at index 2*s (x) and 2*s+1 (y)
        int x = Sources[2*s];       // X-coordinate (column)
        int y = Sources[2*s+1];     // Y-coordinate (row)
        
        // Add energy at this source location
        // Note: += is used because multiple sources might overlap
        plane[IDX(x, y)] += energy;

        // Handle periodic boundary conditions
        // If source is at edge, also inject at opposite edge (wrap-around)
        if ( periodic )
            {
                // If source is at left edge (x=1), also inject at right ghost (x=xsize+1)
                if ( x == 1 )
                    plane[IDX(mysize[_x_]+1, y)] += energy;
                
                // If source is at right edge (x=xsize), also inject at left ghost (x=0)
                if ( x == mysize[_x_] )
                    plane[IDX(0, y)] += energy;
                
                // If source is at bottom edge (y=1), also inject at top ghost (y=ysize+1)
                if ( y == 1 )
                    plane[IDX(x, mysize[_y_]+1)] += energy;
                
                // If source is at top edge (y=ysize), also inject at bottom ghost (y=0)
                if ( y == mysize[_y_] )
                    plane[IDX(x, 0)] += energy;
            }
    }
   #undef IDX  // Clean up macro
    
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
 *   U_{i,j}^{n+1} = α·U_{i,j}^n + (1-α)/4·(sum of 4 neighbors)
 * 
 * WHERE:
 *   - U_{i,j}^n is the current value at grid point (i,j) at time n
 *   - U_{i,j}^{n+1} is the new value at the next time step
 *   - α (alpha) is a diffusion parameter (0 < α < 1)
 *   - The four neighbors are (i±1,j) and (i,j±1)
 * 
 * PHYSICAL INTERPRETATION:
 *   - α = 0.6 means 60% of energy stays at (i,j), 40% diffuses to neighbors
 *   - Each neighbor receives 10% (= 40%/4) of the diffused energy
 *   - This conserves total energy (in periodic BC)
 * 
 * NON-PERIODIC BOUNDARIES:
 *   - Ghost cells (i=0, i=xsize+1, j=0, j=ysize+1) are always 0
 *   - This creates an "infinite heat sink" at boundaries
 *   - Energy is lost when it diffuses to ghost cells
 *   - Total energy decreases over time (unless sources compensate)
 * 
 * OPENMP PARALLELIZATION:
 *   - The outer loop (j) is parallelized across threads
 *   - Each thread computes a subset of rows
 *   - No race conditions: each thread writes to different memory locations
 *   - Typical speedup: 7-8x on 8-core CPU
 */
inline int update_plane ( const int     periodic,   // 1 if periodic BC, 0 otherwise
                          const int     size[2],    // Grid dimensions [width, height]
			  const double *old    ,    // Current grid state (time t)
                                double *new    )    // Output grid state (time t+dt)
/*
 * Calculate the new energy values using 5-point stencil.
 * The old plane contains the current data, the new plane
 * will store the updated data.
 *
 * NOTE: In serial, we compute the entire grid in one function call.
 *       In parallel (MPI), this would be split into subdomains.
 */
{
    // Extract grid dimensions for convenience
    // 'register' keyword: hint to compiler to keep in CPU register (faster)
    register const int fxsize = size[_x_]+2;  // Full width (with ghost cells)
    register const int xsize = size[_x_];     // Interior width (without ghosts)
    register const int ysize = size[_y_];     // Interior height (without ghosts)
    
    // Macro for 2D to 1D index conversion (row-major layout)
   #define IDX( i, j ) ( (j)*fxsize + (i) )

    // COMPILER OPTIMIZATION HINTS:
    // 
    // HINT: You may attempt to manually unroll the loop or ask the compiler to do it
    // Example:
    // #pragma GCC unroll 4
    //
    // Loop unrolling processes multiple iterations per loop cycle:
    //   - Reduces loop overhead (fewer branch instructions)
    //   - Enables better instruction-level parallelism (ILP)
    //   - Can improve performance by 10-20%
    //   - Trade-off: larger code size, may hurt instruction cache
    //
    // HINT: This loop is a good candidate for OpenMP parallelization
    //       (already implemented below!)
    
    // =============================================================
    // OPENMP PARALLELIZATION OF STENCIL COMPUTATION
    // =============================================================
    // #pragma omp parallel for schedule(static)
    //   - Parallelizes outer loop (j) across threads
    //   - Inner loop (i) remains sequential for cache efficiency
    //   - Static scheduling: iterations divided evenly at compile time
    //
    // WHY OPENMP?
    //   - Modern CPUs have 8-64 cores per socket
    //   - Without OpenMP, only 1 core is used (huge waste!)
    //   - With OpenMP, all cores work in parallel
    //   - Near-linear speedup: 8 cores ≈ 7-8x faster
    //
    // PERFORMANCE DETAILS:
    //   - Row-major memory layout matches loop order (cache-friendly)
    //   - Each thread processes consecutive rows (good spatial locality)
    //   - No synchronization needed inside loop (very efficient!)
    //   - Only implicit barrier at end (threads wait for each other)
    // =============================================================
    
    // Diffusion parameters
    double alpha = 0.6;                      // 60% stays at current location
    double constant =  (1-alpha) / 4.0;      // 10% goes to each of 4 neighbors
    
    // OpenMP parallelization of the stencil computation
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= ysize; j++)        // Loop over rows (excluding ghost cells)
        for ( int i = 1; i <= xsize; i++)   // Loop over columns (excluding ghost cells)
            {
                // =============================================================
                // FIVE-POINT STENCIL FORMULA
                // =============================================================
                //
                // Compute new value at (i,j) based on:
                //   - Current value at (i,j): contributes α·old[i,j]
                //   - West neighbor (i-1,j): contributes β·old[i-1,j]
                //   - East neighbor (i+1,j): contributes β·old[i+1,j]
                //   - South neighbor (i,j-1): contributes β·old[i,j-1]
                //   - North neighbor (i,j+1): contributes β·old[i,j+1]
                //   where β = (1-α)/4
                //
                // ALTERNATIVE STENCIL (COMMENTED OUT BELOW):
                //   There's a more sophisticated stencil that uses adaptive
                //   time-stepping based on diffusivity. However, that requires
                //   careful stability analysis and is more complex.
                //   The simple stencil above is stable and conserves energy.
                // =============================================================

                // Simpler stencil with no explicit diffusivity
                // Always conserves the smoothed quantity
                // Alpha here mimics how "easily" the heat travels
                // Higher alpha (closer to 1) = slower diffusion
                // Lower alpha (closer to 0) = faster diffusion
                
                double result = old[ IDX(i,j) ] * alpha           // 60% stays here
                              + (old[IDX(i-1, j)]                 // 10% from west
                              + old[IDX(i+1, j)]                  // 10% from east
                              + old[IDX(i, j-1)]                  // 10% from south
                              + old[IDX(i, j+1)])                 // 10% from north
                              * constant;

                /*
                 * =============================================================
                 * ALTERNATIVE STENCIL (NOT USED)
                 * =============================================================
                 * 
                 * This is a more physically accurate stencil derived from
                 * 3-point 2nd order finite difference approximation.
                 * 
                 * It uses the heat equation: ∂u/∂t = α∇²u
                 * where ∇²u ≈ (u[i-1,j] + u[i+1,j] - 2u[i,j]) / Δx²
                 *           + (u[i,j-1] + u[i,j+1] - 2u[i,j]) / Δy²
                 * 
                 * However, this requires adaptive time-stepping to ensure
                 * numerical stability (CFL condition). The explicit method
                 * is conditionally stable: Δt ≤ Δx²/(4α)
                 * 
                 * For simplicity, we use the simpler stencil above, which
                 * is unconditionally stable for α ≥ 0.5.

               #define alpha_guess 0.5     // Mimic the heat diffusivity

                double alpha = alpha_guess;
                double sum = old[IDX(i,j)];
                
                int   done = 0;
                do
                    {                
                        // Compute Laplacian in x-direction
                        double sum_i = alpha * (old[IDX(i-1, j)] + old[IDX(i+1, j)] - 2*sum);
                        
                        // Compute Laplacian in y-direction
                        double sum_j = alpha * (old[IDX(i, j-1)] + old[IDX(i, j+1)] - 2*sum);
                        
                        // New value = old value + time derivative
                        result = sum + ( sum_i + sum_j);
                        
                        // Check for stability: change should not be too large
                        double ratio = fabs((result-sum)/(sum!=0? sum : 1.0));
                        
                        // Accept if change is reasonable and result is physical
                        done = ( (ratio < 2.0) && (result >= 0) );
                        
                        // If unstable, reduce time step (halve alpha)
                        alpha /= 2;
                    }
                while ( !done );
                */

                // Write result to new grid
                new[ IDX(i,j) ] = result;
                
            }

    // =============================================================
    // PERIODIC BOUNDARY CONDITIONS
    // =============================================================
    // After computing interior points, update ghost cells for periodic BC.
    // Ghost cells at one edge copy from opposite edge (wrap-around effect).
    //
    // NOTE: In distributed memory (MPI), periodic BC is handled via
    //       halo exchange between processes. Here, we handle it locally.
    // =============================================================
    
    if ( periodic )
        /*
         * Propagate boundaries if they are periodic
         *
         * NOTE: When is this needed in distributed memory, if any?
         *   - If N[_x_] == 1 (single process in x-direction), handle x-periodic locally
         *   - If N[_x_] > 1 (multiple processes in x), MPI handles x-periodic via halo
         *   - Similarly for y-direction
         */
        {
            // Propagate y-boundaries (top/bottom wrap-around)
            for ( int i = 1; i <= xsize; i++ )
                {
                    // Bottom ghost (j=0) copies from top edge (j=ysize)
                    new[ IDX(i, 0) ] = new[ IDX(i, ysize) ];
                    
                    // Top ghost (j=ysize+1) copies from bottom edge (j=1)
                    new[ IDX(i, ysize+1) ] = new[ IDX(i, 1) ];
                }
            
            // Propagate x-boundaries (left/right wrap-around)
            for ( int j = 1; j <= ysize; j++ )
                {
                    // Left ghost (i=0) copies from right edge (i=xsize)
                    new[ IDX( 0, j) ] = new[ IDX(xsize, j) ];
                    
                    // Right ghost (i=xsize+1) copies from left edge (i=1)
                    new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ];
                }
        }
    
    return 0;  // Success

   #undef IDX  // Clean up macro
}

 

/*
 * =============================================================================
 * GET_TOTAL_ENERGY: Parallel sum of all grid values with OpenMP reduction
 * =============================================================================
 * 
 * Computes the total energy in the system by summing all grid point values.
 * Essential for verifying energy conservation.
 * 
 * ENERGY CONSERVATION:
 *   - Periodic BC: total energy should be constant (unless sources/sinks)
 *   - Non-periodic BC: energy leaks at boundaries, total decreases over time
 * 
 * NUMERICAL PRECISION:
 *   - We use 'double' (64-bit) by default
 *   - Can enable 'long double' (80-bit or 128-bit) for large grids
 *   - Reduces roundoff errors when summing millions of values
 */
inline int get_total_energy( const int     size[2],    // Grid dimensions
                             const double *plane,      // Grid to analyze
                                   double *energy )    // Output: total energy
/*
 * NOTE: This routine is a good candidate for OpenMP parallelization
 *       (already implemented below!)
 */
{
    // Extract grid dimensions
    register const int xsize = size[_x_];  // Width (interior points only)
    
    // Macro for 2D to 1D index conversion
   #define IDX( i, j ) ( (j)*(xsize+2) + (i) )

   // Choose precision: long double for high accuracy, double for speed
   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;  // 80-bit or 128-bit (platform-dependent)
   #else
    double totenergy = 0;       // 64-bit (standard double precision)
   #endif

    // COMPILER OPTIMIZATION HINTS:
    // HINT: You may attempt to manually unroll the loop or ask the compiler to do it
    // Example: #pragma GCC unroll 4
    // (See comments in update_plane for details on loop unrolling)
    
    // =============================================================
    // OPENMP PARALLELIZATION FOR ENERGY CALCULATION
    // =============================================================
    // #pragma omp parallel for schedule(static) reduction(+:totenergy)
    //   - Each thread sums a subset of grid points
    //   - reduction(+:totenergy) ensures thread-safe summation
    //   - Compiler handles synchronization automatically
    //
    // See stencil_hypercommented.h for detailed explanation of
    // OpenMP reduction and why it's better than alternatives.
    // =============================================================
    
    // OpenMP parallelization for energy calculation
    #pragma omp parallel for schedule(static) reduction(+:totenergy)
    for ( int j = 1; j <= size[_y_]; j++ )      // Loop over rows (interior only)
        for ( int i = 1; i <= size[_x_]; i++ )  // Loop over columns (interior only)
            // Accumulate energy (sum all grid values)
            // Note: We only sum interior points, not ghost cells
            totenergy += plane[ IDX(i, j) ];
    
   #undef IDX  // Clean up macro

    // Convert result to double (in case we used long double)
    *energy = (double)totenergy;
    return 0;  // Success
}

/*
 * =============================================================================
 * GET_BORDER_DISSIPATION: Calculate energy lost at boundaries
 * =============================================================================
 * 
 * This function calculates how much energy is dissipated at the boundaries
 * for non-periodic boundary conditions.
 * 
 * WHAT IS BORDER DISSIPATION?
 *   - In non-periodic BC, ghost cells are always 0 (infinite heat sink)
 *   - Border points exchange energy with these 0-valued ghost cells
 *   - Energy flowing to ghost cells is "lost" (dissipated to environment)
 *   - This function quantifies that energy loss
 * 
 * WHY IS THIS USEFUL?
 *   - Explains where energy goes in non-periodic simulations
 *   - Verifies energy conservation: E_new = E_old + E_injected - E_dissipated
 *   - Helps debug energy balance issues
 * 
 * HOW IS DISSIPATION CALCULATED?
 *   - Each border point contributes β·U[border] to its ghost neighbor
 *   - where β = (1-α)/4 is the diffusion coefficient
 *   - Ghost neighbor has value 0, so this energy is lost
 *   - Sum over all border points to get total dissipation
 * 
 * EXAMPLE:
 *   - Border point at (i,1) has value 100
 *   - It contributes β·100 to ghost at (i,0) which has value 0
 *   - Dissipation from this point = β·100
 *   - Total dissipation = sum over all border points
 * 
 * PERIODIC BC:
 *   - No dissipation! Energy wraps around
 *   - This function returns 0 for periodic BC
 */
inline int get_border_dissipation( const int     periodic,      // 1 if periodic BC, 0 otherwise
                                    const int     size[2],       // Grid dimensions
                                    const double *plane,         // Grid to analyze
                                    double       *dissipation )  // Output: energy dissipated
/*
 * Calculate energy dissipation at borders for non-periodic boundaries.
 * The dissipation occurs because border cells exchange energy with halo cells (which are 0).
 * This function calculates the energy flow from border cells to the halo.
 */
{
    // No dissipation for periodic boundaries (energy wraps around)
    if (periodic) {
        *dissipation = 0.0;
        return 0;
    }

    // Extract grid dimensions
    register const int xsize = size[_x_];   // Width (interior points)
    register const int ysize = size[_y_];   // Height (interior points)
    register const int fxsize = xsize + 2;  // Full width (with ghost cells)
    
    // Macro for 2D to 1D index conversion
   #define IDX( i, j ) ( (j)*fxsize + (i) )

    // Diffusion parameters (same as update_plane)
    double alpha = 0.6;                              // 60% stays at current location
    double beta = (1.0 - alpha) / 4.0;               // 10% goes to each neighbor
    
    // Beta is the coefficient for neighbor contribution
    // When a border point contributes to a ghost cell (value 0),
    // the energy beta * border_value is lost
    
    double total_dissipation = 0.0;

    // =============================================================
    // NORTH BORDER (j=1): Energy flows to halo at j=0 (which is 0)
    // =============================================================
    // Bottom border points lose energy to bottom ghost cells
    for (int i = 1; i <= xsize; i++) {
        // Energy that would flow from cell (i,1) to halo (i,0)
        // In the stencil formula, cell (i,1) contributes beta * plane[i,1] to cell (i,0)
        // Since cell (i,0) is 0 (ghost), this energy is lost
        total_dissipation += plane[IDX(i, 1)] * beta;
    }

    // =============================================================
    // SOUTH BORDER (j=ysize): Energy flows to halo at j=ysize+1
    // =============================================================
    // Top border points lose energy to top ghost cells
    for (int i = 1; i <= xsize; i++) {
        // Energy that would flow from cell (i,ysize) to halo (i,ysize+1)
        total_dissipation += plane[IDX(i, ysize)] * beta;
    }

    // =============================================================
    // WEST BORDER (i=1): Energy flows to halo at i=0
    // =============================================================
    // Left border points lose energy to left ghost cells
    for (int j = 1; j <= ysize; j++) {
        // Energy that would flow from cell (1,j) to halo (0,j)
        total_dissipation += plane[IDX(1, j)] * beta;
    }

    // =============================================================
    // EAST BORDER (i=xsize): Energy flows to halo at i=xsize+1
    // =============================================================
    // Right border points lose energy to right ghost cells
    for (int j = 1; j <= ysize; j++) {
        // Energy that would flow from cell (xsize,j) to halo (xsize+1,j)
        total_dissipation += plane[IDX(xsize, j)] * beta;
    }

    // =============================================================
    // CORNER EFFECTS
    // =============================================================
    // Note: Corner points are counted twice in the loops above:
    //   - Corner (1,1) counted in NORTH loop and WEST loop
    //   - Corner (xsize,1) counted in NORTH loop and EAST loop
    //   - Corner (1,ysize) counted in SOUTH loop and WEST loop
    //   - Corner (xsize,ysize) counted in SOUTH loop and EAST loop
    //
    // This is correct! Corner points have TWO ghost neighbors (not just one).
    // For example, corner (1,1) loses energy to both (0,1) and (1,0).
    // So corner dissipation should be counted twice.
    //
    // Actually, corners have MORE than 2 ghost neighbors (they have 3):
    //   - Corner (1,1) neighbors: (0,1), (1,0), and diagonal (0,0)
    // But in the 5-point stencil, we only consider 4 neighbors (no diagonals).
    // So we only count dissipation to (0,1) and (1,0), not to (0,0).
    // =============================================================

   #undef IDX  // Clean up macro

    // Return total dissipation
    *dissipation = total_dissipation;
    return 0;  // Success
}
                            

