/*
 * =============================================================================
 * HYPER-COMMENTED SERIAL STENCIL COMPUTATION WITH OPENMP
 * =============================================================================
 * 
 * This file implements a serial version of the 5-points stencil computation
 * with OpenMP parallelization for shared-memory systems.
 * 
 * KEY CONCEPTS FROM ASSIGNMENT SLIDES:
 * 
 * 1. STENCIL COMPUTATION:
 *    - Each grid point is updated based on its neighbors
 *    - 5-point stencil: center + 4 neighbors (North, South, East, West)
 *    - Heat equation discretization: dT/dt = α∇²T
 * 
 * 2. OPENMP PARALLELIZATION:
 *    - Shared-memory parallelization using OpenMP pragmas
 *    - Parallel loops over grid points
 *    - Static scheduling for load balancing
 * 
 * 3. ENERGY CONSERVATION:
 *    - Total energy should be conserved in the system
 *    - Energy injection at source points
 *    - Energy diffusion through the stencil operation
 * 
 * 4. BOUNDARY CONDITIONS:
 *    - Periodic: grid wraps around (infinite domain)
 *    - Non-periodic: fixed boundaries at domain edges
 *
 * =============================================================================
 * OPENMP PARALLELIZATION DETAILS
 * =============================================================================
 * 
 * This implementation uses OpenMP in THREE key locations:
 * 
 * 1. STENCIL COMPUTATION (main computational kernel):
 *    Location: Main iteration loop, line ~159
 *    Pragma: #pragma omp parallel for schedule(static)
 *    Purpose: Parallelize the 5-point stencil computation across grid rows
 *    Strategy: Each thread processes a subset of rows (j indices)
 *    Benefits: Distributes main computational work across all threads
 * 
 * 2. DUMP FUNCTION - MIN/MAX COMPUTATION:
 *    Location: dump() function, line ~596
 *    Pragma: #pragma omp parallel for schedule(static) reduction(min:_min_) reduction(max:_max_)
 *    Purpose: Parallel computation of minimum and maximum values
 *    Strategy: Each thread computes local min/max, combined at the end
 *    Benefits: Speeds up min/max computation using OpenMP reduction
 * 
 * KEY OPENMP CONCEPTS USED:
 * 
 * A. #pragma omp parallel for
 *    - Creates a team of threads to execute loop iterations in parallel
 *    - Each thread gets a subset of iterations to execute
 *    - Implicit barrier at the end of the loop
 * 
 * B. schedule(static)
 *    - Divides iterations evenly among threads at compile time
 *    - Best for loops with uniform workload per iteration
 *    - Minimal runtime overhead (no dynamic load balancing)
 *    - Better cache locality (threads work on contiguous data)
 * 
 * C. reduction(operation:variable)
 *    - Each thread maintains a private copy of the variable
 *    - At the end, all private copies are combined using the operation
 *    - Operations: min, max, +, *, &, |, ^, &&, ||
 *    - Thread-safe way to compute aggregate values in parallel
 * 
 * THREAD SAFETY AND DATA RACES:
 *    - NO DATA RACES in stencil computation:
 *      * All threads READ from planes[current] (shared, read-only)
 *      * Each thread WRITES to DIFFERENT locations in planes[!current]
 *      * Each (i,j) is updated by exactly ONE thread
 * 
 * PERFORMANCE CONSIDERATIONS:
 *    - Only outer loop (j) is parallelized for better cache utilization
 *    - Inner loop (i) remains sequential to exploit spatial locality
 *    - This follows C's row-major memory layout
 *    - Static scheduling minimizes overhead for uniform workload
 * 
 * SCALABILITY:
 *    - Expected speedup: near-linear up to ~8-16 threads
 *    - Bottleneck: memory bandwidth on large grids
 *    - Amdahl's law limits: initialization and I/O are serial
 */

#define _XOPEN_SOURCE
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include "stencil_template_serial.h"
#include <float.h>

typedef unsigned int uint;

// Function declarations
int dump ( const double *, const int [2], const char *, double *, double * );

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
  // =================================================================
  // VARIABLE DECLARATIONS
  // =================================================================
  
  int  Niterations;        // Number of time steps to simulate
  int  periodic;           // Flag for periodic boundary conditions (0=no, 1=yes)
  int  S[2];               // Grid dimensions [x, y]
  
  int     Nsources;        // Total number of heat sources in the system
  int    *Sources;         // Array of source coordinates [x,y] pairs
  double  energy_per_source; // Energy injected per source per iteration

  double *planes[2];       // Two planes: OLD (current state) and NEW (updated state)
  
  double injected_heat = 0;       // Total heat injected so far
  int injection_frequency;        // Frequency of energy injection
  int output_energy_at_steps = 0; // Flag to print energy statistics at each step
  int test = 0;  // whether we are running a scalability test
  test = getenv("TEST_TYPE") != NULL;
   
  // =================================================================
  // INITIALIZATION PHASE
  // =================================================================
  
  /* argument checking and setting */
  // Parse command line arguments and set up the simulation parameters
  // This includes: grid size, number of iterations, heat sources, etc.
  initialize ( argc, argv, &S[0], &periodic, &Niterations,
	       &Nsources, &Sources, &energy_per_source, &planes[0],
	       &output_energy_at_steps, &injection_frequency );
  
  // =================================================================
  // TIMING VARIABLES
  // =================================================================
  // Initialize timing variables for performance analysis
  double total_time = 0.0;
  double computation_time = 0.0;
  double start_time, end_time;
  
  int current = OLD;  // Start with OLD plane as current

  // =================================================================
  // INITIAL ENERGY INJECTION
  // =================================================================
  // If injection frequency is > 1, inject energy only at the beginning
  if ( injection_frequency > 1 )
    {
      // Inject energy at all source locations
      inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current] );
      injected_heat += Nsources * energy_per_source;
    }

  // =================================================================
  // MAIN SIMULATION LOOP
  // =================================================================
  // This is the core of the stencil computation
  // Each iteration updates the entire grid based on the stencil formula
  
  total_time = omp_get_wtime();  // Start total timing
  
  for (int iter = 0; iter < Niterations; ++iter)
    {
      // -------------------------------------------------------------
      // STEP 1: ENERGY INJECTION
      // -------------------------------------------------------------
      /* new energy from sources */
      // Inject energy at source locations based on injection frequency
      if ( iter % injection_frequency == 0 )
	{
	  inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current] );
	  injected_heat += Nsources*energy_per_source;
	}

      // -------------------------------------------------------------
      // STEP 2: STENCIL COMPUTATION (5-POINT STENCIL)
      // -------------------------------------------------------------
      /* update the plane */
      // This is the core stencil computation implementing Equation 3 from slides:
      // U_{m,l}^{n+1} = U_{m,l}^n + (αΔt/Δx²)(U_{m-1,l}^n + U_{m+1,l}^n - 2U_{m,l}^n) 
      //                  + (αΔt/Δy²)(U_{m,l-1}^n + U_{m,l+1}^n - 2U_{m,l}^n)
      // 
      // Each grid point is updated based on its 4 neighbors (North, South, East, West)
      
      start_time = omp_get_wtime();  // Start computation timing
      
      // =============================================================
      // OPENMP PARALLELIZATION STRATEGY:
      // =============================================================
      // #pragma omp parallel for schedule(static)
      //   - Creates a team of threads to parallelize the outer loop
      //   - Each thread processes a subset of rows (j indices)
      //   - The work is distributed among threads at compile time
      //
      // WHY schedule(static)?
      //   - Static scheduling divides iterations evenly among threads
      //   - Best for loops with uniform workload (each iteration similar)
      //   - Minimal runtime overhead (no dynamic load balancing)
      //   - Better cache locality (each thread works on contiguous data)
      //
      // PARALLELIZATION CORRECTNESS:
      //   - Each thread writes to DIFFERENT locations in planes[!current]
      //   - All threads read from planes[current] (shared, read-only)
      //   - NO DATA RACES: each (i,j) is written by exactly one thread
      //
      // PERFORMANCE CONSIDERATIONS:
      //   - Only outer loop is parallelized (j loop)
      //   - Inner loop (i) remains sequential for better cache utilization
      //   - This follows row-major memory layout in C
      // =============================================================
      
      #pragma omp parallel for schedule(static)
      for (int j = 1; j < S[_y_]-1; j++)  // Skip boundary points (j=0 and j=S[_y_]-1)
	{
	  for (int i = 1; i < S[_x_]-1; i++)  // Skip boundary points (i=0 and i=S[_x_]-1)
	    {
	      // -----------------------------------------------------
	      // CORRECTED STENCIL FORMULA (ENERGY CONSERVING)
	      // -----------------------------------------------------
	      // This formula ensures energy conservation:
	      // Total energy = α × center + (1-α)/4 × sum(neighbors)
	      // When α < 1, energy diffuses from hot to cold regions
	      
	      double alpha = 0.6;  // Diffusion coefficient (0 < α < 1)
	      double constant = (1-alpha) / 4.0;  // Pre-calculate constant for efficiency
	      
	      // 5-point stencil: center + 4 neighbors
	      // This implements the discretized Laplacian operator ∇²u
	      double result = planes[current][IDX(i,j)] * alpha +  // Center point contribution
		(planes[current][IDX(i-1, j)] +   // West neighbor
		 planes[current][IDX(i+1, j)] +   // East neighbor
		 planes[current][IDX(i, j-1)] +   // North neighbor
		 planes[current][IDX(i, j+1)]) * constant;  // South neighbor
	      
	      // Store result in NEW plane (avoiding read-after-write conflicts)
	      planes[!current][IDX(i,j)] = result;
	    }
	}
      
      end_time = omp_get_wtime();  // End computation timing
      computation_time += (end_time - start_time);

      // -------------------------------------------------------------
      // STEP 3: BOUNDARY CONDITIONS (SLIDE 10)
      // -------------------------------------------------------------
      /* boundary conditions */
      // Apply boundary conditions to the updated plane
      // This implements the boundary condition strategies from slide 10
      //
      // TWO TYPES OF BOUNDARY CONDITIONS:
      // 1. PERIODIC: Grid wraps around (infinite domain simulation)
      // 2. NON-PERIODIC: Fixed values at boundaries (finite domain)
      
      if ( periodic )
	{
	  // =======================================================
	  // PERIODIC BOUNDARY CONDITIONS (SLIDE 10)
	  // =======================================================
	  // "When the boundary conditions are periodic, your plate 
	  //  behaves like it is infinite"
	  //
	  // Implementation: copy opposite edges to create wrap-around effect
	  // - Left edge values → right boundary ghost cells
	  // - Right edge values → left boundary ghost cells
	  // - Top edge values → bottom boundary ghost cells
	  // - Bottom edge values → top boundary ghost cells
	  //
	  // NOTE: These loops could be parallelized with OpenMP,
	  // but the overhead is typically not worth it for small boundary sizes
	  
	  // Copy left edge to right boundary (wrap around horizontally)
	  for (int j = 0; j < S[_y_]; j++)
	    planes[!current][IDX(S[_x_]-1, j)] = planes[!current][IDX(1, j)];
	  
	  // Copy right edge to left boundary (wrap around horizontally)
	  for (int j = 0; j < S[_y_]; j++)
	    planes[!current][IDX(0, j)] = planes[!current][IDX(S[_x_]-2, j)];
	  
	  // Copy top edge to bottom boundary (wrap around vertically)
	  for (int i = 0; i < S[_x_]; i++)
	    planes[!current][IDX(i, S[_y_]-1)] = planes[!current][IDX(i, 1)];
	  
	  // Copy bottom edge to top boundary (wrap around vertically)
	  for (int i = 0; i < S[_x_]; i++)
	    planes[!current][IDX(i, 0)] = planes[!current][IDX(i, S[_y_]-2)];
	}
      else
	{
	  // =======================================================
	  // NON-PERIODIC BOUNDARY CONDITIONS (SLIDE 10)
	  // =======================================================
	  // Fixed boundaries: set all boundary points to zero
	  // This simulates a finite domain with heat dissipation at edges
	  //
	  // Physical interpretation: edges are kept at zero temperature
	  // (e.g., boundaries in contact with a cold reservoir)
	  
	  // Left and right boundaries (vertical edges)
	  for (int j = 0; j < S[_y_]; j++)
	    {
	      planes[!current][IDX(0, j)] = 0.0;              // Left edge
	      planes[!current][IDX(S[_x_]-1, j)] = 0.0;      // Right edge
	    }
	  
	  // Top and bottom boundaries (horizontal edges)
	  for (int i = 0; i < S[_x_]; i++)
	    {
	      planes[!current][IDX(i, 0)] = 0.0;              // Top edge
	      planes[!current][IDX(i, S[_y_]-1)] = 0.0;      // Bottom edge
	    }
	}

      // -------------------------------------------------------------
      // STEP 4: OUTPUT AND MONITORING
      // -------------------------------------------------------------
      /* output if needed */
      // Print energy statistics if requested
      if ( output_energy_at_steps )
	output_energy_stat ( iter, planes[!current], S, injected_heat );
	
      // Dump binary output for visualization
      char filename[100];
      sprintf( filename, "plane_%05d.bin", iter );
      dump( planes[!current], S, filename, NULL, NULL );

      // -------------------------------------------------------------
      // STEP 5: PREPARE FOR NEXT ITERATION
      // -------------------------------------------------------------
      /* swap plane indexes for the new iteration */
      // Swap OLD and NEW planes for the next iteration
      current = !current;
    }
  
  // =================================================================
  // FINAL STATISTICS AND CLEANUP
  // =================================================================
  total_time = omp_get_wtime() - total_time;  // Calculate total time

  // Calculate final system energy
  double system_heat = 0.0;
  for (int j = 0; j < S[_y_]; j++)
    for (int i = 0; i < S[_x_]; i++)
      system_heat += planes[!current][IDX(i,j)];

  // Print final results
  printf("Final results:\n");
  printf("Injected energy: %g\n", injected_heat);
  printf("System energy: %g\n", system_heat);
  printf("Total time: %.6f seconds\n", total_time);
  printf("Computation time: %.6f seconds\n", computation_time);
  printf("Computation efficiency: %.2f%%\n", (computation_time/total_time)*100.0);
  
  // CSV output system for scalability testing
  if (test) {
      // Get test parameters from environment variables
      const char* test_type = getenv("TEST_TYPE");
      const char* build_variant = getenv("BUILD_VARIANT"); // "ofast", "o1", "o0", "noarch"
      const char* threads_str = getenv("OMP_NUM_THREADS");
      
      if (test_type == NULL) {
          printf("Error: TEST_TYPE environment variable not set\n");
          return 1;
      }
      
      // Use default build variant if not specified
      if (build_variant == NULL) {
          build_variant = "ofast";  // default optimization level
      }
      
      // Convert environment variables to integers
      int threads = threads_str ? atoi(threads_str) : 1;

      // Create filename based on test_type
      char* filename = malloc(strlen(test_type) + 20);
      if (filename == NULL) {
          printf("Error: failed to allocate memory for filename\n");
          return 1;
      }
      
      sprintf(filename, "data/%s_serial_omp_results.csv", test_type);

      // Create data directory if it doesn't exist
      #ifdef _WIN32
      system("mkdir data 2>nul");
      #else
      system("mkdir -p data 2>/dev/null");
      #endif

      // Open the file in append mode
      FILE *results_file = fopen(filename, "a");
      if (results_file == NULL) {
          printf("Error opening results file: %s\n", strerror(errno));
          free(filename);
          return 1;
      }
  
      // Write header if file is empty
      fseek(results_file, 0, SEEK_END);
      long size = ftell(results_file);
      if (size == 0) {
          fprintf(results_file, "TestType,BuildVariant,Threads,EnergySources,XDim,YDim,Iterations,TotalTime,ComputationTime\n");
      }

      // Write data row
      fprintf(results_file, "%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f\n",
              test_type, build_variant, threads, Nsources, S[0], S[1], Niterations, total_time, computation_time);
  
      fclose(results_file);
      free(filename);
  }
  
  memory_release( planes[OLD], Sources );
  return 0;
}

/* ==========================================================================
   =                                                                        =
   =   routines called within the integration loop                          =
   ========================================================================== */

/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ========================================================================== */

int memory_allocate ( const int [2],
		      double ** );

int initialize_sources( int      [2],
			int       ,
			int     ** );

int initialize ( int      argc,                // the argc from command line
		 char   **argv,                // the argv from command line
		 int     *S,                   // two-uint array defining the x,y dimensions of the grid
		 int     *periodic,            // periodic-boundary tag
		 int     *Niterations,         // how many iterations
		 int     *Nsources,            // how many heat sources
		 int    **Sources,
		 double  *energy_per_source,   // how much heat per source
		 double **planes,
		 int     *output_energy_at_steps,
		 int     *injection_frequency
		 )
{
  int ret = 0;
  
  // ··································································
  // set default values

  S[_x_]            = 1000;
  S[_y_]            = 1000;
  *periodic         = 0;
  *Nsources         = 1;
  *Niterations      = 99;
  *output_energy_at_steps = 0;
  *energy_per_source = 1.0;
  *injection_frequency = *Niterations;

  // ··································································
  // parse the command line

  int opt;
  double freq = 0;
  while((opt = getopt(argc, argv, ":hx:y:e:E:f:n:o:p:v:")) != -1)
    {
      switch( opt )
	{
	case 'x': S[_x_] = atoi(optarg);
	  break;

	case 'y': S[_y_] = atoi(optarg);
	  break;

	case 'e': *Nsources = atoi(optarg);
	  break;

	case 'E': *energy_per_source = atof(optarg);
	  break;

	case 'f': freq = atof(optarg);
	  break;

	case 'n': *Niterations = atoi(optarg);
	  break;

	case 'o': *output_energy_at_steps = (atoi(optarg) > 0);
	  break;

	case 'p': *periodic = (atoi(optarg) > 0);
	  break;

	case 'v': printf("verbose output not implemented in this version\n");
	  break;

	case 'h': {
	  printf( "\nvalid options are ( values btw [] are the default values ):\n"
		  "-x    x size of the plate [1000]\n"
		  "-y    y size of the plate [1000]\n"
		  "-e    how many energy sources on the plate [1]\n"
		  "-E    how many energy sources on the plate [1.0]\n"
		  "-f    frequency of energy injection (0-1) [1.0]\n"
		  "-n    how many iterations [99]\n"
		  "-o    whether to output energy statistics at each step [0 = false]\n"
		  "-p    whether periodic boundaries applies  [0 = false]\n"
		  "-v    verbose output [0 = false]\n\n"
		  );
	  ret = 1; }
	  break;
	  
	case ':': printf( "option -%c requires an argument\n", optopt);
	  break;
	  
	case '?': printf(" -------- help unavailable ----------\n");
	  break;
	}
    }

  if ( ret )
    return 1;

  // Calculate injection frequency from freq parameter
  if ( freq == 0 )
    *injection_frequency = 1;
  else
    {
      freq = (freq > 1.0 ? 1.0 : freq );
      *injection_frequency = freq * *Niterations;
    }

  // ··································································
  // memory allocation

  ret = memory_allocate( S, planes );
  if ( ret )
    return 1;

  // ··································································
  // sources initialization

  ret = initialize_sources( S, *Nsources, Sources );
  if ( ret )
    return 1;

  return 0;
}

int memory_allocate ( const int S[2], double **planes )
{
  if( planes == NULL )
    return -1;

  // Allocate memory for both OLD and NEW planes
  // Each plane has S[0] * S[1] elements
  planes[OLD] = (double*)malloc( S[0] * S[1] * sizeof(double) );
  if ( planes[OLD] == NULL )
    return -1;

  planes[NEW] = (double*)malloc( S[0] * S[1] * sizeof(double) );
  if ( planes[NEW] == NULL )
    return -1;

  // Initialize all elements to zero
  // This is important for the stencil computation
  for (int j = 0; j < S[_y_]; j++)
    for (int i = 0; i < S[_x_]; i++)
      {
	planes[OLD][IDX(i,j)] = 0.0;
	planes[NEW][IDX(i,j)] = 0.0;
      }

  return 0;
}

int initialize_sources( int S[2], int Nsources, int **Sources )
{
  if( Sources == NULL )
    return -1;

  // Allocate memory for source coordinates
  // Each source has 2 coordinates (x, y)
  *Sources = (int*)malloc( 2 * Nsources * sizeof(int) );
  if ( *Sources == NULL )
    return -1;

  // Use fixed seed for reproducibility
  srand48(42);
  
  // Generate random source positions
  // Sources are placed randomly within the grid boundaries
  for (int s = 0; s < Nsources; s++)
    {
      (*Sources)[s*2] = 1+ lrand48() % S[_x_];
      (*Sources)[s*2+1] = 1+ lrand48() % S[_y_];
    }

  return 0;
}

int memory_release ( double *data, int *sources )
{
  if( data != NULL )
    free( data );

  if( sources != NULL )
    free( sources );

  return 0;
}

/*
 * =============================================================================
 * DUMP FUNCTION (WITH OPENMP OPTIMIZATION)
 * =============================================================================
 * 
 * This function writes the grid data to a binary file for visualization.
 * It also computes min/max values for normalization.
 * 
 * OPENMP OPTIMIZATION:
 * The min/max computation is parallelized using OpenMP reduction.
 */
int dump ( const double *data, const int size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
    {
      FILE *outfile = fopen( filename, "w" );
      if ( outfile == NULL )
	return 2;
      
      float *array = (float*)malloc( size[0] * sizeof(float) );
      
      double _min_ = DBL_MAX;
      double _max_ = 0;

      // =============================================================
      // FIRST PASS: PARALLEL MIN/MAX COMPUTATION
      // =============================================================
      // #pragma omp parallel for schedule(static) reduction(min:_min_) reduction(max:_max_)
      //   - Parallelizes the min/max computation across threads
      //   - reduction(min:_min_): each thread has private _min_, combined at end with min()
      //   - reduction(max:_max_): each thread has private _max_, combined at end with max()
      //   - This is a classic use case for OpenMP reductions
      //
      // WHY SEPARATE PASSES?
      //   - File writing (fwrite) must be sequential (cannot be parallelized)
      //   - Separating passes allows us to parallelize what we can
      // =============================================================
      
      #pragma omp parallel for schedule(static) reduction(min:_min_) reduction(max:_max_)
      for ( int j = 0; j < (int)size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < (int)size[0]; i++ ) {
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	}
      
      // =============================================================
      // SECOND PASS: SEQUENTIAL FILE WRITING
      // =============================================================
      // File I/O must be sequential - cannot be parallelized
      // Each row is converted to float and written to file
      
      for ( int j = 0; j < (int)size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < (int)size[0]; i++ ) {
	    array[i] = (float)line[i]; }
	  
	  fwrite( array, sizeof(float), size[0], outfile );
	}
      
      free( array );
      
      fclose( outfile );

      if ( min != NULL )
	*min = _min_;
      if ( max != NULL )
	*max = _max_;
    }

  else return 1;
  
  return 0;
}
