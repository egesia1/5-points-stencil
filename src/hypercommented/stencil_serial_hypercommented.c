/*
 * =============================================================================
 * HYPER-COMMENTED SERIAL STENCIL COMPUTATION
 * =============================================================================
 * 
 * This file implements a serial version of the 5-points stencil computation
 * for solving the 2D heat equation. This serves as the foundation for the
 * parallel implementation.
 * 
 * MATHEMATICAL BACKGROUND (from slides):
 * The heat equation in 2D is:
 * ∂u(t,x,y)/∂t = α(∂²u/∂x² + ∂²u/∂y²)
 * 
 * Discretized with finite differences on an m×l grid:
 * U_{m,l}^{n+1} = U_{m,l}^n + (αΔt/Δx²)(U_{m-1,l}^n + U_{m+1,l}^n - 2U_{m,l}^n) 
 *                           + (αΔt/Δy²)(U_{m,l-1}^n + U_{m,l+1}^n - 2U_{m,l}^n)
 * 
 * This gives us the 5-points stencil:
 *     ●
 *   ● ○ ●  where ○ depends on its 4 neighbors ●
 *     ●
 * 
 * The stencil formula used here:
 * new[i,j] = α * old[i,j] + (1-α)/4 * (old[i-1,j] + old[i+1,j] + old[i,j-1] + old[i,j+1])
 * where α = 0.6 (heat retention factor)
 * 
 * ENERGY CONSERVATION:
 * This implementation conserves energy perfectly when using the correct stencil formula.
 * The total energy in the system should equal the sum of all injected energy.
 */

#define _XOPEN_SOURCE
#include "stencil.h"
#include <stdlib.h>
#include <float.h>

// Function declarations
int dump ( const double *, const uint [2], const char *, double *, double * );

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
  // =================================================================
  // VARIABLE DECLARATIONS
  // =================================================================
  
  int  Niterations;        // Number of time steps to simulate
  int  periodic;           // Flag for periodic boundary conditions (0=no, 1=yes)
  vec2_t S, N;            // S: global grid size [x,y], N: MPI grid size [x,y] (unused in serial)
  
  int      Nsources;       // Total number of heat sources in the system
  int      Nsources_local; // Number of sources in this process (same as Nsources in serial)
  vec2_t  *Sources_local;  // Array of source coordinates [x,y] pairs
  double   energy_per_source; // Energy injected per source per iteration

  plane_t   planes[2];     // Two planes: OLD (current state) and NEW (updated state)
  buffers_t buffers[2];    // Communication buffers (unused in serial)
  
  int output_energy_stat_perstep; // Flag to print energy statistics at each step
  int injection_frequency;        // Frequency of energy injection
  double injected_heat = 0;       // Total heat injected so far
  int test = 0;  // whether we are running a scalability test
  test = getenv("TEST_TYPE") != NULL;
  
  // =================================================================
  // INITIALIZATION PHASE
  // =================================================================
  
  /* argument checking and setting */
  // Parse command line arguments and set up the simulation parameters
  // This includes: grid size, number of iterations, heat sources, etc.
  int ret = initialize ( NULL, 0, 1, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
			 NULL, &Niterations,
			 &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
			 &planes[0], &buffers[0], &injection_frequency );

  if ( ret )
    {
      printf("Initialization failed with code %d\n", ret);
      return 1;
    }
  
  // =================================================================
  // MAIN SIMULATION LOOP
  // =================================================================
  
  int current = OLD;       // Start with OLD plane as current
  double t1 = MPI_Wtime(); // Start timing (MPI_Wtime works even in serial)
  
  // Initial energy injection if frequency > 1
  if ( injection_frequency > 1 )
    inject_energy( periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N );
  
  for (int iter = 0; iter < Niterations; ++iter)
    {
      // -------------------------------------------------------------
      // STEP 1: ENERGY INJECTION
      // -------------------------------------------------------------
      /* new energy from sources */
      // Inject heat energy at the source locations based on frequency
      // This simulates external heat sources in the system
      if ( iter % injection_frequency == 0 )
	{
	  inject_energy( periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N );
	  injected_heat += Nsources*energy_per_source;
	}

      // -------------------------------------------------------------
      // STEP 2: STENCIL COMPUTATION (THE CORE ALGORITHM)
      // -------------------------------------------------------------
      /* update grid points */
      // Apply the 5-points stencil to compute new values
      // This is where the heat diffusion happens
      update_plane( periodic, N, &planes[current], &planes[!current] );

      // -------------------------------------------------------------
      // STEP 3: OUTPUT AND MONITORING
      // -------------------------------------------------------------
      /* output if needed */
      // Print energy statistics if requested
      // This helps verify energy conservation
      if ( output_energy_stat_perstep )
	{
	  output_energy_stat ( iter, &planes[!current], injected_heat, 0, NULL );
	  
	  // Dump binary output for visualization
	  char filename[100];
	  sprintf( filename, "plane_%05d.bin", iter );
	  dump( planes[!current].data, planes[!current].size, filename, NULL, NULL );
	}
	
      // -------------------------------------------------------------
      // STEP 4: PREPARE FOR NEXT ITERATION
      // -------------------------------------------------------------
      /* swap plane indexes for the new iteration */
      // The NEW plane becomes OLD for the next iteration
      // This implements the time-stepping scheme
      current = !current;
    }
  
  // =================================================================
  // FINAL STATISTICS AND CLEANUP
  // =================================================================
  
  t1 = MPI_Wtime() - t1;   // Calculate total execution time

  // Print final energy statistics
  // This is crucial for verifying energy conservation
  output_energy_stat ( -1, &planes[!current], injected_heat, 0, NULL );
  
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
      
      sprintf(filename, "data/%s_serial_basic_results.csv", test_type);

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
  
  // Clean up allocated memory
  memory_release( &planes[0], &buffers[0] );
  
  return 0;
}

int dump ( const double *data, const uint size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
    {
      FILE *outfile = fopen( filename, "w" );
      if ( outfile == NULL )
	return 2;
      
      float *array = (float*)malloc( size[0] * sizeof(float) );
      
      double _min_ = DBL_MAX;
      double _max_ = 0;

      for ( int j = 0; j < size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < size[0]; i++ ) {
	    array[i] = (float)line[i];
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	  
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
  
}

/* ==========================================================================
   =                                                                        =
   =   INITIALIZATION ROUTINES                                             =
   ========================================================================== */

uint simple_factorization( uint, int *, uint ** );

int initialize_sources( int       ,
			int       ,
			MPI_Comm  *,
			vec2_t    ,
			int       ,
			int      *,
			vec2_t  ** );

int memory_allocate ( const int       *,
		      const vec2_t     ,
		            buffers_t *,
		            plane_t   * );

int initialize ( MPI_Comm *Comm,           // MPI communicator (NULL in serial)
		 int      Me,                   // Process rank (0 in serial)
		 int      Ntasks,               // Total processes (1 in serial)
		 int      argc,                 // Command line argument count
		 char   **argv,                 // Command line arguments
		 vec2_t  *S,                    // Global grid dimensions [x,y]
		 vec2_t  *N,                    // MPI process grid [x,y] (unused in serial)
		 int     *periodic,             // Periodic boundary flag
		 int     *output_energy_stat,   // Output energy statistics flag
		 int     *neighbours,           // Neighbor process ranks (unused in serial)
		 int     *Niterations,          // Number of time iterations
		 int     *Nsources,             // Total number of heat sources
		 int     *Nsources_local,       // Local sources (same as total in serial)
		 vec2_t **Sources_local,        // Array of source coordinates
		 double  *energy_per_source,    // Energy per source per iteration
		 plane_t *planes,               // Data planes for OLD/NEW states
		 buffers_t *buffers,            // Communication buffers (unused in serial)
		 int     *injection_frequency   // Frequency of energy injection
		 )
{
  int halt = 0;
  int ret;
  int verbose = 0;
  double freq = 0;
  
  // =================================================================
  // SET DEFAULT PARAMETERS
  // =================================================================
  // These are the default values if no command line arguments are provided
  
  (*S)[_x_]         = 10000;        // Default grid width
  (*S)[_y_]         = 10000;        // Default grid height
  *periodic         = 0;            // Default: non-periodic boundaries
  *Nsources         = 4;            // Default: 4 heat sources
  *Nsources_local   = 0;            // Will be set to Nsources in serial
  *Sources_local    = NULL;         // Will be allocated later
  *Niterations      = 1000;         // Default: 1000 time steps
  *energy_per_source = 1.0;         // Default: 1.0 energy unit per source
  *injection_frequency = *Niterations; // Default: inject every iteration

  // Initialize plane sizes to 0 (will be set later)
  planes[OLD].size[0] = planes[OLD].size[1] = 0;
  planes[NEW].size[0] = planes[NEW].size[1] = 0;
  
  // Initialize neighbor array (unused in serial)
  for ( int i = 0; i < 4; i++ )
    neighbours[i] = MPI_PROC_NULL;

  // Initialize buffer pointers (unused in serial)
  for ( int b = 0; b < 2; b++ )
    for ( int d = 0; d < 4; d++ )
      buffers[b][d] = NULL;
  
  // =================================================================
  // COMMAND LINE ARGUMENT PARSING
  // =================================================================
  // Parse command line arguments using getopt
  // This allows users to customize the simulation parameters
  
  while ( 1 )
  {
    int opt;
    while((opt = getopt(argc, argv, ":hx:y:e:E:f:n:o:p:v:")) != -1)
      {
	switch( opt )
	  {
	  case 'x': (*S)[_x_] = (uint)atoi(optarg);  // Set grid width
	    break;

	  case 'y': (*S)[_y_] = (uint)atoi(optarg);  // Set grid height
	    break;

	  case 'e': *Nsources = atoi(optarg);        // Set number of heat sources
	    break;

	  case 'E': *energy_per_source = atof(optarg); // Set energy per source
	    break;

	  case 'f': freq = atof(optarg);             // Set injection frequency
	    break;

	  case 'n': *Niterations = atoi(optarg);     // Set number of iterations
	    break;

	  case 'o': *output_energy_stat = (atoi(optarg) > 0); // Enable energy output
	    break;

	  case 'p': *periodic = (atoi(optarg) > 0);  // Enable periodic boundaries
	    break;

	  case 'v': verbose = atoi(optarg);          // Set verbosity level
	    break;

	  case 'h': {  // Print help message
	    if ( Me == 0 )
	      printf( "\nvalid options are ( values btw [] are the default values ):\n"
		      "-x    x size of the plate [10000]\n"
		      "-y    y size of the plate [10000]\n"
		      "-e    how many energy sources on the plate [4]\n"
		      "-E    how many energy sources on the plate [1.0]\n"
		      "-f    the frequency of energy injection [1.0]\n"
		      "-n    how many iterations [1000]\n"
		      "-p    whether periodic boundaries applies  [0 = false]\n"
		      "-v    verbose output [0 = false]\n\n"
		      );
	    halt = 1; }
	    break;
	    
	  case ':': printf( "option -%c requires an argument\n", optopt);
	    break;
	    
	  case '?': printf(" -------- help unavailable ----------\n");
	    break;
	  }
      }

    if ( opt == -1 )
      break;
  }

  if ( halt )
    return 1;
  
  // Calculate injection frequency from freq parameter
  if ( freq == 0 )
    *injection_frequency = 1;
  else
    {
      freq = (freq > 1.0 ? 1.0 : freq );
      *injection_frequency = freq * *Niterations;
    }
  
  // =================================================================
  // PARAMETER VALIDATION
  // =================================================================
  // Here we should check that all parameters are meaningful
  // For example: positive grid sizes, reasonable number of iterations, etc.
  
  // In serial, all sources are local to the single process
  *Nsources_local = *Nsources;
  
  // =================================================================
  // MEMORY ALLOCATION
  // =================================================================
  // Allocate memory for the data planes and communication buffers
  
  ret = memory_allocate( neighbours, *N, buffers, planes );
  if ( ret != 0 ) {
    printf("Memory allocation failed\n");
    return ret;
  }

  // =================================================================
  // HEAT SOURCE INITIALIZATION
  // =================================================================
  // Set up the heat sources at random locations in the grid
  
  ret = initialize_sources( Me, Ntasks, Comm, *S, *Nsources, Nsources_local, Sources_local );
  if ( ret != 0 ) {
    printf("Source initialization failed\n");
    return ret;
  }
  
  return 0;  
}

uint simple_factorization( uint A, int *Nfactors, uint **factors )
/*
 * SIMPLE FACTORIZATION ALGORITHM
 * 
 * This function finds all prime factors of a number A.
 * It's used in the parallel version to determine the optimal
 * MPI process grid decomposition.
 * 
 * For example: A=12 → factors=[2,2,3], Nfactors=3
 * 
 * This is a simple trial division algorithm, suitable for small numbers
 * (which is fine since A represents the number of MPI processes).
 */
{
  int N = 0;
  int f = 2;
  uint _A_ = A;

  // First pass: count the number of factors
  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	N++;
	_A_ /= f; }

      f++;
    }

  *Nfactors = N;
  uint *_factors_ = (uint*)malloc( N * sizeof(uint) );

  // Second pass: collect the actual factors
  N   = 0;
  f   = 2;
  _A_ = A;

  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	_factors_[N++] = f;
	_A_ /= f; }
      f++;
    }

  *factors = _factors_;
  return 0;
}

int dump ( const double *data, const uint size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
    {
      FILE *outfile = fopen( filename, "w" );
      if ( outfile == NULL )
	return 2;
      
      float *array = (float*)malloc( size[0] * sizeof(float) );
      
      double _min_ = DBL_MAX;
      double _max_ = 0;

      for ( int j = 0; j < size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < size[0]; i++ ) {
	    array[i] = (float)line[i];
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	  
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
  
}

int initialize_sources( int       Me,           // Process rank (0 in serial)
			int       Ntasks,        // Total processes (1 in serial)
			MPI_Comm *Comm,          // MPI communicator (NULL in serial)
			vec2_t    mysize,        // Local grid size (same as global in serial)
			int       Nsources,      // Total number of sources
			int      *Nsources_local, // Local sources (same as total in serial)
			vec2_t  **Sources        // Array of source coordinates
			)
/*
 * HEAT SOURCE INITIALIZATION
 * 
 * This function randomly places heat sources in the computational domain.
 * In the parallel version, sources are distributed among MPI processes,
 * but in serial, all sources are local to the single process.
 * 
 * The sources are placed at random coordinates within the grid boundaries.
 * We use a fixed seed (42) for reproducibility across runs.
 */
{
  // Use fixed seed for reproducibility
  srand48(42);
  
  // In serial, all sources are local
  *Nsources_local = Nsources;
  
  if ( Nsources > 0 )
    {
      // Allocate memory for source coordinates
      // Each source has 2 coordinates: [x, y]
      vec2_t * restrict helper = (vec2_t*)malloc( Nsources * sizeof(vec2_t) );
      if ( helper == NULL ) {
	printf("Failed to allocate memory for sources\n");
	return -1;
      }
      
      // Place sources at random locations within the grid
      for ( int s = 0; s < Nsources; s++ )
	{
	  // Random x coordinate: 1 to mysize[_x_] (avoiding boundaries)
	  helper[s][_x_] = 1 + lrand48() % mysize[_x_];
	  // Random y coordinate: 1 to mysize[_y_] (avoiding boundaries)
	  helper[s][_y_] = 1 + lrand48() % mysize[_y_];
	}

      *Sources = helper;
    }
  
  return 0;
}

int dump ( const double *data, const uint size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
    {
      FILE *outfile = fopen( filename, "w" );
      if ( outfile == NULL )
	return 2;
      
      float *array = (float*)malloc( size[0] * sizeof(float) );
      
      double _min_ = DBL_MAX;
      double _max_ = 0;

      for ( int j = 0; j < size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < size[0]; i++ ) {
	    array[i] = (float)line[i];
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	  
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
  
}

int memory_allocate ( const int       *neighbours,  // Neighbor process ranks (unused in serial)
		      const vec2_t     N,           // MPI process grid (unused in serial)
		            buffers_t *buffers_ptr,  // Communication buffers (unused in serial)
		            plane_t   *planes_ptr    // Data planes for OLD/NEW states
		      )
/*
 * MEMORY ALLOCATION FOR DATA PLANES
 * 
 * This function allocates memory for the computational grid.
 * In serial, we only need the data planes, not communication buffers.
 * 
 * MEMORY LAYOUT:
 * - We allocate space for the actual grid plus a 1-point halo around it
 * - The halo is used for boundary conditions and will be used for
 *   communication in the parallel version
 * - Grid size: (xsize+2) × (ysize+2) to include the halo
 * 
 * The planes array contains:
 * - planes[OLD]: Current state of the grid
 * - planes[NEW]: Updated state of the grid
 * 
 * At each time step, we compute NEW from OLD, then swap them.
 */
{
  if (planes_ptr == NULL )
    return -1;

  if (buffers_ptr == NULL )
    return -1;
    
  // =================================================================
  // CALCULATE MEMORY REQUIREMENTS
  // =================================================================
  // We need space for the grid plus a 1-point halo around it
  // This halo will be used for boundary conditions and communication
  
  unsigned int frame_size = (planes_ptr[OLD].size[_x_]+2) * (planes_ptr[OLD].size[_y_]+2);

  // =================================================================
  // ALLOCATE OLD PLANE
  // =================================================================
  // Allocate memory for the current state of the grid
  planes_ptr[OLD].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[OLD].data == NULL )
    return -1;
  
  // Initialize to zero using OpenMP for better memory bandwidth
  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < frame_size; i++) {
    planes_ptr[OLD].data[i] = 0.0;
  }

  // =================================================================
  // ALLOCATE NEW PLANE
  // =================================================================
  // Allocate memory for the updated state of the grid
  planes_ptr[NEW].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[NEW].data == NULL )
    return -1;
  
  // Initialize to zero using OpenMP for better memory bandwidth
  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < frame_size; i++) {
    planes_ptr[NEW].data[i] = 0.0;
  }

  // =================================================================
  // COMMUNICATION BUFFERS (UNUSED IN SERIAL)
  // =================================================================
  // In the parallel version, we would allocate buffers for communication
  // with neighboring processes. In serial, we just initialize them to NULL.
  
  for ( int b = 0; b < 2; b++ )
    for ( int d = 0; d < 4; d++ )
      (*buffers_ptr)[b][d] = NULL;
  
  return 0;
}

int dump ( const double *data, const uint size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
    {
      FILE *outfile = fopen( filename, "w" );
      if ( outfile == NULL )
	return 2;
      
      float *array = (float*)malloc( size[0] * sizeof(float) );
      
      double _min_ = DBL_MAX;
      double _max_ = 0;

      for ( int j = 0; j < size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < size[0]; i++ ) {
	    array[i] = (float)line[i];
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	  
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
  
}

int memory_release ( plane_t   *planes,    // Data planes to free
		     buffers_t *buffers   // Communication buffers to free
		     )
/*
 * MEMORY CLEANUP
 * 
 * This function frees all allocated memory to prevent memory leaks.
 * It's called at the end of the program to clean up resources.
 */
{
  // Free data planes
  if ( planes != NULL )
    {
      if ( planes[OLD].data != NULL )
	free (planes[OLD].data);
      
      if ( planes[NEW].data != NULL )
	free (planes[NEW].data);
    }

  // Free communication buffers (unused in serial, but kept for compatibility)
  if ( buffers != NULL )
    {
      for ( int b = 0; b < 2; b++ )
	for ( int d = 0; d < 4; d++ )
	  if ( (*buffers)[b][d] != NULL )
	    free ((*buffers)[b][d]);
    }
      
  return 0;
}

int dump ( const double *data, const uint size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
    {
      FILE *outfile = fopen( filename, "w" );
      if ( outfile == NULL )
	return 2;
      
      float *array = (float*)malloc( size[0] * sizeof(float) );
      
      double _min_ = DBL_MAX;
      double _max_ = 0;

      for ( int j = 0; j < size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < size[0]; i++ ) {
	    array[i] = (float)line[i];
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	  
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
  
}

int output_energy_stat ( int step,        // Current time step (-1 for final)
                        plane_t *plane,   // Grid to analyze
                        double budget,    // Total energy injected so far
                        int Me,           // Process rank (0 in serial)
                        MPI_Comm *Comm    // MPI communicator (NULL in serial)
                        )
/*
 * ENERGY STATISTICS OUTPUT
 * 
 * This function calculates and prints energy statistics for the system.
 * It's crucial for verifying energy conservation in the simulation.
 * 
 * ENERGY CONSERVATION CHECK:
 * - Injected energy: sum of all energy added by heat sources
 * - System energy: sum of all energy currently in the grid
 * - These should be equal (within numerical precision) for energy conservation
 * 
 * In the parallel version, this function uses MPI_Reduce to sum up
 * energy from all processes. In serial, it just calculates the local energy.
 */
{
  double system_energy = 0;
  double tot_system_energy = 0;
  
  // Calculate total energy in the current grid
  get_total_energy ( plane, &system_energy );
  
  // In serial, local energy equals total energy
  tot_system_energy = system_energy;
  
  // Print energy statistics
  if ( Me == 0 )
    {
      if ( step >= 0 )
	printf(" [ step %4d ] ", step ); fflush(stdout);

      printf( "total injected energy is %g, "
	      "system energy is %g "
	      "( in avg %g per grid point)\n",
	      budget,
	      tot_system_energy,
	      tot_system_energy / (plane->size[_x_]*plane->size[_y_]) );
    }
  
  return 0;
}

int dump ( const double *data, const uint size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
    {
      FILE *outfile = fopen( filename, "w" );
      if ( outfile == NULL )
	return 2;
      
      float *array = (float*)malloc( size[0] * sizeof(float) );
      
      double _min_ = DBL_MAX;
      double _max_ = 0;

      for ( int j = 0; j < size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < size[0]; i++ ) {
	    array[i] = (float)line[i];
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	  
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
  
}
