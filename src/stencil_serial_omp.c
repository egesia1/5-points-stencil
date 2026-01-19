/*
 * Serial stencil computation with OpenMP optimization
 * Based on the template but with OpenMP parallelization
 */

#define _XOPEN_SOURCE
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include "stencil_template_serial.h"

typedef unsigned int uint;

int dump ( const double *, const int [2], const char *, double *, double * );

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
  int  Niterations;
  int  periodic;
  int  S[2];
  
  int     Nsources;
  int    *Sources;
  double  energy_per_source;

  double *planes[2];
  
  double injected_heat = 0;

  int injection_frequency;
  int output_energy_at_steps = 0;
  int test = 0;  // whether we are running a scalability test
  test = getenv("TEST_TYPE") != NULL;
   
  /* argument checking and setting */
  initialize ( argc, argv, &S[0], &periodic, &Niterations,
	       &Nsources, &Sources, &energy_per_source, &planes[0],
	       &output_energy_at_steps, &injection_frequency );
  
  // Initialize timing variables
  double total_time = 0.0;
  double computation_time = 0.0;
  double start_time, end_time;
  
  int current = OLD;

  if ( injection_frequency > 1 )
    inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current] );
  
  // Main computation loop with timing
  start_time = omp_get_wtime();
  
  for (int iter = 0; iter < Niterations; iter++)
    {      
      /* new energy from sources */
      if ( iter % injection_frequency == 0 )
	{
	  inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current] );
	  injected_heat += Nsources*energy_per_source;
	}
                  
      /* update grid points with timing */
      double comp_start = omp_get_wtime();
      update_plane(periodic, S, planes[current], planes[!current] );
      double comp_end = omp_get_wtime();
      computation_time += (comp_end - comp_start);

      if ( output_energy_at_steps )
	{
	  double system_heat;
	  get_total_energy( S, planes[!current], &system_heat);
            
	  printf("step %d :: injected energy is %g, updated system energy is %g\n", iter, 
		 injected_heat, system_heat );

	  char filename[100];
	  sprintf( filename, "plane_%05d.bin", iter );
	  dump( planes[!current], S, filename, NULL, NULL );
	}

      /* swap planes for the new iteration */
      current = !current;
    }
  
  end_time = omp_get_wtime();
  total_time = end_time - start_time;
  
  /* get final heat in the system */
  double system_heat;
  get_total_energy( S, planes[current], &system_heat);

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
      const char* build_variant = getenv("BUILD_VARIANT"); // es. "ofast", "o1", "o0", "noarch"
      const char* threads_str = getenv("OMP_NUM_THREADS");
      
      if (test_type == NULL) {
          printf("Error: TEST_TYPE environment variable not set\n");
          return 1;
      }
      
      // Use default build variant if not specified
      if (build_variant == NULL) {
          build_variant = "ofast_omp_improved";  // default optimization level
      }
      
      // Convert environment variables to integers
      int threads = threads_str ? atoi(threads_str) : 1;

      // Create filename based on test_type (use fixed-size buffer)
      char filename[256];
      snprintf(filename, sizeof(filename), "data/SM3800083_%s_serial_results.csv", test_type);

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

  double freq = 0;
  
  // ··································································
  // process the command line
  // 
  while ( 1 )
  {
    int opt;
    while((opt = getopt(argc, argv, ":x:y:e:E:f:n:p:o:")) != -1)
      {
	switch( opt )
	  {
	  case 'x': S[_x_] = (uint)atoi(optarg);
	    break;

	  case 'y': S[_y_] = (uint)atoi(optarg);
	    break;

	  case 'e': *Nsources = atoi(optarg);
	    break;

	  case 'E': *energy_per_source = atof(optarg);
	    break;

	  case 'n': *Niterations = atoi(optarg);
	    break;

	  case 'p': *periodic = (atoi(optarg) > 0);
	    break;

	  case 'o': *output_energy_at_steps = (atoi(optarg) > 0);
	    break;

	  case 'f': freq = atof(optarg);
	    break;
	    
	  case 'h': printf( "valid options are ( values btw [] are the default values ):\n"
			    "-x    x size of the plate [1000]\n"
			    "-y    y size of the plate [1000]\n"
			    "-e    how many energy sources on the plate [1]\n"
			    "-E    how many energy sources on the plate [1.0]\n"
			    "-f    the frequency of energy injection [0.0]\n"
			    "-n    how many iterations [100]\n"
			    "-p    whether periodic boundaries applies  [0 = false]\n"
			    "-o    whether to print the energy budget at every step [0 = false]\n"
			    );
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

  if ( freq == 0 )
    *injection_frequency = 1;
  else
    {
      freq = (freq > 1.0 ? 1.0 : freq );
      *injection_frequency = freq * *Niterations;
    }

  // ··································································
  /*
   * here we should check for all the parms being meaningful
   *
   */

  // ...
  
  // ··································································
  // allocate the needed memory
  //
  ret = memory_allocate( S, planes );
  if (ret != 0) return ret;
  
  // ··································································
  // allocate the heat sources
  //
  ret = initialize_sources( S, *Nsources, Sources );
  if (ret != 0) return ret;
  
  return 0;  
}

int memory_allocate ( const int      size[2],
		            double **planes_ptr )
{
  if (planes_ptr == NULL )
    {
      // an invalid pointer has been passed
      // manage the situation
      return -1;
    }

  unsigned int bytes = (size[_x_]+2)*(size[_y_]+2);

  planes_ptr[OLD] = (double*)malloc( 2*bytes*sizeof(double) );
  memset ( planes_ptr[OLD], 0, 2*bytes*sizeof(double) );
  planes_ptr[NEW] = planes_ptr[OLD] + bytes;
      
  return 0;
}

int initialize_sources( int      size[2],
			int       Nsources,
			int     **Sources )
{
  *Sources = (int*)malloc( Nsources * 2 *sizeof(uint) );
  srand48(42);  // Fixed seed for reproducibility
  for ( int s = 0; s < Nsources; s++ )
    {
      (*Sources)[s*2] = 1+ lrand48() % size[_x_];
      (*Sources)[s*2+1] = 1+ lrand48() % size[_y_];
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

      // First pass: parallel computation of min/max and conversion
      #pragma omp parallel for schedule(static) reduction(min:_min_) reduction(max:_max_)
      for ( int j = 0; j < (int)size[1]; j++ )
	{
	  const double * restrict line = data + j*size[0];
	  for ( int i = 0; i < (int)size[0]; i++ ) {
	    _min_ = ( line[i] < _min_? line[i] : _min_ );
	    _max_ = ( line[i] > _max_? line[i] : _max_ ); }
	}
      
      // Second pass: sequential file writing (cannot be parallelized)
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
