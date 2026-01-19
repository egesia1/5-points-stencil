/*
 * Parallel stencil computation with MPI and OpenMP
 * Complete implementation with halo communication
 */

#define _XOPEN_SOURCE
#include "stencil.h"
#include <stdlib.h>
#include <errno.h>
#include <float.h>

// Function declarations
int dump ( const double *, const uint [2], const char *, double *, double * );

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
  // Timing variables
  double comm_time = 0.0, comp_time = 0.0, total_time, init_time;
  double internal_comp_time = 0.0, border_comp_time = 0.0;
  double start_time_comm, start_time_comp;
  double pack_time = 0.0, unpack_time = 0.0, halo_copy_time = 0.0;
  double start_time_pack, start_time_unpack, start_time_halo;
  
  MPI_Comm myCOMM_WORLD;
  int  Rank, Ntasks;
  uint neighbours[4];

  int  Niterations;
  int  periodic;
  vec2_t S, N;
  
  int      Nsources;
  int      Nsources_local;
  vec2_t  *Sources_local;
  double   energy_per_source;

  plane_t   planes[2];  
  buffers_t buffers;
  
  int output_energy_stat_perstep = 0;  // Initialize to 0
  int test = 0;  // whether we are running a scalability test
  test = getenv("TEST_TYPE") != NULL;
  
  // Energy injection frequency (from serial)
  int injection_frequency = 1;
  double injected_heat = 0;
  
  /* initialize MPI environment */
  {
    int level_obtained;
    
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &level_obtained );
    if ( level_obtained < MPI_THREAD_FUNNELED ) {
      printf("MPI_thread level obtained is %d instead of %d\n",
	     level_obtained, MPI_THREAD_FUNNELED );
      MPI_Finalize();
      exit(1); }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
    MPI_Comm_dup (MPI_COMM_WORLD, &myCOMM_WORLD);
  }
  
  /* argument checking and setting */
  init_time = MPI_Wtime();
  int ret = initialize ( &myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
			 neighbours, &Niterations,
			 &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
			 &planes[0], &buffers, &injection_frequency );

  if ( ret )
    {
      printf("task %d is opting out with termination code %d\n",
	     Rank, ret );
      
      MPI_Finalize();
      return 0;
    }
  
  init_time = MPI_Wtime() - init_time;
  total_time = MPI_Wtime();
  
  int current = OLD;
  
  // Initial energy injection if frequency > 1
  if ( injection_frequency > 1 )
    inject_energy( periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N );
  
  for (int iter = 0; iter < Niterations; ++iter)
    {
      MPI_Request reqs[8];
      int nreq = 0;
      
      /* new energy from sources */
      if ( iter % injection_frequency == 0 )
	{
	  inject_energy( periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N );
	  injected_heat += Nsources*energy_per_source;
	}

      /* -------------------------------------- */
      /* HALO COMMUNICATION SECTION  */

      // Variables for communication strategy
      uint xsize = planes[current].size[_x_];
      uint ysize = planes[current].size[_y_];
      uint xframe = xsize + 2;
      uint i;

      // Prepare East/West buffers
      if (neighbours[WEST] != MPI_PROC_NULL && buffers[SEND][WEST] != NULL) {
          #pragma omp parallel for schedule(static)
          for (i = 0; i < ysize; i++) {
              // WEST: first effective column (excluding frame)
              buffers[SEND][WEST][i] = planes[current].data[(i + 1) * xframe + 1];
          }
      }
      if (neighbours[EAST] != MPI_PROC_NULL && buffers[SEND][EAST] != NULL) {
          #pragma omp parallel for schedule(static)
          for (i = 0; i < ysize; i++) {
              // EAST: last effective column (excluding frame)
              buffers[SEND][EAST][i] = planes[current].data[(i + 1) * xframe + xsize];
          }
      }
      
      // Set up North/South buffer pointers
      if (neighbours[NORTH] != MPI_PROC_NULL) {
          buffers[SEND][NORTH] = &(planes[current].data[xframe + 1]);     // the first effective row
          buffers[RECV][NORTH] = &(planes[current].data[1]);
      }
      if (neighbours[SOUTH] != MPI_PROC_NULL) {
          buffers[SEND][SOUTH] = &(planes[current].data[ysize * xframe + 1]); // the last effective row
          buffers[RECV][SOUTH] = &(planes[current].data[(ysize + 1) * xframe + 1]);
      }

      // Start communication timing (after buffer setup)
      double comm_start = MPI_Wtime();
      
      // Perform communications with self-communication optimization
      if (neighbours[EAST] != MPI_PROC_NULL) {
          // optimization: if the neighbor is the same rank, we can just copy the data
          if (neighbours[EAST] == Rank) {
              #pragma omp parallel for schedule(static)
              for (i = 0; i < ysize; i++) {
                  buffers[RECV][EAST][i] = buffers[SEND][EAST][i];
              }
          } else {
              MPI_Isend(buffers[SEND][EAST], (int)ysize, MPI_DOUBLE, neighbours[EAST], TAG_E, myCOMM_WORLD, &reqs[nreq++]);
              MPI_Irecv(buffers[RECV][EAST], (int)ysize, MPI_DOUBLE, neighbours[EAST], TAG_W, myCOMM_WORLD, &reqs[nreq++]);
          }
      }
      if (neighbours[WEST] != MPI_PROC_NULL) {
          if (neighbours[WEST] == Rank) {
              #pragma omp parallel for schedule(static)
              for (i = 0; i < ysize; i++) {
                  buffers[RECV][WEST][i] = buffers[SEND][WEST][i];
              }
          } else {
              MPI_Isend(buffers[SEND][WEST], (int)ysize, MPI_DOUBLE, neighbours[WEST], TAG_W, myCOMM_WORLD, &reqs[nreq++]);
              MPI_Irecv(buffers[RECV][WEST], (int)ysize, MPI_DOUBLE, neighbours[WEST], TAG_E, myCOMM_WORLD, &reqs[nreq++]);
          }
      }
      if (neighbours[NORTH] != MPI_PROC_NULL) {
          if (neighbours[NORTH] == Rank) {
              #pragma omp parallel for schedule(static)
              for (i = 0; i < xsize; i++) {
                  buffers[RECV][NORTH][i] = buffers[SEND][NORTH][i];
              }
          } else {
              MPI_Isend(buffers[SEND][NORTH], (int)xsize, MPI_DOUBLE, neighbours[NORTH], TAG_N, myCOMM_WORLD, &reqs[nreq++]);
              MPI_Irecv(buffers[RECV][NORTH], (int)xsize, MPI_DOUBLE, neighbours[NORTH], TAG_S, myCOMM_WORLD, &reqs[nreq++]);
          }
      }
      if (neighbours[SOUTH] != MPI_PROC_NULL) {
          if (neighbours[SOUTH] == Rank) {
              #pragma omp parallel for schedule(static)
              for (i = 0; i < xsize; i++) {
                  buffers[RECV][SOUTH][i] = buffers[SEND][SOUTH][i];
              }
          } else {
              MPI_Isend(buffers[SEND][SOUTH], (int)xsize, MPI_DOUBLE, neighbours[SOUTH], TAG_S, myCOMM_WORLD, &reqs[nreq++]);
              MPI_Irecv(buffers[RECV][SOUTH], (int)xsize, MPI_DOUBLE, neighbours[SOUTH], TAG_N, myCOMM_WORLD, &reqs[nreq++]);
          }
      }

      // Add communication setup time
      double comm_setup_end = MPI_Wtime();
      comm_time += (comm_setup_end - comm_start);

      /* --------------------------------------  */
      /* INTERNAL + BORDER */
      
      // Step 1: Update INTERNAL points while communication is in progress
      double internal_start = MPI_Wtime();
      update_interior( &planes[current], &planes[!current] );
      double internal_end = MPI_Wtime();
      double internal_time_iter = internal_end - internal_start;
      internal_comp_time += internal_time_iter;
      comp_time += internal_time_iter;
      
      // Step 2: Wait for all communications to complete
      double comm_wait_start = MPI_Wtime();
      MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
      double comm_wait_end = MPI_Wtime();
      comm_time += (comm_wait_end - comm_wait_start);
      
      // Step 3: Copy received halo data back to plane (BEFORE updating borders)
      if (neighbours[WEST] != MPI_PROC_NULL && buffers[RECV][WEST] != NULL) {
          #pragma omp parallel for schedule(static)
          for (i = 0; i < ysize; i++) {
              planes[current].data[(i + 1) * xframe + 0] = buffers[RECV][WEST][i];
          }
      }
      if (neighbours[EAST] != MPI_PROC_NULL && buffers[RECV][EAST] != NULL) {
          #pragma omp parallel for schedule(static)
          for (i = 0; i < ysize; i++) {
              planes[current].data[(i + 1) * xframe + (xsize + 1)] = buffers[RECV][EAST][i];
          }
      }
      
      // Step 4: Update BORDER points with correct halo data
      double border_start = MPI_Wtime();
      update_borders( periodic, N, &planes[current], &planes[!current] );
      double border_end = MPI_Wtime();
      double border_time_iter = border_end - border_start;
      border_comp_time += border_time_iter;
      comp_time += border_time_iter;

      /* output if needed */
      if ( output_energy_stat_perstep )
	{
	  output_energy_stat ( iter, &planes[!current], injected_heat, Rank, &myCOMM_WORLD );
	  
	  // Dump binary output for visualization (only from rank 0)
	  if (Rank == 0) {
	    char filename[100];
	    sprintf( filename, "plane_%05d.bin", iter );
	    dump( planes[!current].data, planes[!current].size, filename, NULL, NULL );
	  }
	}
	
      /* swap plane indexes for the new iteration */
      current = !current;
    }
  
  total_time = MPI_Wtime() - total_time;

  // Final energy statistics (this already does MPI_Reduce internally)
  output_energy_stat ( -1, &planes[!current], Niterations * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
  
  // Reduce timing statistics
  double max_comp_time, max_comm_time, max_pack_time, max_unpack_time;
  double max_internal_comp_time, max_border_comp_time;
  MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&pack_time, &max_pack_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&unpack_time, &max_unpack_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&internal_comp_time, &max_internal_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&border_comp_time, &max_border_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  
  // Clean up the duplicated communicator
  MPI_Comm_free(&myCOMM_WORLD);

  if (Rank == 0) {
      printf("Total time: %f\n", total_time);
      printf("Initialization time: %f\n", init_time);
      printf("Computation time: %f\n", max_comp_time);
      printf("  - Internal computation: %f (%.2f%%)\n", max_internal_comp_time, (max_internal_comp_time/total_time)*100.0);
      printf("  - Border computation: %f (%.2f%%)\n", max_border_comp_time, (max_border_comp_time/total_time)*100.0);
      printf("Communication time: %f\n", max_comm_time);
      printf("Pack time: %f\n", max_pack_time);
      printf("Unpack time: %f\n", max_unpack_time);
      printf("Communication/Total ratio: %.2f%%\n", (max_comm_time/total_time)*100.0);
      printf("Computation/Total ratio: %.2f%%\n", (max_comp_time/total_time)*100.0);
      printf("Other time (overhead): %f (%.2f%%)\n", 
          total_time - max_comp_time - max_comm_time, 
          ((total_time - max_comp_time - max_comm_time)/total_time)*100.0);
      
      // Energy is already printed by output_energy_stat above
  }
  
  // CSV output system
  if (test && Rank == 0) {
      
      // Get the necessary information
      // Some are passed as arguments, others are passed through environment variables
      const char* test_type = getenv("TEST_TYPE"); // es. "Strong", "Weak", "Omp"
      const char* build_variant = getenv("BUILD_VARIANT"); // es. "ofast", "o1", "o0", "noarch"
      const char* nodes_str = getenv("SLURM_NNODES");
      const char* total_tasks_str = getenv("SLURM_NTASKS");
      const char* tasks_per_node_str = getenv("SLURM_NTASKS_PER_NODE");
      const char* threads_per_task_str = getenv("OMP_NUM_THREADS");
      
      // Check if test_type is valid
      if (test_type == NULL) {
          printf("Error: TEST_TYPE environment variable not set\n");
          return 1;
      }
      
      // Use default build variant if not specified
      if (build_variant == NULL) {
          build_variant = "ofast_omp_improved";  // default optimization level
      }
      
      // Safely convert environment variables to integers
      int nodes = nodes_str ? atoi(nodes_str) : 0;
      int total_tasks = total_tasks_str ? atoi(total_tasks_str) : 0;
      int tasks_per_node = tasks_per_node_str ? atoi(tasks_per_node_str) : 0;
      int threads_per_task = threads_per_task_str ? atoi(threads_per_task_str) : 0;

      // create filename based on test_type (use fixed-size buffer to avoid overflow)
      char filename[256];
      snprintf(filename, sizeof(filename), "data/SM3800083_%s_parallel_results.csv", test_type);

      // Create data directory if it doesn't exist
      #ifdef _WIN32
      system("mkdir data 2>nul");
      #else
      system("mkdir -p data 2>/dev/null");
      #endif

      // Open the file in append mode to add the results
      FILE *results_file = fopen(filename, "a");
      if (results_file == NULL) {
          printf("Error opening results file: %s\n", strerror(errno));
          return 1;
      }
  
      // If the file is empty, write the header
      fseek(results_file, 0, SEEK_END);
      long size = ftell(results_file);
      if (size == 0) {
          fprintf(results_file, "TestType,BuildVariant,Nodes,TotalTasks,TasksPerNode,ThreadsPerTask,EnergySources,XDim,YDim,Iterations,TotalTime,ComputationTime,InternalCompTime,BorderCompTime,CommunicationTime,InitTime\n");
      }

      // Print the data row
      fprintf(results_file, "%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
              test_type,
              build_variant,
              nodes,
              total_tasks,
              tasks_per_node,
              threads_per_task,
              Nsources,       // Number of energy sources (global)
              S[_x_], // The global X dimension
              S[_y_], // The global Y dimension
              Niterations, // The number of iterations
              total_time,     // The total time measured
              max_comp_time,   // The computation time measured
              max_internal_comp_time,  // Internal computation time
              max_border_comp_time,    // Border computation time
              max_comm_time,     // The communication time measured
              init_time);        // The initialization time measured
  
      fclose(results_file);
      
  }
  
  memory_release( &planes[0], &buffers );
  
  MPI_Finalize();
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
      
      return 0;
    }

  else return 1;
}

/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
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

int initialize ( MPI_Comm *Comm,
		 int      Me,                  // the rank of the calling process
		 int      Ntasks,              // the total number of MPI ranks
		 int      argc,                // the argc from command line
		 char   **argv,                // the argv from command line
		 vec2_t  *S,                   // the size of the plane
		 vec2_t  *N,                   // two-uint array defining the MPI tasks' grid
		 int     *periodic,            // periodic-boundary tag
		 int     *output_energy_stat,
		 int     *neighbours,          // four-int array that gives back the neighbours of the calling task
		 int     *Niterations,         // how many iterations
		 int     *Nsources,            // how many heat sources
		 int     *Nsources_local,
		 vec2_t **Sources_local,
		 double  *energy_per_source,   // how much heat per source
		 plane_t *planes,
		 buffers_t *buffers,
		 int     *injection_frequency  // frequency of energy injection
		 )
{
  int halt = 0;
  int ret;
  int verbose = 0;
  double freq = 0;
  
  // ··································································
  // set default values

  (*S)[_x_]         = 10000;
  (*S)[_y_]         = 10000;
  *periodic         = 0;
  *Nsources         = 4;
  *Nsources_local   = 0;
  *Sources_local    = NULL;
  *Niterations      = 1000;
  *energy_per_source = 1.0;
  *injection_frequency = *Niterations;  // Default: inject every iteration

  if ( planes == NULL ) {
    // manage the situation
  }

  planes[OLD].size[0] = planes[OLD].size[1] = 0;
  planes[NEW].size[0] = planes[NEW].size[1] = 0;
  
  for ( int i = 0; i < 4; i++ )
    neighbours[i] = MPI_PROC_NULL;

  for ( int b = 0; b < 2; b++ )
    for ( int d = 0; d < 4; d++ )
      (*buffers)[b][d] = NULL;
  
  // ··································································
  // process the command line
  // 
  int opt;
  while((opt = getopt(argc, argv, ":hx:y:e:E:f:n:o:p:v:")) != -1)
    {
      switch( opt )
	{
	case 'x': (*S)[_x_] = (uint)atoi(optarg);
	  break;

	case 'y': (*S)[_y_] = (uint)atoi(optarg);
	  break;

	case 'e': *Nsources = atoi(optarg);
	  break;

	case 'E': *energy_per_source = atof(optarg);
	  break;

	case 'f': freq = atof(optarg);
	  break;

	case 'n': *Niterations = atoi(optarg);
	  break;

	case 'o': *output_energy_stat = (atoi(optarg) > 0);
	  break;

	case 'p': *periodic = (atoi(optarg) > 0);
	  break;

	case 'v': verbose = atoi(optarg);
	  break;

	case 'h': {
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
  
  // ··································································
  /*
   * find a suitable domain decomposition
   * very simple algorithm, you may want to
   * substitute it with a better one
   *
   * the plane Sx x Sy will be solved with a grid
   * of Nx x Ny MPI tasks
   */

  vec2_t Grid;
  double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_]/(*S)[_y_] : (double)(*S)[_y_]/(*S)[_x_] );
  int    dimensions = 2 - (Ntasks <= ((int)formfactor+1) );

  
  if ( dimensions == 1 )
    {
      if ( (*S)[_x_] >= (*S)[_y_] )
	Grid[_x_] = Ntasks, Grid[_y_] = 1;
      else
	Grid[_x_] = 1, Grid[_y_] = Ntasks;
    }
  else
    {
      int   Nf;
      uint *factors;
      uint  first = 1;
      ret = simple_factorization( Ntasks, &Nf, &factors );
      
      for ( int i = 0; (i < Nf) && ((Ntasks/first)/first > formfactor); i++ )
	first *= factors[i];

      if ( (*S)[_x_] > (*S)[_y_] )
	Grid[_x_] = Ntasks/first, Grid[_y_] = first;
      else
	Grid[_x_] = first, Grid[_y_] = Ntasks/first;
    }

  (*N)[_x_] = Grid[_x_];
  (*N)[_y_] = Grid[_y_];
  
  // ··································································
  // my coordinates in the grid of processors
  //
  int X = Me % Grid[_x_];
  int Y = Me / Grid[_x_];

  // ··································································
  // find my neighbours
  //

  if ( *periodic ) {
      // Periodic boundary conditions
      if ( Grid[_x_] > 1 ) {
	int eastX = (X + 1) % Grid[_x_];
	int westX = (X - 1 + Grid[_x_]) % Grid[_x_];
	neighbours[EAST] = Y * Grid[_x_] + eastX;
	neighbours[WEST] = Y * Grid[_x_] + westX;
      }
      if ( Grid[_y_] > 1 ) {
	int northY = (Y - 1 + Grid[_y_]) % Grid[_y_];
	int southY = (Y + 1) % Grid[_y_];
	neighbours[NORTH] = northY * Grid[_x_] + X;
	neighbours[SOUTH] = southY * Grid[_x_] + X;
      }
  } else {
      // Non-periodic boundary conditions
      if ( Grid[_x_] > 1 ) {
	neighbours[EAST]  = ( X < Grid[_x_]-1 ? Me+1 : MPI_PROC_NULL );
	neighbours[WEST]  = ( X > 0 ? Me-1 : MPI_PROC_NULL ); 
      }
      if ( Grid[_y_] > 1 ) {
	neighbours[NORTH] = ( Y > 0 ? Me - Grid[_x_]: MPI_PROC_NULL );
	neighbours[SOUTH] = ( Y < Grid[_y_]-1 ? Me + Grid[_x_] : MPI_PROC_NULL );
      }
  }

  // ··································································
  // the size of my patch
  //

  vec2_t mysize;
  uint s = (*S)[_x_] / Grid[_x_];
  uint r = (*S)[_x_] % Grid[_x_];
  mysize[_x_] = s + (X < r);
  s = (*S)[_y_] / Grid[_y_];
  r = (*S)[_y_] % Grid[_y_];
  mysize[_y_] = s + (Y < r);

  planes[OLD].size[0] = mysize[0];
  planes[OLD].size[1] = mysize[1];
  planes[NEW].size[0] = mysize[0];
  planes[NEW].size[1] = mysize[1];
  
  if ( verbose > 0 )
    {
      if ( Me == 0 ) {
	printf("Tasks are decomposed in a grid %d x %d\n\n",
		 Grid[_x_], Grid[_y_] );
	fflush(stdout);
      }

      MPI_Barrier(*Comm);
      
      for ( int t = 0; t < Ntasks; t++ )
	{
	  if ( t == Me )
	    {
	      printf("Task %4d :: "
		     "\tgrid coordinates : %3d, %3d\n"
		     "\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
		     Me, X, Y,
		     neighbours[NORTH], neighbours[EAST],
		     neighbours[SOUTH], neighbours[WEST] );
	      fflush(stdout);
	    }

	  MPI_Barrier(*Comm);
	}
    }

  // ··································································
  // allocate the needed memory
  //
  ret = memory_allocate( neighbours, *N, buffers, planes );

  // ··································································
  // allocate the heat sources
  //
  ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
  
  return 0;  
}

uint simple_factorization( uint A, int *Nfactors, uint **factors )
{
  int N = 0;
  int f = 2;
  uint _A_ = A;

  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	N++;
	_A_ /= f; }

      f++;
    }

  *Nfactors = N;
  uint *_factors_ = (uint*)malloc( N * sizeof(uint) );

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

int initialize_sources( int       Me,
			int       Ntasks,
			MPI_Comm *Comm,
			vec2_t    mysize,
			int       Nsources,
			int      *Nsources_local,
			vec2_t  **Sources )
{
  srand48(42);
  int *tasks_with_sources = (int*)malloc( Nsources * sizeof(int) );
  
  if ( Me == 0 )
    {
      for ( int i = 0; i < Nsources; i++ )
	tasks_with_sources[i] = (int)lrand48() % Ntasks;
    }
  
  MPI_Bcast( tasks_with_sources, Nsources, MPI_INT, 0, *Comm );

  int nlocal = 0;
  for ( int i = 0; i < Nsources; i++ )
    nlocal += (tasks_with_sources[i] == Me);
  *Nsources_local = nlocal;
  
  if ( nlocal > 0 )
    {
      vec2_t * restrict helper = (vec2_t*)malloc( nlocal * sizeof(vec2_t) );      
      for ( int s = 0; s < nlocal; s++ )
	{
	  helper[s][_x_] = 1 + lrand48() % mysize[_x_];
	  helper[s][_y_] = 1 + lrand48() % mysize[_y_];
	}

      *Sources = helper;
    }
  
  free( tasks_with_sources );

  return 0;
}

int memory_allocate ( const int       *neighbours  ,
		      const vec2_t     N           ,
		            buffers_t *buffers_ptr ,
		            plane_t   *planes_ptr
		      )
{
  if (planes_ptr == NULL )
    return -1;

  if (buffers_ptr == NULL )
    return -1;
    
  // ··················································
  // allocate memory for data
  // we allocate the space needed for the plane plus a contour frame
  // that will contains data form neighbouring MPI tasks
  unsigned int frame_size = (planes_ptr[OLD].size[_x_]+2) * (planes_ptr[OLD].size[_y_]+2);

  planes_ptr[OLD].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[OLD].data == NULL )
    return -1;
  // memset ( planes_ptr[OLD].data, 0, frame_size * sizeof(double) );
  
  //? first-touch allocation with OpenMP
  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < frame_size; i++) {
    planes_ptr[OLD].data[i] = 0.0;
  }

  planes_ptr[NEW].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[NEW].data == NULL )
    return -1;
  // memset ( planes_ptr[NEW].data, 0, frame_size * sizeof(double) );
  
  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < frame_size; i++) {
    planes_ptr[NEW].data[i] = 0.0;
  }

  // ··················································
  // allocate buffers for east/west communication
  // (north/south are contiguous and don't need separate buffers)
  //
  // Initialize all buffer pointers to NULL first
  for ( int b = 0; b < 2; b++ )
    {
      for ( int d = 0; d < 4; d++ ) {
        (*buffers_ptr)[b][d] = NULL;
      }
    }

  // Allocate East/West buffers
  if (neighbours[EAST] != MPI_PROC_NULL) {
      (*buffers_ptr)[SEND][EAST] = (double*)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
      (*buffers_ptr)[RECV][EAST] = (double*)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
      if ((*buffers_ptr)[SEND][EAST] == NULL || (*buffers_ptr)[RECV][EAST] == NULL) {
          printf("Error: failed to allocate memory for the EAST buffers\n");
          return -1;
      }
  }
  if (neighbours[WEST] != MPI_PROC_NULL) {
      (*buffers_ptr)[SEND][WEST] = (double*)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
      (*buffers_ptr)[RECV][WEST] = (double*)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
      if ((*buffers_ptr)[SEND][WEST] == NULL || (*buffers_ptr)[RECV][WEST] == NULL) {
          printf("Error: failed to allocate memory for the WEST buffers\n");
          return -1;
      }
  }
  
  // North/South use direct pointers (no separate allocation needed)
  // This will be set up in the communication loop
  
  return 0;
}

int memory_release ( plane_t   *planes,
		     buffers_t *buffers )
{
  if ( planes != NULL )
    {
      if ( planes[OLD].data != NULL )
	free (planes[OLD].data);
      
      if ( planes[NEW].data != NULL )
	free (planes[NEW].data);
    }

  if ( buffers != NULL )
    {
      // Only free East/West buffers (North/South use direct pointers)
      for ( int b = 0; b < 2; b++ )
	{
	  if ( (*buffers)[b][EAST] != NULL )
	    free ((*buffers)[b][EAST]);
	  if ( (*buffers)[b][WEST] != NULL )
	    free ((*buffers)[b][WEST]);
	  // North/South are direct pointers, don't free them
	}
    }
      
  return 0;
}

int output_energy_stat ( int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm )
{
  double system_energy = 0;
  double tot_system_energy = 0;
  get_total_energy ( plane, &system_energy );
  
  MPI_Reduce ( &system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm );
  
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

