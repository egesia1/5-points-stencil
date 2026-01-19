/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <float.h>
#include <math.h>



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



// ============================================================
//
// function prototypes

int initialize ( int      ,
		 char   **,
		 int    *,
		 int     *,
		 int     *,
		 int     *,
		 int   **,
		 double  *,
		 double **,
                 int     *,
                 int     *
		 );

int memory_release ( double *, int * );


extern int inject_energy ( const  int,
                           const int    ,
			   const int   *,
			   const double  ,
			   const int    [2],
                                 double * );

extern int update_plane ( const int       ,
			  const int    [2],
			  const double   *,
		                double   * );


extern int get_total_energy( const int     [2],
                             const double *,
                             double * );

extern int get_border_dissipation( const int     periodic,
                                    const int     size[2],
                                    const double *plane,
                                    double       *dissipation );


// ============================================================
//
// function definition for inline functions

inline int inject_energy ( const int     periodic,
                           const int     Nsources,
			   const int    *Sources,
			   const double  energy,
			   const int     mysize[2],
                           double *plane )
{
   #define IDX( i, j ) ( (j)*(mysize[_x_]+2) + (i) )
    for (int s = 0; s < Nsources; s++) {
        
        int x = Sources[2*s];
        int y = Sources[2*s+1];
        plane[IDX(x, y)] += energy;

        if ( periodic )
            {
                if ( x == 1 )
                    plane[IDX(mysize[_x_]+1, y)] += energy;
                if ( x == mysize[_x_] )
                    plane[IDX(0, y)] += energy;
                if ( y == 1 )
                    plane[IDX(x, mysize[_y_]+1)] += energy;
                if ( y == mysize[_y_] )
                    plane[IDX(x, 0)] += energy;
            }
    }
   #undef IDX
    
    return 0;
}




inline int update_plane ( const int     periodic, 
                          const int     size[2],
			  const double *old    ,
                                double *new    )
/*
 * calculate the new energy values
 * the old plane contains the current data, the new plane
 * will store the updated data
 *
 * NOTE: in parallel, every MPI task will perform the
 *       calculation for its patch
 *
 */
{
    register const int fxsize = size[_x_]+2;
    register const int xsize = size[_x_];
    register const int ysize = size[_y_];
    
   #define IDX( i, j ) ( (j)*fxsize + (i) )

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    //
    // HINT: in any case, this loop is a good candidate
    //       for openmp parallelization
    
    // OpenMP parallelization of the stencil computation
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= ysize; j++)
        for ( int i = 1; i <= xsize; i++)
            {
                //
                // five-points stencil formula
                //

                
                // simpler stencil with no explicit diffusivity
                // always conserve the smoohed quantity
                // alpha here mimics how much "easily" the heat
                // travels
                
                double alpha = 0.6;
                double constant =  (1-alpha) / 4.0;
                double result = old[ IDX(i,j) ] *alpha + (old[IDX(i-1, j)] + old[IDX(i+1, j)] + old[IDX(i, j-1)] + old[IDX(i, j+1)]) * constant;
                

                /*

                  // implentation from the derivation of
                  // 3-points 2nd order derivatives
                  // however, that should depends on an adaptive
                  // time-stepping so that given a diffusivity
                  // coefficient the amount of energy diffused is
                  // "small"
                  // however the imlic methods are not stable
                  
               #define alpha_guess 0.5     // mimic the heat diffusivity

                double alpha = alpha_guess;
                double sum = old[IDX(i,j)];
                
                int   done = 0;
                do
                    {                
                        double sum_i = alpha * (old[IDX(i-1, j)] + old[IDX(i+1, j)] - 2*sum);
                        double sum_j = alpha * (old[IDX(i, j-1)] + old[IDX(i, j+1)] - 2*sum);
                        result = sum + ( sum_i + sum_j);
                        double ratio = fabs((result-sum)/(sum!=0? sum : 1.0));
                        done = ( (ratio < 2.0) && (result >= 0) );    // not too fast diffusion and
                                                                     // not so fast that the (i,j)
                                                                     // goes below zero energy
                        alpha /= 2;
                    }
                while ( !done );
                */

                new[ IDX(i,j) ] = result;
                
            }

    if ( periodic )
        /*
         * propagate boundaries if they are periodic
         *
         * NOTE: when is that needed in distributed memory, if any?
         */
        {
            for ( int i = 1; i <= xsize; i++ )
                {
                    new[ IDX(i, 0) ] = new[ IDX(i, ysize) ];
                    new[ IDX(i, ysize+1) ] = new[ IDX(i, 1) ];
                }
            for ( int j = 1; j <= ysize; j++ )
                {
                    new[ IDX( 0, j) ] = new[ IDX(xsize, j) ];
                    new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ];
                }
        }
    
    return 0;

   #undef IDX
}

 

inline int get_total_energy( const int     size[2],
                             const double *plane,
                                   double *energy )
/*
 * NOTE: this routine a good candiadate for openmp
 *       parallelization
 */
{

    register const int xsize = size[_x_];
    
   #define IDX( i, j ) ( (j)*(xsize+2) + (i) )

   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    
    // OpenMP parallelization for energy calculation
    #pragma omp parallel for schedule(static) reduction(+:totenergy)
    for ( int j = 1; j <= size[_y_]; j++ )
        for ( int i = 1; i <= size[_x_]; i++ )
            totenergy += plane[ IDX(i, j) ];
    
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}

inline int get_border_dissipation( const int     periodic,
                                    const int     size[2],
                                    const double *plane,
                                    double       *dissipation )
/*
 * Calculate energy dissipation at borders for non-periodic boundaries.
 * The dissipation occurs because border cells exchange energy with halo cells (which are 0).
 * This function calculates the energy flow from border cells to the halo.
 */
{
    if (periodic) {
        *dissipation = 0.0;
        return 0;
    }

    register const int xsize = size[_x_];
    register const int ysize = size[_y_];
    register const int fxsize = xsize + 2;
    
   #define IDX( i, j ) ( (j)*fxsize + (i) )

    double alpha = 0.6;
    double beta = (1.0 - alpha) / 4.0;  // coefficient for neighbor contribution
    double total_dissipation = 0.0;

    // North border (j=1): energy flows to halo at j=0 (which is 0)
    for (int i = 1; i <= xsize; i++) {
        // Energy that would flow from cell (i,1) to halo (i,0)
        total_dissipation += plane[IDX(i, 1)] * beta;
    }

    // South border (j=ysize): energy flows to halo at j=ysize+1
    for (int i = 1; i <= xsize; i++) {
        total_dissipation += plane[IDX(i, ysize)] * beta;
    }

    // West border (i=1): energy flows to halo at i=0
    for (int j = 1; j <= ysize; j++) {
        total_dissipation += plane[IDX(1, j)] * beta;
    }

    // East border (i=xsize): energy flows to halo at i=xsize+1
    for (int j = 1; j <= ysize; j++) {
        total_dissipation += plane[IDX(xsize, j)] * beta;
    }

   #undef IDX

    *dissipation = total_dissipation;
    return 0;
}
                            
