/*
 * Main.c
 * 
 * Copyright 2015 Federico Botta 
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * To compile:
 * gcc -lm -o executable_file ParallelisedOptimisedModDens.c Main.c
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ParallelisedOptimisedModDens.h"

int main(int argc, char *argv[])
{
	//seed the random number generator
	srand((long int) time(NULL));
	
	//initialise the tolerance parameters
	toler=0.0001;
	toler_pwm=0.0001;
	toler_bisec=0.1;
	double *values_modularity_density;
	values_modularity_density=malloc(100*sizeof(double));

	//variable received from command line
	
	//number of nodes
	N=atoi(argv[1]);

	int index1,index2;

	//number of iterations of the algorithm
	int number_iteration;
	number_iteration=atoi(argv[2]);

	int iteration;
	
	for (iteration=0;iteration<number_iteration;iteration++){
		
		//prepare the memory for the global variables
		prepare_memory();
		//initialise the adjacency matrix
		int i,j;
		for (i=0;i<N;i++){
			for (j=0;j<N;j++){
				adj_mat[i][j]=0;
			}
		}
					
		FILE *reading_file;
		reading_file=fopen("DolphinsAdjMat.txt","r");
				
		for (i=0;i<N;i++){
			for (j=0;j<N;j++){
				fscanf(reading_file,"%i\t",&adj_mat[i][j]);
			}
		}
		fclose(reading_file);
		community_detection_via_modularity_density_maximisation(values_modularity_density,iteration);
	}


	
	//FILE *modularity_density_file;
	//modularity_density_file = fopen("modularity_density_dolphins.txt","w");

//	for (iteration=0; iteration<100;iteration++){
//		fprintf(modularity_density_file, "%lf\n", values_modularity_density[iteration]);
//	}
//	fclose(modularity_density_file);
	
	free(values_modularity_density);
	
	return 0;
}

