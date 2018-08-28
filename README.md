# ModularityDensity

If you use this code for your research, I kindly ask you to cite Ref. 1 in your publications.

</div>
For questions, requests or notifications, please write to f.botta@warwick.ac.uk.

&nbsp;
<h2>USAGE</h2>
The code consists of the files Main. c, ParallelisedOptimisedModDens.h and ParallelisedOptimisedModDens.c. To use the code, simply include ParallelisedOptimisedModDens.h in your code, and compile ParallelisedOptimisedModDens.c with your other files, bearing in mind to use the option -lm to link the math library. The OpenMP library can be included simply by removing the comment in the file ParallelisedOptimised.h.

The algorithm receives as input two variables from command line: N, the number of nodes in the network, and the number of iterations of the algorithm that you want to run. In order to run the code on your own data, remember to change the file name of the input file storing the adjacency matrix.

The algorithm initially sets the tolerance parameters (see Ref. 1 for a detailed explanation of these parameters). In general, the larger the network size, the smaller you should set the tolerance parameters. However, a rigorous analysis should explore a range of values of the tolerances. The code then allocates the memory needed for the analysis calling the function prepare_memory(). After having read the input file, the community detection algorithm can be called with:

community_detection_via_modularity_density_maximisation(double * values_modularity_density, int iteration)

where values_modularity_density is a pointer to an array that stores the detected values of modularity density, which are then available for further analysis if needed; iteration is simply the index of the current iteration of the algorithm. In the available code, the value of modularity density and the partition detected are saved to file at each iteration invoking the function write_to_file(iteration) at the bottom of the code in ParallelisedOptimisedModDens.c.

At the end of the algorithm, the memory is automatically cleaned with the function free_memory() invoked at the bottom of the code in ParallelisedOptimisedModDens.c.

In the available code, there is a simple proof of concept that uses the Dolphin network dataset (D. Lusseau, K. Schneider, O. J. Boisseau, P. Haase, E. Slooten, and S. M. Dawson, Behavioral Ecology and Sociobiology 54, 396-405, 2003). This network has 62 nodes.

To compile the code, use the commands:

gcc -lm -o executable_file ParallelisedOptimisedModDens.c Main.c

You can then run it with:

./executable_file [number_of_nodes] [number_of_iterations]

For the proof of concept, you should use [number_of_nodes]=62.

&nbsp;
<h2>REFERENCES</h2>
Botta F, Del Genio CI. Finding network communities using modularity density. J Stat Mech - Theory E. (2016),�123402.

