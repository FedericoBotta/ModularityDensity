/*
 * Implementation of a community detection algorithm for unweighted and
 * undirected networks; the algorithm is based on modularity density
 * maximisation and it consists of several steps:
 * 
 * 1. initial spectral bisection with modularity eigenvector
 * 2. fine tuning of the bisection
 * 3. final tuning on the whole network
 * 4. agglomeration step on the whole network
 * 
 * This is the header file that defines the global variables, structure
 * definitions and prototypes of the functions.
*/


/* Header files to include */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
//#include <omp.h>


/* Global variables needed by the algorithm:
 * - N is the number of nodes
 * - number_edges is the total number of edges in the network
 * - current_number_communities keeps track of the number of communities
 *   during the execution of the algorithm; its value changes every time
 *   we add/remove one community
 * - working_community keeps track of the community that the algorithm
 *   is currently working on
 * - degree_sequence is an array of pointers that keeps track of the
 *   degree sequence of the network
 * - community_size is a vector of length given by the current number
 *   of communities; each entry of this vector stores the size of the
 *   corresponding community
 * - community_blocked is a vector of length given by the current number
 *   of communities; each entry is a flag variable that indicates
 *   whether the corresponding community is blocked or not, i.e. it has
 *   undergone the bisection and fine tuning steps
 * - adj_mat is a pointer to a pointer that stores the adjacency matrix
 * - adj_list is a pointer to a pointer that stores the adjacency list
 * - edge_community matrix is a matrix of size (number of nodes) times
 *   current number of communities. Each entry stores the number of
 *   connections of each node to all the other (current) communities
 * - community_density is a square matrix of size given by the current
 *   number of communities; each entry stores the link between the two
 *   communities (therefore internal links of each community will be
 *   stored on the diagonal of this matrix)
 * - dq_bisec keeps track of the change in modularity density given by
 *   the bisection step
 * - dq_fine keeps track of the change in modularity density given by 
 *   the fine tuning step
 * - dq_final keeps track of the change in modularity density given by
 *   the final tuning step
 * - dq_agglom keeps track of the change in modularity density given by
 *   the agglomeration step
 * - dq keeps track of the overall change in modularity density, given 
 *   by the sum of all the changes produced by the various steps
 * - toler is the calculation tolerance parameter 
 * - toler_pwm is the tolerance parameter for the calculation of the
 *   leading eigenvalue/eigenvector in the power method algorithm
 * - toler_bisec is the tolerance parameter used to decide whether to
 *   keep the partition obtained in the bisection step or not, since
 *   the bisection based on the modularity eigenvalue might results in
 *   a decrease of the value of modularity density
 */
int N, number_edges, current_number_communities, working_community;
int *degree_sequence, *community_size, *community_blocked;
int **adj_mat, **adj_list;
long int **edge_community_matrix, **community_density;
double dq_bisec, dq_fine, dq_final, dq_agglom, dq, toler,toler_pwm,toler_bisec;

/* Definition of a structure that stores the best partition, its
 * modularity density value and the zscore found at the end of the
 * algorithm
 */
typedef struct{
	double modularity_density;
	double z_score;
	int *community_list;
} community_struct;

/* Global structure to store the result of the community detection */
static community_struct results;


/* Prototypes of functions */

//Computes leading eigenvalue and eigenvector of modularity matrix
void power_method(double **mod_mat, int number_nodes, double *leading_eigenvalue, double *leading_eigenvector);
//Allocates memory for the global variables
void prepare_memory();
//Creates degree sequence from adjacency matrix and computes the total
//number of edges in the network
void create_degree_sequence();
//Creates adjacency list from adjacency matrix
void adjlist_from_adjmat();
//Creates adjacency matrix from adjacency list
void adjmat_from_adjlist();
//Creates sub modularity matrix for a subset of the nodes
void create_sub_modularity_matrix(double ***mod_mat, int *subgraph_nodes, int number_nodes);
//Computes the value of modularity density
void compute_modularity_density(double *q_ds);
//Updates the edge_community_matrix
void update_edge_community_matrix(int node_to_move, int old_community_label, int new_community_label);
//Updates the community_density matrix
void update_community_density_matrix(int node_to_move, int old_community_label, int new_community_label);
//Updates the community_size vector
void update_community_size_vector(int old_community_label, int new_community_label);
//Computes the change in modularity density from tuning steps when old_community_label>new_community_label
void compute_change_modularity_density_tuning_steps_1(int node_to_move, int old_community_label, int new_community_label, double *local_dq, long int **temporary_community_density,long int **temporary_edge_community_matrix, int *temporary_community_size,double *contributions_to_modularity_density,double *split_penalty);
//Computes the change in modularity density from tuning steps when old_community_label<new_community_label
void compute_change_modularity_density_tuning_steps_2(int node_to_move, int old_community_label, int new_community_label, double *local_dq, long int **temporary_community_density,long int **temporary_edge_community_matrix, int *temporary_community_size,double *contributions_to_modularity_density,double *split_penalty);
//Allocates the memory for a new community
void create_new_community();
//Bisects the eigenvector found with the power method according to sign
void bisection(double *leading_eigenvector, int *nodes, int current_community_size);
//Fine tuning step of the algorithm (Kernighan-Lie)
void fine_tuning(int *current_nodes, int number_nodes, int community_label_1, int community_label_2);
//Final tuning step of the algorithm
void final_tuning();
//Compute the change in modularity density from agglomeration step
void compute_change_modularity_density_agglomeration( int first_community, int second_community, int *flag_community_blocked, double *local_dq, long int **temporary_community_density,long int **temporary_edge_community_matrix, int *temporary_community_size,double *contributions_to_modularity_density,double *split_penalty);
//Merge two communities
void merge_communities(int first_community_to_merge, int second_community_to_merge);
//Allocates the correct memory after a whole agglomeration step
void merge_memory_after_agglomeration_step(int communities_cancelled[current_number_communities], int new_number_communities);
//Agglomeration step of the algorithm
void agglomeration();
//Free the memory allocated during the algorithm
void free_memory();
//Write the results to file
void write_to_file(int flag_index);
//Implementation of the algorithm to detect communities via modularity
//density maximisation
void community_detection_via_modularity_density_maximisation(double * values_modularity_density, int iteration);
