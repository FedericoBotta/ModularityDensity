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
 * This is the implementation of the algorithm; it require the file
 * ModDens.h to be included, where all the global variables, structure
 * definitions and prototypes of functions are defined.
 */

#include "ParallelisedOptimisedModDens.h"

/* This function implements the power method to find the leading
 * eigenvalue of a matrix and the corresponding eigenvector.
 * In this particular implementation, it receives five inputs:
 * 1. the modularity matrix (note that this may correspond to only part
 *    of the network, since the algorithm might be trying to partition
 *    a subset of nodes). The matrix is passed by reference
 * 2. the number of nodes (i.e. the size of the modularity  matrix
 *    considered)
 * 3. a pointer to a variable that will store the leading eigenvalue
 *    found by the power method
 * 4. a pointer to a vector that will store the eigenvector
 *    corresponding to the leading eigenvalue found
 * 5. a variable that keeps track of the shifts to the matrix
 * 
 * The function inizialises the eigenvector randomly, then normalises
 * it, and then performs the power method algorithm.
 */
void power_method(double **mod_mat, int number_nodes, double *leading_eigenvalue, double *leading_eigenvector)
{
	//Vectors used in the power method
	double guess_vector[number_nodes], temp[number_nodes];	
	double norm=0, old_norm=0, shift=0;		//Norm of the vectors used
	int i=0, j=0;
	//Make a random guess for the initial vector and find squared norm
	for (i=0;i<number_nodes;i++){
		guess_vector[i]= (double)rand()/(double)RAND_MAX;//insert random number generator here
		norm=norm+pow(guess_vector[i],2.0);
	}
	norm=sqrt(norm);		//Take square root to find the norm
	//Normalise initial vector
	for (i=0;i<number_nodes;i++){
		guess_vector[i]=guess_vector[i]/norm;
	}
	//Power method algorithm
	while (1){
		#pragma omp parallel private(j)
		{
			#pragma omp for
			for (i=0;i<number_nodes;i++){
				temp[i]=0;
				for (j=0;j<number_nodes;j++){
					temp[i]=temp[i]+mod_mat[i][j]*guess_vector[j];
				}
			}
		}
		old_norm=norm;
		norm=0;
		for (i=0;i<number_nodes;i++){
			norm=norm+temp[i]*temp[i];
		}
		norm=sqrt(norm);
		for (i=0;i<number_nodes;i++){
			guess_vector[i]=temp[i]/norm;
		}
		if (fabs(old_norm-norm)<toler_pwm){
			break;
		}
	}
	/* If the leading eigenvalue found is smaller than zero and we have
	 * not shifted the matrix yet, then shift by a value corresponding
	 * to the eigenvalue found and perform again the power method on
	 * the shifted matrix.
	 */
	int flag_condition=0;
	double temp_eig=0.0;
	for (i=0;i<number_nodes;i++){
		temp_eig=temp_eig+mod_mat[0][i]*guess_vector[i];
	}
	if ((temp_eig<0 && guess_vector[0]>0)||(temp_eig>0 && guess_vector[0]<0)){
		for (i=0;i<number_nodes;i++){
			mod_mat[i][i]=mod_mat[i][i]+norm;
		}
		shift=shift+norm;
		
		//Make a random guess for the initial vector and find squared norm
		for (i=0;i<number_nodes;i++){
			guess_vector[i]= (double)rand()/(double)RAND_MAX;//insert random number generator here
			norm=norm+pow(guess_vector[i],2.0);
		}
		norm=sqrt(norm);		//Take square root to find the norm
		//Normalise initial vector
		for (i=0;i<number_nodes;i++){
			guess_vector[i]=guess_vector[i]/norm;
		}
		//Power method algorithm
		while (1){
			for (i=0;i<number_nodes;i++){
				temp[i]=0;
				for (j=0;j<number_nodes;j++){
					temp[i]=temp[i]+mod_mat[i][j]*guess_vector[j];
				}
			}
			old_norm=norm;
			norm=0;
			for (i=0;i<number_nodes;i++){
				norm=norm+temp[i]*temp[i];
			}
			norm=sqrt(norm);
			for (i=0;i<number_nodes;i++){
				guess_vector[i]=temp[i]/norm;
			}
			if (fabs(old_norm-norm)<toler_pwm){
				break;
			}
		}
		flag_condition=1;
	}
	
	/* Perform one hundred additional steps of the power method to check
	 * that the eigenvalue it converged does not oscillate between
	 * ones of similar magnitude but different signs. If it does, then
	 * perform a shift of the matrix and perform the power method again
	 * on the shifted matrix. Note that this additional condition is
	 * checked only if the matrix has not already been shifted. If the
	 * matrix has already been shifted in the previous part of this
	 * function, all the eigenvalues should be positive and therefore
	 * there should be no oscillations.
	 */
	 if (flag_condition==0){
		 int stability_count=0;
		 int stability_flag=0;
		 for (stability_count=0;stability_count<100;stability_count++){
			 for (i=0;i<number_nodes;i++){
				temp[i]=0;
				for (j=0;j<number_nodes;j++){
					temp[i]=temp[i]+mod_mat[i][j]*guess_vector[j];
				}
			}
			old_norm=norm;
			norm=0;
			for (i=0;i<number_nodes;i++){
				norm=norm+temp[i]*temp[i];
			}
			norm=sqrt(norm);
			for (i=0;i<number_nodes;i++){
				guess_vector[i]=temp[i]/norm;
			}
			if (fabs(old_norm-norm)>2*toler_pwm){
				stability_flag=1;
			}
		}
		if (stability_flag==1){
			for (i=0;i<number_nodes;i++){
				mod_mat[i][i]=mod_mat[i][i]+norm;
			}
			shift=shift+norm;
			//Make a random guess for the initial vector and find squared norm
			for (i=0;i<number_nodes;i++){
				guess_vector[i]= (double)rand()/(double)RAND_MAX;//insert random number generator here
				norm=norm+pow(guess_vector[i],2.0);
			}
			norm=sqrt(norm);		//Take square root to find the norm
			//Normalise initial vector
			for (i=0;i<number_nodes;i++){
				guess_vector[i]=guess_vector[i]/norm;
			}
			//Power method algorithm
			while (1){
				for (i=0;i<number_nodes;i++){
					temp[i]=0;
					for (j=0;j<number_nodes;j++){
						temp[i]=temp[i]+mod_mat[i][j]*guess_vector[j];
					}
				}
				old_norm=norm;
				norm=0;
				for (i=0;i<number_nodes;i++){
					norm=norm+temp[i]*temp[i];
				}
				norm=sqrt(norm);
				for (i=0;i<number_nodes;i++){
					guess_vector[i]=temp[i]/norm;
				}
				if (fabs(old_norm-norm)<toler_pwm){
					break;
				}
			}
		}
	}

	*leading_eigenvalue=norm-fabs(shift);	//Leading eigenvalue
	//Eigenvector associated with leading eigenvalue
	for (i=0;i<number_nodes;i++){
		leading_eigenvector[i]=guess_vector[i];
	}

	return;
}

/* This function allocates the necessary memory for the global
 * variables that are needed in the algorithm
 */
void prepare_memory()
{
	degree_sequence=malloc(N*sizeof(int));	//memory for degree sequence
	adj_mat=malloc(N*sizeof(int*));	//memory for adjacency matrix
	int i=0;
	for (i=0;i<N;i++){
		adj_mat[i]=malloc(N*sizeof(int));
	}
	adj_list=malloc(N*sizeof(int*));	//memory for adjacency list
	
	community_size=malloc(sizeof(int));	//memory for community size vector
	community_blocked=malloc(sizeof(int));	//memory for flag community blocked vector
	edge_community_matrix=malloc(N*sizeof(long int*));	//memory for edge
													//community matrix
	for (i=0;i<N;i++)edge_community_matrix[i]=malloc(sizeof(long int));
	
	community_density=malloc(sizeof(long int*));	//memory for community density
	community_density[0]=malloc(sizeof(long int));
	
	results.community_list=malloc(N*sizeof(int));	//memory for community
													//assignment list
	return;
}


/* This function computes the degree sequence of the network, starting
 * from the adjacency matrix. Note that the degree sequence array
 * is defined globally and named degree_sequence. It also computes the
 * overall number of edges in the network, stored in the global
 * variable number_edges.
 */
void create_degree_sequence()
{
	int row, column;
	number_edges=0;
	for (row=0;row<N;row++){
		degree_sequence[row]=0;		//to initialize the array with zeros
		for (column=0;column<N;column++){
			degree_sequence[row]=degree_sequence[row]+adj_mat[row][column];
		}
		number_edges=number_edges+degree_sequence[row];
	}
	number_edges=number_edges/2;	//the sum of the degrees gives twice
									//the number of edges	
	return;
}

/* This function creates an adjacency list from the adjacency matrix;
 * it also uses the global array degree_sequence.
 */
void adjlist_from_adjmat()
{
	int row, column, edge_index;
	for (row=0; row<N;row++){
		adj_list[row]=malloc(degree_sequence[row]*sizeof(int));
		edge_index=0;
		for (column=0;column<N;column++){
			if(adj_mat[row][column]==1){
				adj_list[row][edge_index]=column;
				edge_index=edge_index+1;
			}
		}
	}
	
	return;
}

/* This function creates the adjacency matrix from an adjacency list;
 * it uses the global array degre_sequence.
 */
void adjmat_from_adjlist()
{
	 int node, edge;
	 //Initialise the adjacency matrix entries to zero
	 for (node=0;node<N;node++){
		 for (edge=0;edge<N;edge++){
			 adj_mat[node][edge]=0;
		 }
	 }
	 //Fill in the adjacency matrix from adjacency list
	 for (node=0; node<N; node++){
		 for (edge=0;edge<degree_sequence[node];edge++){
			 adj_mat[node][adj_list[node][edge]]=1;
		 }
	 }
	 
	 return;
}

/* This function creates the sub-modularity matrix for a specific subset
 * of nodes that it receives as input. 
 */
void create_sub_modularity_matrix(double ***mod_mat, int *subgraph_nodes, int number_nodes)
{
	*mod_mat=(double **)malloc(number_nodes*sizeof(double *));
	//degree sequence of the nodes only considering connections in the
	//subset of nodes under consideration
	int sub_degree_sequence[number_nodes];
	//sum of overall degrees of nodes in the subgraph considered
	int sum_degree_nodes=0;
	
	int i,j;
	
	for (i=0;i<number_nodes;i++){
		//initialise the sub_degree_sequence to zero
		sub_degree_sequence[i]=0;
		for (j=0;j<number_nodes;j++){
			//sum over the connections between two nodes belonging to
			//the subgraph under consideration
			sub_degree_sequence[i]=sub_degree_sequence[i]+adj_mat[subgraph_nodes[i]][subgraph_nodes[j]];			
		}
		//update the sum of the degrees
		sum_degree_nodes=sum_degree_nodes+sub_degree_sequence[i];
	}
	
	//construct the sub modularity matrix
	for (i=0;i<number_nodes;i++) (*mod_mat)[i]=(double *)malloc(number_nodes*sizeof(double));
	
	for(i=0;i<number_nodes;i++){
		(*mod_mat)[i][i]=(double)adj_mat[subgraph_nodes[i]][subgraph_nodes[i]]-(double)(degree_sequence[subgraph_nodes[i]]*degree_sequence[subgraph_nodes[i]])/((double)(2*number_edges))-(double)sub_degree_sequence[i]+((double)degree_sequence[subgraph_nodes[i]]*(double)sum_degree_nodes)/((double)(2*number_edges));
		for(j=i+1;j<number_nodes;j++){
			(*mod_mat)[i][j]=(double)adj_mat[subgraph_nodes[i]][subgraph_nodes[j]]-(double)(degree_sequence[subgraph_nodes[i]]*degree_sequence[subgraph_nodes[j]])/((double)(2*number_edges));
			(*mod_mat)[j][i]=(*mod_mat)[i][j];
		}
	}
	
	return;
}
/* This function allocates the memory for a new community. It updates
 * the current number of communities adding one, and then allocates
 * new memory for all the variables whose size depends on the number
 * of communities. Note that the new community created by this function
 * will be empty.
 */
void create_new_community()
{
	int i, row, column;
	
	//Update current number of communities
	current_number_communities=current_number_communities+1;
	
	//Allocate memory for new community in edge_community_matrix
	for (i=0;i<N;i++){
		edge_community_matrix[i]=(long int *)realloc(edge_community_matrix[i],sizeof(long int)*current_number_communities);
	}
	
	//Initialise the new community in edge_community_matrix to zero
	for (row=0;row<N;row++){
		edge_community_matrix[row][current_number_communities-1]=0;
	}

	//Allocate memory for new community in community size vector
	community_size=realloc(community_size,sizeof(int)*current_number_communities);
	//Initialise the new community in community size vector to zero
	community_size[current_number_communities-1]=0;
	
	//Allocate memory for new community in flag vector community_blocked
	community_blocked=realloc(community_blocked,sizeof(int)*current_number_communities);
	//Initialise the new flag entry to zero
	community_blocked[current_number_communities-1]=0;
	//Allocate memory for new community in community_density matrix
	community_density=(long int **)realloc(community_density,sizeof(long int *)*current_number_communities);
	//initialise the pointer to the new row to NULL to avoid problems
	//of segmentation fault
	community_density[current_number_communities-1]=NULL;
	for (i=0;i<current_number_communities;i++){
		community_density[i]=(long int *)realloc(community_density[i],sizeof(long int)*current_number_communities);
	}
	
	//Initialise the new community in community_density matrix to zero
	for (row=0;row<current_number_communities;row++){
		community_density[row][current_number_communities-1]=0;
	}
	for (column=0;column<current_number_communities;column++){
		community_density[current_number_communities-1][column]=0;
	}
	
	return;
}

/* This function computes the value of the modularity density.
 * It receives as input a pointer to the variable that will store the
 * value of the modularity density produced as output.
 */
void compute_modularity_density(double *q_ds)
{
	//initialise the input variable to zero.
	*q_ds=0.0;
	
	int index1, index2;
	//compute the split penalty term in the expression for modularity
	//density. Compute also the external number of links of
	//each community.
	double split_penalty[current_number_communities];
	double external_links[current_number_communities];
	for (index1=0;index1<current_number_communities;index1++){
		//initialise arrays' entries to zero
		split_penalty[index1]=0;
		external_links[index1]=0;
		for (index2=0;index2<index1;index2++){
			split_penalty[index1]=split_penalty[index1]+((double)community_density[index1][index2]*(double)community_density[index1][index2])/(2.0*(double)number_edges*(double)community_size[index1]*(double)community_size[index2]);
			external_links[index1]=external_links[index1]+(double)community_density[index1][index2];
		}
		for (index2=index1+1;index2<current_number_communities;index2++){
			split_penalty[index1]=split_penalty[index1]+((double)community_density[index1][index2]*(double)community_density[index1][index2])/(2.0*(double)number_edges*(double)community_size[index1]*(double)community_size[index2]);
			external_links[index1]=external_links[index1]+(double)community_density[index1][index2];
		}
	}
	
	double temp1, temp2, temp3;
	//compute modularity density; compute the various terms and then
	//add them together.
	for (index1=0;index1<current_number_communities;index1++){
		//number of internal edges in the community under consideration
		//divided by overall number of edges in the network.
		temp1=((double)community_density[index1][index1])/((double)number_edges);
		//number of internal edges in the community under consideration
		//divided by community size
		temp2=(2.0*(double)community_density[index1][index1])/((double)community_size[index1]*((double)community_size[index1]-1.0));
		//term related to the null model
		temp3=(2.0*(double)community_density[index1][index1]+external_links[index1])/(2.0*(double)number_edges);
		
		//modularity density
		*q_ds=*q_ds+temp1*temp2-(temp3*temp2)*(temp3*temp2)-split_penalty[index1];
	}
	
	
	return;
}		

/* This function updates the edge community matrix when a node is moved
 * from one community to another. It needs to receive as input the
 * following three variables:
 * - the node label
 * - the old community to which the node was assigned in the previous
 *   step
 * - the new community to which the node is moving
 */
void update_edge_community_matrix(int node_to_move, int old_community_label, int new_community_label)
{
	//create a temporary variable storing the degree of the node which
	//changed community.
	int number_of_links;
	number_of_links=degree_sequence[node_to_move];
	
	int i;
	for (i=0;i<number_of_links;i++){		
		//update the edge_community matrix
		
		edge_community_matrix[adj_list[node_to_move][i]][old_community_label]=edge_community_matrix[adj_list[node_to_move][i]][old_community_label]-1;
		edge_community_matrix[adj_list[node_to_move][i]][new_community_label]=edge_community_matrix[adj_list[node_to_move][i]][new_community_label]+1;
	}
	
	return;
}
 
/* This function updates the community density matrix when a node is
 * moved from one community to another. It needs to receive as input
 * the following three variables:
 * - the node label
 * - the old community to which the node was assigned in the previous
 *   step
 * - the new community to which the node is moving
 */
void update_community_density_matrix(int node_to_move, int old_community_label, int new_community_label)
{
	//updates due to the node changing community
	community_density[old_community_label][old_community_label]=community_density[old_community_label][old_community_label]-edge_community_matrix[node_to_move][old_community_label];
	community_density[new_community_label][new_community_label]=community_density[new_community_label][new_community_label]+edge_community_matrix[node_to_move][new_community_label];
	community_density[old_community_label][new_community_label]=community_density[old_community_label][new_community_label]+edge_community_matrix[node_to_move][old_community_label]-edge_community_matrix[node_to_move][new_community_label];
	community_density[new_community_label][old_community_label]=community_density[old_community_label][new_community_label];
	
	int community_index;
	
	//updates due to the nodes connected to the nodes changing community
	
	//to split the for loops and avoid too many if statements, check
	//which community has the largest index and separate the two cases
	if (old_community_label<new_community_label){
		for (community_index=0;community_index<old_community_label;community_index++){
			//update the entry representing the number of links between
			//the old community and every community to which the node
			//moving is connected to
			community_density[old_community_label][community_index]=community_density[old_community_label][community_index]-edge_community_matrix[node_to_move][community_index];
			community_density[community_index][old_community_label]=community_density[old_community_label][community_index];
			
			//update the entry representing the number of links between
			//the new community and every community to which the node
			//moving is connected to
			community_density[new_community_label][community_index]=community_density[new_community_label][community_index]+edge_community_matrix[node_to_move][community_index];
			community_density[community_index][new_community_label]=community_density[new_community_label][community_index];
		}
		
		for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
			//update the entry representing the number of links between
			//the old community and every community to which the node
			//moving is connected to
			community_density[old_community_label][community_index]=community_density[old_community_label][community_index]-edge_community_matrix[node_to_move][community_index];
			community_density[community_index][old_community_label]=community_density[old_community_label][community_index];
			
			//update the entry representing the number of links between
			//the new community and every community to which the node
			//moving is connected to
			community_density[new_community_label][community_index]=community_density[new_community_label][community_index]+edge_community_matrix[node_to_move][community_index];
			community_density[community_index][new_community_label]=community_density[new_community_label][community_index];
		}
		for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
			//update the entry representing the number of links between
			//the old community and every community to which the node
			//moving is connected to
			community_density[old_community_label][community_index]=community_density[old_community_label][community_index]-edge_community_matrix[node_to_move][community_index];
			community_density[community_index][old_community_label]=community_density[old_community_label][community_index];
			
			//update the entry representing the number of links between
			//the new community and every community to which the node
			//moving is connected to
			community_density[new_community_label][community_index]=community_density[new_community_label][community_index]+edge_community_matrix[node_to_move][community_index];
			community_density[community_index][new_community_label]=community_density[new_community_label][community_index];
		}
	}
	else{
		for (community_index=0;community_index<new_community_label;community_index++){
			//update the entry representing the number of links between
			//the old community and every community to which the node
			//moving is connected to
			community_density[old_community_label][community_index]=community_density[old_community_label][community_index]-edge_community_matrix[node_to_move][community_index];
			community_density[community_index][old_community_label]=community_density[old_community_label][community_index];
			
			//update the entry representing the number of links between
			//the new community and every community to which the node
			//moving is connected to
			community_density[new_community_label][community_index]=community_density[new_community_label][community_index]+edge_community_matrix[node_to_move][community_index];
			community_density[community_index][new_community_label]=community_density[new_community_label][community_index];
		}
		
		for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
			//update the entry representing the number of links between
			//the old community and every community to which the node
			//moving is connected to
			community_density[old_community_label][community_index]=community_density[old_community_label][community_index]-edge_community_matrix[node_to_move][community_index];
			community_density[community_index][old_community_label]=community_density[old_community_label][community_index];
			
			//update the entry representing the number of links between
			//the new community and every community to which the node
			//moving is connected to
			community_density[new_community_label][community_index]=community_density[new_community_label][community_index]+edge_community_matrix[node_to_move][community_index];
			community_density[community_index][new_community_label]=community_density[new_community_label][community_index];
		}
		for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
			//update the entry representing the number of links between
			//the old community and every community to which the node
			//moving is connected to
			community_density[old_community_label][community_index]=community_density[old_community_label][community_index]-edge_community_matrix[node_to_move][community_index];
			community_density[community_index][old_community_label]=community_density[old_community_label][community_index];
			
			//update the entry representing the number of links between
			//the new community and every community to which the node
			//moving is connected to
			community_density[new_community_label][community_index]=community_density[new_community_label][community_index]+edge_community_matrix[node_to_move][community_index];
			community_density[community_index][new_community_label]=community_density[new_community_label][community_index];
		}
	}
	

	
	return;
}
 
 
/* This function updates the community size vector. When a node moves
 * from one community to another, the old community decreases by one
 * node in size and the new one increases by one. The function needs
 * the following two variables:
 * - the old community that has lost one node
 * - the new community that has gained one node
 */
void update_community_size_vector(int old_community_label, int new_community_label)
{
	//decrease by one the size of the community the node has left
	community_size[old_community_label]=community_size[old_community_label]-1;
	
	//increase by one the size of the community the node moved to
	community_size[new_community_label]=community_size[new_community_label]+1;

	return;
}


/* This function computes the change in modularity that would result if
 * the input node changes community. It receives as input the following
 * variables:
 * - the node that we are trying to move
 * - the community to which the node belongs to (old_community_label)
 * - the community to which we try to move the node (new_community_label)
 * - a pointer to a variable that stores the change in modularity
 *   density that would result from moving the node
 * - a matrix which is a temporary copy of the community density matrix
 * - a matrix which is a temporary copy of the edge community matrix
 * - a vector which is a temporary copy of the community size vector
 * - a vector which contains the contributions to the first part of
 *   modularity density
 * - a vector which contains all the split penalty terms
 * This function computes the value of modularity density that we would
 * obtain if the node received as input was moved between the two
 * communities passed to the function.
 * This is then compared to the current global value of the modularity
 * density. The difference between these two values is then assigned
 * to the pointer variable received as input.
 */
void compute_change_modularity_density_tuning_steps_1(int node_to_move, int old_community_label, int new_community_label, double *local_dq, long int **temporary_community_density,long int **temporary_edge_community_matrix, int *temporary_community_size,double *contributions_to_modularity_density,double *split_penalty)
{
	//variable that stores the modularity density value if the node
	//node_to_move changes community
	double temp_modularity_density=0.0;
	
	int community_index;
	double single_community_contribution=0.0;
	double old_contribution1, old_contribution2, new_contribution1, new_contribution2;
	double temp_contribution1, temp_contribution2, temp_contribution3;
	//contribution to modularity density given by all the communities
	//which are neither the old nor the new one

	for (community_index=0;community_index<new_community_label;community_index++){
		old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
		old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
		temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
		temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
		single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2)/((double)2.0*(double)number_edges);
		temp_modularity_density=temp_modularity_density+single_community_contribution;
	}
	for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
		old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
		old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
		temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
		temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
		single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2)/((double)2.0*(double)number_edges);
		temp_modularity_density=temp_modularity_density+single_community_contribution;
	}
	for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
		old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
		old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
		temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
		temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
		single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2)/((double)2.0*(double)number_edges);
		temp_modularity_density=temp_modularity_density+single_community_contribution;
	}


	//now compute the contribution to modularity density given by the
	//old community
	//first compute the temporary split penalty term for the old community
	double temp_split_penalty_contribution=0.0;
	double external_links=0.0;

	for (community_index=0;community_index<new_community_label;community_index++){
		temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];	
	}
	for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
		temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];	
	}
	for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
		temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];	
	}
	temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]);
	new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
	temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
	external_links=external_links+(double)temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label];

	
	//then compute the first term in the modularity density formula
	//for the old community
	temp_contribution1=((double)temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[node_to_move][old_community_label])/((double)number_edges);
	temp_contribution2=(2.0*((double)temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[node_to_move][old_community_label]))/(((double)temporary_community_size[old_community_label]-1.0)*((double)temporary_community_size[old_community_label]-1.0-1.0));
	temp_contribution3=(2.0*((double)temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[node_to_move][old_community_label])+external_links)/(2.0*(double)number_edges);
	new_contribution2=temp_contribution1*temp_contribution2-(temp_contribution3*temp_contribution2)*(temp_contribution3*temp_contribution2);
	//and then add them to the temporary value of the modularity density
	temp_modularity_density=temp_modularity_density+new_contribution2-temp_split_penalty_contribution/((double)2.0*(double)number_edges);
		
	//now compute the contribution to modularity density given by the
	//new community
	//first compute the temporary split penalty term for the new community
	temp_split_penalty_contribution=0.0;
	external_links=0.0;


	for (community_index=0;community_index<new_community_label;community_index++){
		temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
	}
	for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
		temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
	}
	for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
		temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
	}
	temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]);
	new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
	temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
	external_links=external_links+(double)temporary_community_density[new_community_label][old_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label];

	
	//then compute the first term in the modularity density formula
	//for the new community
	temp_contribution1=((double)temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][new_community_label])/((double)number_edges);
	temp_contribution2=(2.0*((double)temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][new_community_label]))/(((double)temporary_community_size[new_community_label]+1.0)*((double)temporary_community_size[new_community_label]+1.0-1.0));
	temp_contribution3=(2.0*((double)temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][new_community_label])+external_links)/(2.0*(double)number_edges);
	new_contribution2=temp_contribution1*temp_contribution2-(temp_contribution3*temp_contribution2)*(temp_contribution3*temp_contribution2);
	//and then add them to the temporary value of the modularity density
	temp_modularity_density=temp_modularity_density+new_contribution2-temp_split_penalty_contribution/((double)2.0*(double)number_edges);
	//difference between the modularity density value when we move
	//node_to_move and the current modularity density value
	double temp_local=temp_modularity_density-results.modularity_density;
	*local_dq=temp_modularity_density;
	return;
}

void compute_change_modularity_density_tuning_steps_2(int node_to_move, int old_community_label, int new_community_label, double *local_dq, long int **temporary_community_density,long int **temporary_edge_community_matrix, int *temporary_community_size,double *contributions_to_modularity_density,double *split_penalty)
{
	//variable that stores the modularity density value if the node
	//node_to_move changes community
	double temp_modularity_density=0.0;
	
	int community_index;
	double single_community_contribution=0.0;
	double old_contribution1, old_contribution2, new_contribution1, new_contribution2;
	double temp_contribution1, temp_contribution2, temp_contribution3;
	//contribution to modularity density given by all the communities
	//which are neither the old nor the new one

	for (community_index=0;community_index<old_community_label;community_index++){
		old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
		old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
		temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
		temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
		single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2)/((double)2.0*(double)number_edges);
		temp_modularity_density=temp_modularity_density+single_community_contribution;
	}
	for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
		old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
		old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
		temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
		temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
		single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2)/((double)2.0*(double)number_edges);
		temp_modularity_density=temp_modularity_density+single_community_contribution;
	}
	for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
		old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
		old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
		temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
		temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
		single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2)/((double)2.0*(double)number_edges);
		temp_modularity_density=temp_modularity_density+single_community_contribution;
	}
	


	//now compute the contribution to modularity density given by the
	//old community
	//first compute the temporary split penalty term for the old community
	double temp_split_penalty_contribution=0.0;
	double external_links=0.0;

	for (community_index=0;community_index<old_community_label;community_index++){
		temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];	
	}
	for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
		temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];	
	}
	for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
		temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];	
	}
	temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]);
	new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
	temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
	external_links=external_links+(double)temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label];

	
	
	//then compute the first term in the modularity density formula
	//for the old community
	temp_contribution1=((double)temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[node_to_move][old_community_label])/((double)number_edges);
	temp_contribution2=(2.0*((double)temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[node_to_move][old_community_label]))/(((double)temporary_community_size[old_community_label]-1.0)*((double)temporary_community_size[old_community_label]-1.0-1.0));
	temp_contribution3=(2.0*((double)temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[node_to_move][old_community_label])+external_links)/(2.0*(double)number_edges);
	new_contribution2=temp_contribution1*temp_contribution2-(temp_contribution3*temp_contribution2)*(temp_contribution3*temp_contribution2);
	//and then add them to the temporary value of the modularity density
	temp_modularity_density=temp_modularity_density+new_contribution2-temp_split_penalty_contribution/((double)2.0*(double)number_edges);
		
	//now compute the contribution to modularity density given by the
	//new community
	//first compute the temporary split penalty term for the new community
	temp_split_penalty_contribution=0.0;
	external_links=0.0;

	for (community_index=0;community_index<old_community_label;community_index++){
		temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
	}
	for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
		temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
	}
	for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
		temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
		new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
		temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
		external_links=external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
	}
	temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]);
	new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
	temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
	external_links=external_links+(double)temporary_community_density[new_community_label][old_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label];

	
	
	//then compute the first term in the modularity density formula
	//for the new community
	temp_contribution1=((double)temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][new_community_label])/((double)number_edges);
	temp_contribution2=(2.0*((double)temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][new_community_label]))/(((double)temporary_community_size[new_community_label]+1.0)*((double)temporary_community_size[new_community_label]+1.0-1.0));
	temp_contribution3=(2.0*((double)temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][new_community_label])+external_links)/(2.0*(double)number_edges);
	new_contribution2=temp_contribution1*temp_contribution2-(temp_contribution3*temp_contribution2)*(temp_contribution3*temp_contribution2);
	//and then add them to the temporary value of the modularity density
	temp_modularity_density=temp_modularity_density+new_contribution2-temp_split_penalty_contribution/((double)2.0*(double)number_edges);
	//difference between the modularity density value when we move
	//node_to_move and the current modularity density value
	double temp_local=temp_modularity_density-results.modularity_density;
	*local_dq=temp_modularity_density;
	return;
}

/* This function bisects the input vector according to the sign of each
 * element. The inputs are the leading eigenvector found with the power
 * method, an array with the indices of the nodes of the community under
 * consideration and the size of this community.
 * Every time a node is assigned to a different community, the global
 * variables are updated accordingly.
 */
void bisection(double *leading_eigenvector, int *current_nodes, int current_community_size)
{
	int i, current_community_label;
	//formally create a new community preparing the memory. Note also
	//that this function already updates the current number of
	//communities.
	create_new_community();

	for (i=0;i<current_community_size;i++){
		if (leading_eigenvector[i]<0){
			//label of the community that the node is leaving
			current_community_label=results.community_list[current_nodes[i]];
			//assign the node to its new community
			results.community_list[current_nodes[i]]=current_number_communities;
			//update edge community matrix
			update_edge_community_matrix(current_nodes[i],current_community_label-1, current_number_communities-1);
			//update community density matrix
			update_community_density_matrix(current_nodes[i],current_community_label-1,current_number_communities-1);
			//update community size vector
			update_community_size_vector(current_community_label-1, current_number_communities-1);
		}	
	}
	
	return;
}

	
/* This function implements the fine tuning step after the bisection.
 * It receives as inputs:
 * - the labels of the nodes undergoing the fine tuning step
 * - the number of nodes undergoing the fine tuning step
 * - the labels of the two communities involved in the fine tuning step
 * The function creates temporary local variables that mimic the global
 * ones, so that we can perform local updates to see how the modularity
 * density value would change when moving nodes from one community to
 * the other. The layout of the function is the following:
 * - create the local variables
 * - try to move every node from its community to the other one under
 *   consideration
 * - store the change in modularity density that would result from that
 *   potential move
 * - find all the nodes whose moves result in the highest (within a
 *   global tolerance parameter) increase in modularity density
 * - select randomly amongst these nodes and perform the move only
 *   locally
 * - repeat the step as many times as the number of input nodes
 * - at each step, store the change in modularity density that would
 *   result from the chosen move in each iteration
 * - find all the steps that result in the largest (within a global
 *   tolerance parameter) increase in modularity density
 * - if this increase is larger than zero (within a global tolerance
 *   parameter), then pick randomly amongst these steps
 * - perform all the steps up to the one selected and update all the
 *   global variables
 * - repeat all the steps until modularity density increases
 */
void fine_tuning(int *current_nodes, int number_nodes, int community_label_1, int community_label_2)
{
	int i,j,memory_index;
	//for (i=0;i<number_nodes;i++)printf("Nodes considered in fine tuning %i\n",current_nodes[i]);
	//initialise the global variable that stores the increase in
	//modularity density obtained in the fine tuning step
	dq_fine=0.0;
	
	//variable storing the old value of modularity density initialised
	//to the global value of modularity density obtained so far
	double old_modularity_density;
	
	//array storing the sequence of nodes to move that would result
	//in the highest increase in modularity density
	int *sequence_of_nodes_to_move;
	sequence_of_nodes_to_move=malloc(number_nodes*sizeof(int));
	//array storing the increase in modularity density at each step
	double *dq_fine_tuning_tree;
	dq_fine_tuning_tree=malloc(number_nodes*sizeof(double));
	//array storing the potential increase in modularity density
	//resulting from temporary moves of the various nodes
	double *dq_steps;
	dq_steps=malloc(number_nodes*sizeof(double));
	//array containing the indices of nodes corresponding to the
	//maximum modularity density increase (at any given step)
	int *node_maximum_modularity;
	node_maximum_modularity=malloc(number_nodes*sizeof(int));
	//array containing a flag to indicate whether a node has been
	//"blocked" (i.e. already moved) at a previous step
	int *flag_node_blocked;
	flag_node_blocked=malloc(number_nodes*sizeof(int));
	
	//declaration of local copies of global variables
	long int **temporary_community_density;
	temporary_community_density=malloc(current_number_communities*sizeof(long int *));
	for (memory_index=0;memory_index<current_number_communities;memory_index++) temporary_community_density[memory_index]=malloc(current_number_communities*sizeof(long int));
	int *temporary_community_size;
	temporary_community_size=malloc(current_number_communities*sizeof(int));
	long int **temporary_edge_community_matrix;
	temporary_edge_community_matrix=malloc(N*sizeof(long int *));
	for (memory_index=0;memory_index<N;memory_index++) temporary_edge_community_matrix[memory_index]=malloc(current_number_communities*sizeof(long int));
	int *temporary_community_list;
	temporary_community_list=malloc(number_nodes*sizeof(int));
	
	//create split penalty vector, first contribution to
	//modularity density and number of external links arrays
	double *first_contribution_to_modularity_density;
	first_contribution_to_modularity_density=malloc(current_number_communities*sizeof(double));
	double *split_penalty;
	split_penalty=malloc(current_number_communities*sizeof(double));
	double *external_links;
	external_links=malloc(current_number_communities*sizeof(double));
	
	//repeat the fine tuning step as long as it results in an increase
	//in the value of modularity density. We use a flag variable to
	//indicate whether the step has achieved an increase (within
	//tolerance) or not. If there is an increase, we repeat.
	int flag_increase=0;

	do{
		//set the flag to 0 at the start of every repetition
		flag_increase=0;
		
		//initialise the variable storing the old value of the
		//modularity density
		old_modularity_density=results.modularity_density;
		
		//variable counting how many steps have been performed
		int number_of_steps=0;
		
		//Initialise the nodes as all being unblocked
		int i;
		for (i=0;i<number_nodes;i++){ flag_node_blocked[i]=0;}
		
		//temporary variable to store the modularity density change that
		//would result from the move of a node
		double temp_dq;
		temp_dq=0.0;
		
		//local variable that will represent a temporary copy of the
		//community density matrix.
		
		int index1,index2;
		//create the local copy of the community density matrix
		for (index1=0;index1<current_number_communities;index1++){
			for (index2=0;index2<current_number_communities;index2++){
				temporary_community_density[index1][index2]=community_density[index1][index2];
			}
		}
		
		//local variable that will represent a temporary copy of the
		//community size vector
		
		//create the local copy of the community size vector
		for (index1=0;index1<current_number_communities;index1++){
			temporary_community_size[index1]=community_size[index1];
		}
		
		//create split penalty vector, first contribution to
		//modularity density and number of external links arrays
		for (index1=0;index1<current_number_communities;index1++){
			split_penalty[index1]=0;
			external_links[index1]=0;
			//split the for loop to avoid the indices being equal and
			//to avoid using an if statement
			for (index2=0;index2<index1;index2++){
				split_penalty[index1]=split_penalty[index1]+((double)community_density[index1][index2]*(double)community_density[index1][index2])/((double)community_size[index1]*(double)community_size[index2]);
				external_links[index1]=external_links[index1]+(double)community_density[index1][index2];
			}
			for (index2=index1+1;index2<current_number_communities;index2++){
				split_penalty[index1]=split_penalty[index1]+((double)community_density[index1][index2]*(double)community_density[index1][index2])/((double)community_size[index1]*(double)community_size[index2]);
				external_links[index1]=external_links[index1]+(double)community_density[index1][index2];
			}
		}
		
		double temp1,temp2,temp3;
		for (index1=0;index1<current_number_communities;index1++){
			temp1=((double)community_density[index1][index1])/((double)number_edges);
			temp2=(2.0*(double)community_density[index1][index1])/((double)community_size[index1]*((double)community_size[index1]-1.0));
			temp3=(2.0*(double)community_density[index1][index1]+external_links[index1])/(2.0*(double)number_edges);
			first_contribution_to_modularity_density[index1]=temp1*temp2-(temp3*temp2)*(temp3*temp2);
		}
		
		
		//local variable that will represent a temporary copy of the edge
		//community matrix.
		
		//create the local copy of the edge community matrix
		for (index1=0;index1<N;index1++){
			for (index2=0;index2<current_number_communities;index2++){
				temporary_edge_community_matrix[index1][index2]=edge_community_matrix[index1][index2];
			}
		}
		
		//local copy of the community assignments.
		
		for (index1=0;index1<number_nodes;index1++){
			temporary_community_list[index1]=results.community_list[current_nodes[index1]];
		}
		
		//try and move each node, compute the potential change in modularity
		//density and then move the one which would result in the highest
		//change. If more than one node has the highest change (within a
		//tolerance limit set as global parameter), then pick randomly.
		//Repeat this procedure number_nodes times.
		for (index1=0;index1<number_nodes;index1++){
			
			//initialise the dq at every step to a large negative number
			for (index2=0;index2<number_nodes;index2++)dq_steps[index2]=-1000.0;
			//try to move every node and store the change in modularity
			//density that results from every potential move
			#pragma omp parallel private(temp_dq)
			{
				#pragma omp for //private(temporary_community_list, temporary_edge_community_matrix, temporary_community_size, first_contribution_to_modularity_density, split_penalty)
				for (index2=0;index2<number_nodes;index2++){
					if (flag_node_blocked[index2]==0){
						if (temporary_community_list[index2]==community_label_1 && temporary_community_size[community_label_1-1]>2 && temporary_edge_community_matrix[current_nodes[index2]][community_label_2-1]>0){
							compute_change_modularity_density_tuning_steps_2(current_nodes[index2], community_label_1-1, community_label_2-1, &temp_dq, temporary_community_density, temporary_edge_community_matrix, temporary_community_size, first_contribution_to_modularity_density, split_penalty);
							dq_steps[index2]=temp_dq-old_modularity_density;
						}
						else if (temporary_community_list[index2]==community_label_2 && temporary_community_size[community_label_2-1]>2 && temporary_edge_community_matrix[current_nodes[index2]][community_label_1-1]>0){
							compute_change_modularity_density_tuning_steps_1(current_nodes[index2], community_label_2-1, community_label_1-1, &temp_dq, temporary_community_density, temporary_edge_community_matrix, temporary_community_size, first_contribution_to_modularity_density, split_penalty);
							dq_steps[index2]=temp_dq-old_modularity_density;
						}
					}
				}
			}
			
			//initialise the maximum increase in modularity to a very
			//large negative number
			double max_dq=-1000.0;
			//find the maximum modularity density change
			for (index2=0;index2<number_nodes;index2++){
				if (dq_steps[index2]>max_dq && flag_node_blocked[index2]==0){
					max_dq=dq_steps[index2];
				}
			}
			
			if (max_dq==-1000.0){
				number_of_steps=index1;
				break;
			}
			else{
				number_of_steps=index1;
			}

			
			//variable representing the multeplicity of nodes achieving
			//the highest change in modularity density (within tolerance).
			int multeplicity=0;
			//find all the nodes whose moves result in the maximum change
			//(within the tolerance parameter)
			for (index2=0;index2<number_nodes;index2++){
				if (max_dq-dq_steps[index2]<toler && flag_node_blocked[index2]==0){
					node_maximum_modularity[multeplicity]=index2;
					multeplicity=multeplicity+1;
				}
			}
			
			//choose the node to move randomly between the ones achieving
			//the highest possible change in modularity
			int node_to_move;
			node_to_move=node_maximum_modularity[((int)(rand()%multeplicity))];
			//store the node to move in the sequence
			sequence_of_nodes_to_move[index1]=current_nodes[node_to_move];
			//block the node that has been moved
			flag_node_blocked[node_to_move]=1;
			//find the increase in modularity density obtained so far
			if (index1==0){
				dq_fine_tuning_tree[0]=dq_steps[node_to_move];
			}
			else {
				dq_fine_tuning_tree[index1]=dq_fine_tuning_tree[index1-1]+dq_steps[node_to_move];
			}
			
			//to make the following updates easier, identify and label
			//properly which one is the old and the new community.
			
			int old_community_label, new_community_label;
			if (temporary_community_list[node_to_move]==community_label_1){
				old_community_label=community_label_1-1;
				new_community_label=community_label_2-1;
			}
			else{
				old_community_label=community_label_2-1;
				new_community_label=community_label_1-1;
			}
			//update the temporary community list
			temporary_community_list[node_to_move]=new_community_label+1;
			//the next section performs updates to the split penalty
			//vector due to the node that changed community and updates
			//also the vector storing the external number of links of
			//each community
			double single_community_contribution=0.0;
			double old_contribution1, old_contribution2, new_contribution1, new_contribution2;
			double temp_contribution1, temp_contribution2, temp_contribution3;
			double temp_split_penalty_contribution=0.0;
			double temp_external_links=0.0;
			//update the entries corresponding to the communities not
			//involved in the move. Note that for these communities
			//the overall external number of links has not changed.
			int community_index;
			//split the for loop to avoid the indices being equal and
			//to avoid using an if statement
			if (old_community_label<new_community_label){
				for (community_index=0;community_index<old_community_label;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				
				//update the entry corresponding to the old community
				temp_split_penalty_contribution=0.0;
				temp_external_links=0.0;
				for (community_index=0;community_index<old_community_label;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				
				temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label]);
				new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
				temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label];	
				
				split_penalty[old_community_label]=temp_split_penalty_contribution;
				external_links[old_community_label]=temp_external_links;
				
				//update the entry corresponding to the new community
				temp_split_penalty_contribution=0.0;
				temp_external_links=0.0;
				for (community_index=0;community_index<old_community_label;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}					

				temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label]);
				new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
				temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][old_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label];

				split_penalty[new_community_label]=temp_split_penalty_contribution;
				external_links[new_community_label]=temp_external_links;
				
				//The next section of code performs an update to the temporary
				//community density moving the node that was identified as
				//giving the highest increase in modularity density.
				
				//updates due to the node changing community
				temporary_community_density[old_community_label][old_community_label]=temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label];
				temporary_community_density[new_community_label][new_community_label]=temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label];
				temporary_community_density[old_community_label][new_community_label]=temporary_community_density[old_community_label][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label];
				temporary_community_density[new_community_label][old_community_label]=temporary_community_density[old_community_label][new_community_label];
				
				//updates due to the nodes connected to the nodes changing communities
	
				for (community_index=0;community_index<old_community_label;community_index++){
					
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
				for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
					
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
				for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
					
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
			}
			else{
				for (community_index=0;community_index<new_community_label;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				
				//update the entry corresponding to the old community
				temp_split_penalty_contribution=0.0;
				temp_external_links=0.0;
				for (community_index=0;community_index<new_community_label;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				
				temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label]);
				new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
				temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label];	
				
				split_penalty[old_community_label]=temp_split_penalty_contribution;
				external_links[old_community_label]=temp_external_links;
				
				//update the entry corresponding to the new community
				temp_split_penalty_contribution=0.0;
				temp_external_links=0.0;
				for (community_index=0;community_index<new_community_label;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}
				for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
				}					

				temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label]);
				new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
				temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][old_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label];

				split_penalty[new_community_label]=temp_split_penalty_contribution;
				external_links[new_community_label]=temp_external_links;
				
				//The next section of code performs an update to the temporary
				//community density moving the node that was identified as
				//giving the highest increase in modularity density.
				
				//updates due to the node changing community
				temporary_community_density[old_community_label][old_community_label]=temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label];
				temporary_community_density[new_community_label][new_community_label]=temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label];
				temporary_community_density[old_community_label][new_community_label]=temporary_community_density[old_community_label][new_community_label]+temporary_edge_community_matrix[current_nodes[node_to_move]][old_community_label]-temporary_edge_community_matrix[current_nodes[node_to_move]][new_community_label];
				temporary_community_density[new_community_label][old_community_label]=temporary_community_density[old_community_label][new_community_label];
				
				//updates due to the nodes connected to the nodes changing communities
	
				for (community_index=0;community_index<new_community_label;community_index++){
					
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
				for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
					
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
				for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
					
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[current_nodes[node_to_move]][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
			}
			
			//the following section performs a local update to the temporary
			//edge community matrix. The update concerns the node that has
			//been decided to move in the previous step.
			
			//create a temporary variable storing the degree of the node which
			//changed community.
			int number_of_links;
			number_of_links=degree_sequence[current_nodes[node_to_move]];
			
			int i;
			for (i=0;i<number_of_links;i++){		
				//update the edge_community matrix
				temporary_edge_community_matrix[adj_list[current_nodes[node_to_move]][i]][old_community_label]=temporary_edge_community_matrix[adj_list[current_nodes[node_to_move]][i]][old_community_label]-1;
				temporary_edge_community_matrix[adj_list[current_nodes[node_to_move]][i]][new_community_label]=temporary_edge_community_matrix[adj_list[current_nodes[node_to_move]][i]][new_community_label]+1;
			}
			
			//the following section performs an update to the temporary
			//community size vector
			temporary_community_size[old_community_label]=temporary_community_size[old_community_label]-1;
			temporary_community_size[new_community_label]=temporary_community_size[new_community_label]+1;
			

			//the following section updates the vector storing the first
			//part of the modularity density. Note that only the two
			//entries relative to the old and new community get updated.
			
			//update of the entry relative to the old community
			temp1=((double)temporary_community_density[old_community_label][old_community_label])/((double)number_edges);
			temp2=(2.0*(double)temporary_community_density[old_community_label][old_community_label])/((double)temporary_community_size[old_community_label]*((double)temporary_community_size[old_community_label]-1.0));
			temp3=(2.0*(double)temporary_community_density[old_community_label][old_community_label]+external_links[old_community_label])/(2.0*(double)number_edges);
			first_contribution_to_modularity_density[old_community_label]=temp1*temp2-(temp3*temp2)*(temp3*temp2);
			
			//update of the entry relative to the new community
			temp1=((double)temporary_community_density[new_community_label][new_community_label])/((double)number_edges);
			temp2=(2.0*(double)temporary_community_density[new_community_label][new_community_label])/((double)temporary_community_size[new_community_label]*((double)temporary_community_size[new_community_label]-1.0));
			temp3=(2.0*(double)temporary_community_density[new_community_label][new_community_label]+external_links[new_community_label])/(2.0*(double)number_edges);
			first_contribution_to_modularity_density[new_community_label]=temp1*temp2-(temp3*temp2)*(temp3*temp2);

			//update the value of modularity density to compare
			old_modularity_density=0.0;
			for (index2=0;index2<current_number_communities;index2++){
				old_modularity_density=old_modularity_density+first_contribution_to_modularity_density[index2]-split_penalty[index2]/((double)2.0*(double)number_edges);
			}		
		}
		
		//find the step in the tree which achieves the highest increase
		//in the value of modularity density. Initialise the
		//maximum_increase to a large negative number
		double maximum_increase=-1000.0;
		for (index2=0;index2<number_of_steps;index2++){
			if (dq_fine_tuning_tree[index2]>maximum_increase){
				maximum_increase=dq_fine_tuning_tree[index2];
			}
		}
		//if the increase in modularity density is larger than zero (within
		//tolerance), then find if there are other steps which achieve
		//the same (within tolerance) increase and pick one randomly.
		if (maximum_increase>toler){
			
			int multiple_increase=0;
			for (index2=0;index2<number_of_steps;index2++){
				if (maximum_increase-dq_fine_tuning_tree[index2]<toler){
					node_maximum_modularity[multiple_increase]=index2;
					multiple_increase=multiple_increase+1;
				}
			}
			//pick randomly the step
			int step_in_tree;
			step_in_tree=node_maximum_modularity[((int)(rand()%multiple_increase))];
			//update the accepted increase in modularity density
			dq_fine=dq_fine+dq_fine_tuning_tree[step_in_tree];
			//then perform all the necessary updates and switches in the
			//community assignment as stored in the decision tree
			index2=0;
			while(index2<=step_in_tree){
				if (results.community_list[sequence_of_nodes_to_move[index2]]==community_label_1){
					results.community_list[sequence_of_nodes_to_move[index2]]=community_label_2;
					update_edge_community_matrix(sequence_of_nodes_to_move[index2], community_label_1-1,community_label_2-1);
					update_community_density_matrix(sequence_of_nodes_to_move[index2],community_label_1-1,community_label_2-1);
					update_community_size_vector(community_label_1-1,community_label_2-1);
					
				}
				else {
					results.community_list[sequence_of_nodes_to_move[index2]]=community_label_1;
					update_edge_community_matrix(sequence_of_nodes_to_move[index2],community_label_2-1,community_label_1-1);
					update_community_density_matrix(sequence_of_nodes_to_move[index2],community_label_2-1,community_label_1-1);
					update_community_size_vector(community_label_2-1,community_label_1-1);
				}
				index2=index2+1;
			}
			
			//update the current value of modularity density
			compute_modularity_density(&results.modularity_density);
			
			//flag that the fine tuning step resulted in an increase in
			//the value of modularity density.
			flag_increase=1;
		}
	} while (flag_increase);
	
	//free the memory of local variables
	free(sequence_of_nodes_to_move);
	free(dq_fine_tuning_tree);
	free(dq_steps);
	free(node_maximum_modularity);
	free(flag_node_blocked);
	for (memory_index=0;memory_index<current_number_communities;memory_index++) free(temporary_community_density[memory_index]);
	free(temporary_community_density);
	free(temporary_community_size);
	for (memory_index=0;memory_index<N;memory_index++) free(temporary_edge_community_matrix[memory_index]);
	free(temporary_community_list);
	free(temporary_edge_community_matrix);
	free(first_contribution_to_modularity_density);
	free(split_penalty);
	free(external_links);
	
	return;
}
 
 
 
/* This function implements the final tuning step of the algorithm
 * 
 */
void final_tuning()
{
	//initialise the global variable that stores the increase in
	//modularity density obtained in the final tuning step
	dq_final=0.0;
	
	//variable storing the old value of modularity density initialised
	//to the global value of modularity density obtained so far
	double old_modularity_density;
	
	int memory_index;
	//array storing the sequence of nodes to move that would result
	//in the highest increase in modularity density and the communities
	//to which the nodes need to move to
	int **sequence_of_nodes_to_move;
	sequence_of_nodes_to_move=malloc(N*sizeof(int *));
	for (memory_index=0;memory_index<N;memory_index++) sequence_of_nodes_to_move[memory_index]=malloc(2*sizeof(int));
	//array storing the increase in modularity density at each step
	double *dq_final_tuning_tree;
	dq_final_tuning_tree=(double *)malloc(N*sizeof(double));
	//matrix storing the potential increase in modularity density
	//resulting from temporary moves of the various nodes;the first
	//dimension (i.e. the rows) corresponds to the node which we
	//consider moving, the second dimension (i.e. the columns) refers
	//to the potential destination (i.e. the community to which we might
	//move the node)
	double **dq_steps;
	dq_steps=(double **)malloc(N*sizeof(double *));
	for (memory_index=0;memory_index<N;memory_index++){
		dq_steps[memory_index]=(double *)malloc(current_number_communities*sizeof(double ));
	}
	//array containing the indices of nodes corresponding to the
	//maximum modularity density increase (at any given step) together
	//with the potential community the node will move to
	int **node_maximum_modularity;
	node_maximum_modularity=malloc(N*current_number_communities*sizeof(int *));
	for (memory_index=0;memory_index<N*current_number_communities;memory_index++) node_maximum_modularity[memory_index]=malloc(2*sizeof(int));
	//array containing a flag to indicate whether a node has been
	//"blocked" (i.e. already moved) at a previous step
	int *flag_node_blocked;
	flag_node_blocked=malloc(N*sizeof(int));
	
	//repeat the final tuning step as long as it results in an increase
	//in the value of modularity density. We use a flag variable to
	//indicate whether the step has achieved an increase (within
	//tolerance) or not. If there is an increase, we repeat.
	int flag_increase=0;
	
	//local variable that will represent a temporary copy of the
	//community density matrix.
	long int **temporary_community_density;
	temporary_community_density=malloc(current_number_communities*sizeof(long int *));
	for (memory_index=0;memory_index<current_number_communities;memory_index++) temporary_community_density[memory_index]=malloc(current_number_communities*sizeof(long int));
	
	//local variable that will represent a temporary copy of the
	//community size vector
	int *temporary_community_size;
	temporary_community_size=malloc(current_number_communities*sizeof(int));
	
	//local variable that will represent a temporary copy of the edge
	//community matrix.
	long int **temporary_edge_community_matrix;
	temporary_edge_community_matrix=malloc(N*sizeof(long int *));
	for (memory_index=0;memory_index<N;memory_index++) temporary_edge_community_matrix[memory_index]=malloc(current_number_communities*sizeof(long int));
	
	//local copy of the community assignments.
	int *temporary_community_list;
	temporary_community_list=malloc(N*sizeof(int));
	
	//create split penalty vector, first contribution to
	//modularity density and number of external links arrays
	double *first_contribution_to_modularity_density;
	first_contribution_to_modularity_density=(double *)malloc(current_number_communities*sizeof(double));
	double *split_penalty;
	split_penalty=(double *)malloc(current_number_communities*sizeof(double));
	double *external_links;
	external_links=(double *)malloc(current_number_communities*sizeof(double));
	
	do{
		//set the flag to 0 at the start of every repetition
		flag_increase=0;
		
		//initialise the variable storing the old value of the
		//modularity density
		old_modularity_density=results.modularity_density;
		
		//variable counting how many steps have been performed
		int number_of_steps=0;
		
		//Initialise the nodes as all being unblocked
		int i;
		for (i=0;i<N;i++){ flag_node_blocked[i]=0;}
		
		//temporary variable to store the modularity density change that
		//would result from the move of a node
		double temp_dq;
		temp_dq=0.0;
		
		int index1,index2,index3,index4;
		//create the local copy of the community density matrix
		for (index1=0;index1<current_number_communities;index1++){
			for (index2=0;index2<current_number_communities;index2++){
				temporary_community_density[index1][index2]=community_density[index1][index2];
			}
		}
		
		//create the local copy of the community size vector
		for (index1=0;index1<current_number_communities;index1++){
			temporary_community_size[index1]=community_size[index1];
		}
		
		//create split penalty vector, first contribution to
		//modularity density and number of external links arrays
		for (index1=0;index1<current_number_communities;index1++){
			split_penalty[index1]=0;
			external_links[index1]=0;
			//split the for loop to avoid the indices being equal and
			//to avoid using an if statement
			for (index2=0;index2<index1;index2++){
				split_penalty[index1]=split_penalty[index1]+((double)community_density[index1][index2]*(double)community_density[index1][index2])/((double)community_size[index1]*(double)community_size[index2]);
				external_links[index1]=external_links[index1]+(double)community_density[index1][index2];
			}
			for (index2=index1+1;index2<current_number_communities;index2++){
				split_penalty[index1]=split_penalty[index1]+((double)community_density[index1][index2]*(double)community_density[index1][index2])/((double)community_size[index1]*(double)community_size[index2]);
				external_links[index1]=external_links[index1]+(double)community_density[index1][index2];
			}
		}
		double temp1,temp2,temp3;
		for (index1=0;index1<current_number_communities;index1++){
			temp1=((double)community_density[index1][index1])/((double)number_edges);
			temp2=(2.0*(double)community_density[index1][index1])/((double)community_size[index1]*((double)community_size[index1]-1.0));
			temp3=(2.0*(double)community_density[index1][index1]+external_links[index1])/(2.0*(double)number_edges);
			first_contribution_to_modularity_density[index1]=temp1*temp2-(temp3*temp2)*(temp3*temp2);
		}
		
		
		//create the local copy of the edge community matrix
		for (index1=0;index1<N;index1++){
			for (index2=0;index2<current_number_communities;index2++){
				temporary_edge_community_matrix[index1][index2]=edge_community_matrix[index1][index2];
			}
		}
		
		//create the local copy of the community assignments
		for (index1=0;index1<N;index1++){
			temporary_community_list[index1]=results.community_list[index1];
		}
		
		//try and move each node, compute the potential change in modularity
		//density and then move the one which would result in the highest
		//change. If more than one node has the highest change (within a
		//tolerance limit set as global parameter), then pick randomly.
		//Repeat this procedure number_nodes times.
		for (index1=0;index1<N;index1++){
			
			//try to move every node and store the change in modularity
			//density that results from every potential move
			#pragma omp parallel private(index4,index3,temp_dq)//shared(flag_node_blocked,temporary_community_size,temporary_community_density,temporary_community_list, temporary_edge_community_matrix,first_contribution_to_modularity_density,split_penalty) private(temp_dq,dq_steps,index3)
			{
				#pragma omp for
				for (index2=0;index2<N;index2++){
					
					if (flag_node_blocked[index2]==0){
	
						//every node will try to be moved to any other existing
						//community
						int test;
						test=temporary_community_list[index2]-1;
						dq_steps[index2][test]=-1000.0;
						//temporary_community_list[index2]-1!=index3
						for (index3=0;index3<test;index3++){
							//printf("index3 %i %i\n",test,index3);
							if (temporary_edge_community_matrix[index2][index3]>0){
								//index4 corresponds to the correct community label
								//index4=index3+1;
								//if the node is not in the community labelled by
								//index4 and the original community of the node
								//has a size larger than 2, try to move the node
								//to community index4
								
								if (temporary_community_size[temporary_community_list[index2]-1]>2){
									compute_change_modularity_density_tuning_steps_1(index2, temporary_community_list[index2]-1, index3, &temp_dq, temporary_community_density, temporary_edge_community_matrix, temporary_community_size, first_contribution_to_modularity_density, split_penalty);
									dq_steps[index2][index3]=temp_dq-old_modularity_density;
								}
								//else assign a large negative number to the
								//entry corresponding to the node moving to its
								//original community
								else{
									dq_steps[index2][index3]=-1000.0;
								}
							}
							else{
								dq_steps[index2][index3]=-1000.0;
							}
						}
						for (index3=test+1;index3<current_number_communities;index3++){
							//printf("index3 %i %i\n",test,index3);
							if (temporary_edge_community_matrix[index2][index3]>0){
								//index4 corresponds to the correct community label
								//index4=index3+1;
								//if the node is not in the community labelled by
								//index4 and the original community of the node
								//has a size larger than 2, try to move the node
								//to community index4
								
								if (temporary_community_size[temporary_community_list[index2]-1]>2){
									compute_change_modularity_density_tuning_steps_2(index2, temporary_community_list[index2]-1, index3, &temp_dq, temporary_community_density, temporary_edge_community_matrix, temporary_community_size, first_contribution_to_modularity_density, split_penalty);
									dq_steps[index2][index3]=temp_dq-old_modularity_density;
								}
								//else assign a large negative number to the
								//entry corresponding to the node moving to its
								//original community
								else{
									dq_steps[index2][index3]=-1000.0;
								}
							}
							else{
								dq_steps[index2][index3]=-1000.0;
							}
						}
					}
				}
			}
			//initialise the maximum increase in modularity to a very
			//large negative number
			double max_dq=-1000.0;
			//find the maximum modularity density change
			for (index2=0;index2<N;index2++){
				if (flag_node_blocked[index2]==0){
					for (index3=0;index3<current_number_communities;index3++){
						if (dq_steps[index2][index3]>max_dq){
							max_dq=dq_steps[index2][index3];
						}
					}
				}
			}
			if (max_dq==-1000.0){
				number_of_steps=index1;
				break;
			}
			else{
				number_of_steps=index1;
			}
			
			//variable representing the multeplicity of nodes achieving
			//the highest change in modularity density (within tolerance).
			int multeplicity=0;
			//find all the nodes whose moves result in the maximum change
			//(within the tolerance parameter)
			for (index2=0;index2<N;index2++){
				if (flag_node_blocked[index2]==0){
					for (index3=0;index3<current_number_communities;index3++){
						if (max_dq-dq_steps[index2][index3]<toler){
							node_maximum_modularity[multeplicity][0]=index2;
							node_maximum_modularity[multeplicity][1]=index3;
							multeplicity=multeplicity+1;
						}
					}
				}
			}
			
			//choose the node to move randomly between the ones achieving
			//the highest possible change in modularity
			int node_to_move;
			int random_index=((int)(rand()%multeplicity));
			node_to_move=node_maximum_modularity[random_index][0];
			int destination_community;
			destination_community=node_maximum_modularity[random_index][1];
			//store the node to move in the sequence and the destination
			//community
			sequence_of_nodes_to_move[index1][0]=node_to_move;
			sequence_of_nodes_to_move[index1][1]=destination_community;
			//block the node that has been moved
			flag_node_blocked[node_to_move]=1;
			//find the increase in modularity density obtained so far
			if (index1==0){
				dq_final_tuning_tree[0]=dq_steps[node_to_move][destination_community];
			}
			else {
				dq_final_tuning_tree[index1]=dq_final_tuning_tree[index1-1]+dq_steps[node_to_move][destination_community];
			}
			
			//to make the following updates easier, label correctly the
			//communities which are involved in the move
			int old_community_label;
			old_community_label=temporary_community_list[node_to_move]-1;
			int new_community_label;
			new_community_label=destination_community;			
			//update the temporary community list
			temporary_community_list[node_to_move]=new_community_label+1;
			
			//the next section performs updates to the split penalty
			//vector due to the node that changed community and updates
			//also the vector storing the external number of links of
			//each community
			double single_community_contribution=0.0;
			double old_contribution1, old_contribution2, new_contribution1, new_contribution2;
			double temp_contribution1, temp_contribution2, temp_contribution3;
			double temp_split_penalty_contribution=0.0;
			double temp_external_links=0.0;
			
			//split the for loop to avoid the indices being equal and
			//to avoid using an if statement
			int community_index;
			
			if (old_community_label < new_community_label){
				//update the entries corresponding to the communities not
				//involved in the move. Note that for these communities
				//the overall external number of links has not changed.
				for (community_index=0;community_index<old_community_label;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				
				//update the entry corresponding to the old community
				temp_split_penalty_contribution=0.0;
				temp_external_links=0.0;
				for (community_index=0;community_index<old_community_label;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
				}
				for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
				}
				for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
				}

				temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]);
				new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
				temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label];
				
				split_penalty[old_community_label]=temp_split_penalty_contribution;
				external_links[old_community_label]=temp_external_links;
				
				//update the entry corresponding to the new community
				temp_split_penalty_contribution=0.0;
				temp_external_links=0.0;
				for (community_index=0;community_index<old_community_label;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
				}
				for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
				}
				for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
				}
				
				temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]);
				new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
				temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][old_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label];
				
				split_penalty[new_community_label]=temp_split_penalty_contribution;
				external_links[new_community_label]=temp_external_links;
				
				//The next section of code performs an update to the temporary
				//community density moving the node that was identified as
				//giving the highest increase in modularity density.
	
				//updates due to the node changing community
				temporary_community_density[old_community_label][old_community_label]=temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[node_to_move][old_community_label];
				temporary_community_density[new_community_label][new_community_label]=temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][new_community_label];
				temporary_community_density[old_community_label][new_community_label]=temporary_community_density[old_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label];
				temporary_community_density[new_community_label][old_community_label]=temporary_community_density[old_community_label][new_community_label];
				
				//updates due to the nodes connected to the nodes changing community
				for (community_index=0;community_index<old_community_label;community_index++){
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
				for (community_index=old_community_label+1;community_index<new_community_label;community_index++){
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
				for (community_index=new_community_label+1;community_index<current_number_communities;community_index++){
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
			}
			else{
				//update the entries corresponding to the communities not
				//involved in the move. Note that for these communities
				//the overall external number of links has not changed.
				for (community_index=0;community_index<new_community_label;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
					old_contribution1=((temporary_community_density[community_index][old_community_label]*temporary_community_density[community_index][old_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[old_community_label]));
					old_contribution2=((temporary_community_density[community_index][new_community_label]*temporary_community_density[community_index][new_community_label])/((double)temporary_community_size[community_index]*(double)temporary_community_size[new_community_label]));
					temp_contribution1=(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][old_community_label]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)temporary_community_size[community_index]*(double)(temporary_community_size[old_community_label]-1.0));
					temp_contribution2=(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[community_index][new_community_label]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution2=temp_contribution2/((double)temporary_community_size[community_index]*(double)(temporary_community_size[new_community_label]+1.0));
					split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution1+new_contribution2;
				}
				
				//update the entry corresponding to the old community
				temp_split_penalty_contribution=0.0;
				temp_external_links=0.0;
				for (community_index=0;community_index<new_community_label;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
				}
				for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
				}
				for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
					temp_contribution1=(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[old_community_label]-1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
				}

				temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]);
				new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
				temp_external_links=temp_external_links+(double)temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label];
				
				split_penalty[old_community_label]=temp_split_penalty_contribution;
				external_links[old_community_label]=temp_external_links;
				
				//update the entry corresponding to the new community
				temp_split_penalty_contribution=0.0;
				temp_external_links=0.0;
				for (community_index=0;community_index<new_community_label;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
				}
				for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
				}
				for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
					temp_contribution1=(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index])*(temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index]);
					new_contribution1=temp_contribution1/((double)(temporary_community_size[new_community_label]+1.0)*(double)temporary_community_size[community_index]);
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
					temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
				}
				
				temp_contribution1=(temporary_community_density[old_community_label][new_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]);
				new_contribution1=(temp_contribution1*temp_contribution1)/((double)(temporary_community_size[old_community_label]-1.0)*(double)(temporary_community_size[new_community_label]+1.0));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution1;
				temp_external_links=temp_external_links+(double)temporary_community_density[new_community_label][old_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label];
				
				split_penalty[new_community_label]=temp_split_penalty_contribution;
				external_links[new_community_label]=temp_external_links;
				
				//The next section of code performs an update to the temporary
				//community density moving the node that was identified as
				//giving the highest increase in modularity density.
	
				//updates due to the node changing community
				temporary_community_density[old_community_label][old_community_label]=temporary_community_density[old_community_label][old_community_label]-temporary_edge_community_matrix[node_to_move][old_community_label];
				temporary_community_density[new_community_label][new_community_label]=temporary_community_density[new_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][new_community_label];
				temporary_community_density[old_community_label][new_community_label]=temporary_community_density[old_community_label][new_community_label]+temporary_edge_community_matrix[node_to_move][old_community_label]-temporary_edge_community_matrix[node_to_move][new_community_label];
				temporary_community_density[new_community_label][old_community_label]=temporary_community_density[old_community_label][new_community_label];
				
				//updates due to the nodes connected to the nodes changing community
				for (community_index=0;community_index<new_community_label;community_index++){
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
				for (community_index=new_community_label+1;community_index<old_community_label;community_index++){
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
				for (community_index=old_community_label+1;community_index<current_number_communities;community_index++){
					//update the entry representing the number of links between
					//the old community and every community to which the node
					//moving is connected to
					temporary_community_density[old_community_label][community_index]=temporary_community_density[old_community_label][community_index]-temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][old_community_label]=temporary_community_density[old_community_label][community_index];
					
					//update the entry representing the number of links between
					//the new community and every community to which the node
					//moving is connected to
					temporary_community_density[new_community_label][community_index]=temporary_community_density[new_community_label][community_index]+temporary_edge_community_matrix[node_to_move][community_index];
					temporary_community_density[community_index][new_community_label]=temporary_community_density[new_community_label][community_index];
				}
			}
			
			//the following section performs a local update to the temporary
			//edge community matrix. The update concerns the node that has
			//been decided to move in the previous step.
			
			//create a temporary variable storing the degree of the node which
			//changed community.
			int number_of_links;
			number_of_links=degree_sequence[node_to_move];
			
			int i;
			for (i=0;i<number_of_links;i++){		
				//update the edge_community matrix
				temporary_edge_community_matrix[adj_list[node_to_move][i]][old_community_label]=temporary_edge_community_matrix[adj_list[node_to_move][i]][old_community_label]-1;
				temporary_edge_community_matrix[adj_list[node_to_move][i]][new_community_label]=temporary_edge_community_matrix[adj_list[node_to_move][i]][new_community_label]+1;
			}
			
			//the following section performs an update to the temporary
			//community size vector
			temporary_community_size[old_community_label]=temporary_community_size[old_community_label]-1;
			temporary_community_size[new_community_label]=temporary_community_size[new_community_label]+1;
			

			//the following section updates the vector storing the first
			//part of the modularity density. Note that only the two
			//entries relative to the old and new community get updated.
			
			//update of the entry relative to the old community
			temp1=((double)temporary_community_density[old_community_label][old_community_label])/((double)number_edges);
			temp2=(2.0*(double)temporary_community_density[old_community_label][old_community_label])/((double)temporary_community_size[old_community_label]*((double)temporary_community_size[old_community_label]-1.0));
			temp3=(2.0*(double)temporary_community_density[old_community_label][old_community_label]+external_links[old_community_label])/(2.0*(double)number_edges);
			first_contribution_to_modularity_density[old_community_label]=temp1*temp2-(temp3*temp2)*(temp3*temp2);
			
			//update of the entry relative to the new community
			temp1=((double)temporary_community_density[new_community_label][new_community_label])/((double)number_edges);
			temp2=(2.0*(double)temporary_community_density[new_community_label][new_community_label])/((double)temporary_community_size[new_community_label]*((double)temporary_community_size[new_community_label]-1.0));
			temp3=(2.0*(double)temporary_community_density[new_community_label][new_community_label]+external_links[new_community_label])/(2.0*(double)number_edges);
			first_contribution_to_modularity_density[new_community_label]=temp1*temp2-(temp3*temp2)*(temp3*temp2);
			
			//update the value of modularity density to compare
			old_modularity_density=0.0;
			for (index2=0;index2<current_number_communities;index2++){
				old_modularity_density=old_modularity_density+first_contribution_to_modularity_density[index2]-split_penalty[index2]/((double)2.0*(double)number_edges);
			}
		}

		//find the step in the tree which achieves the highest increase
		//in the value of modularity density. Initialise it to a large
		//negative number
		double maximum_increase=-1000.0;
		for (index2=0;index2<number_of_steps;index2++){
			if (dq_final_tuning_tree[index2]>maximum_increase){
				maximum_increase=dq_final_tuning_tree[index2];
			}
		}

		//if the increase in modularity density is larger than zero (within
		//tolerance), then find if there are other steps which achieve
		//the same (within tolerance) increase and pick one randomly.
		if (maximum_increase>toler){
			
			int multiple_increase=0;
			for (index2=0;index2<number_of_steps;index2++){
				if (maximum_increase-dq_final_tuning_tree[index2]<toler){
					node_maximum_modularity[multiple_increase][0]=index2;
					multiple_increase=multiple_increase+1;
				}
			}
			//pick randomly the step
			int step_in_tree;
			step_in_tree=node_maximum_modularity[((int)(rand()%multiple_increase))][0];
			//update the accepted increase in modularity density
			dq_final=dq_final+dq_final_tuning_tree[step_in_tree];
			
			//then perform all the necessary updates and switches in the
			//community assignment as stored in the decision tree
			index2=0;
			while(index2<=step_in_tree){
				int old_community_label, new_community_label;
				old_community_label=results.community_list[sequence_of_nodes_to_move[index2][0]];
				new_community_label=sequence_of_nodes_to_move[index2][1]+1;
				results.community_list[sequence_of_nodes_to_move[index2][0]]=new_community_label;
				update_edge_community_matrix(sequence_of_nodes_to_move[index2][0], old_community_label-1,new_community_label-1);
				update_community_density_matrix(sequence_of_nodes_to_move[index2][0],old_community_label-1,new_community_label-1);
				update_community_size_vector(old_community_label-1,new_community_label-1);
				index2=index2+1;
			}
			//update the current value of modularity density
			compute_modularity_density(&results.modularity_density);

			//flag that the final tuning step resulted in an increase in
			//the value of modularity density.
			flag_increase=1;
		}
	} while (flag_increase);
	
	
	//free memory of temporary variables
	for (memory_index=0;memory_index<N;memory_index++) free(sequence_of_nodes_to_move[memory_index]);
	free (sequence_of_nodes_to_move);
	free(dq_final_tuning_tree);
	for (memory_index=0;memory_index<N;memory_index++) free(dq_steps[memory_index]);
	free(dq_steps);
	for (memory_index=0;memory_index<N*current_number_communities;memory_index++) free(node_maximum_modularity[memory_index]);
	free(node_maximum_modularity);
	free(flag_node_blocked);
	for (memory_index=0;memory_index<current_number_communities;memory_index++) free(temporary_community_density[memory_index]);
	free(temporary_community_density);
	free(temporary_community_size);
	for (memory_index=0;memory_index<N;memory_index++) free(temporary_edge_community_matrix[memory_index]);
	free(temporary_edge_community_matrix);
	free(temporary_community_list);
	free(first_contribution_to_modularity_density);
	free(split_penalty);
	free(external_links);
	
	return;
}
 
/* This function computes the change in modularity that would result if
 * we merge two communities. It receives as input the following
 * variables:
 * - the first community to consider in the merging
 * - the second community to consider in the merging
 * - the temporary number of communities in this agglomeration step
 * - a pointer to a variable that stores the change in modularity
 *   density that would result from moving the node
 * - a matrix which is a temporary copy of the community density matrix
 * - a matrix which is a temporary copy of the edge community matrix
 * - a vector which is a temporary copy of the community size vector
 * - a vector which contains the contributions to the first part of
 *   modularity density
 * - a vector which contains all the split penalty terms
 */
void compute_change_modularity_density_agglomeration( int first_community, int second_community, int *flag_community_blocked, double *local_dq, long int **temporary_community_density,long int **temporary_edge_community_matrix, int *temporary_community_size,double *contributions_to_modularity_density,double *split_penalty)
{
	//variable that stores the modularity density value if the node
	//node_to_move changes community
	double temp_modularity_density=0.0;
	
	int community_index;
	double single_community_contribution=0.0;
	double old_contribution1, old_contribution2, temp_contribution, temp_contribution1, temp_contribution2, new_contribution;
	double temp_split_penalty_contribution=0.0;
	double external_links=0.0;
	//contribution to modularity density given by all the communities
	//which are not the two we are trying to merge
	
	//to split the for loops and avoid too many if statements, check
	//which community has the largest index and separate the two cases
	if (first_community < second_community){
		for (community_index=0;community_index<first_community;community_index++){
			if (flag_community_blocked[community_index]==0){
				old_contribution1=((temporary_community_density[community_index][first_community]*temporary_community_density[community_index][first_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[first_community]));
				old_contribution2=((temporary_community_density[community_index][second_community]*temporary_community_density[community_index][second_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[second_community]));
				temp_contribution=(double)temporary_community_density[community_index][first_community]+(double)temporary_community_density[community_index][second_community];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution)/((double)2.0*(double)number_edges);
				temp_modularity_density=temp_modularity_density+single_community_contribution;
			}
		}
		for (community_index=first_community+1;community_index<second_community;community_index++){
			if (flag_community_blocked[community_index]==0){
				old_contribution1=((temporary_community_density[community_index][first_community]*temporary_community_density[community_index][first_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[first_community]));
				old_contribution2=((temporary_community_density[community_index][second_community]*temporary_community_density[community_index][second_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[second_community]));
				temp_contribution=(double)temporary_community_density[community_index][first_community]+(double)temporary_community_density[community_index][second_community];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution)/((double)2.0*(double)number_edges);
				temp_modularity_density=temp_modularity_density+single_community_contribution;
			}
		}
		for (community_index=second_community+1;community_index<current_number_communities;community_index++){
			if (flag_community_blocked[community_index]==0){
				old_contribution1=((temporary_community_density[community_index][first_community]*temporary_community_density[community_index][first_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[first_community]));
				old_contribution2=((temporary_community_density[community_index][second_community]*temporary_community_density[community_index][second_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[second_community]));
				temp_contribution=(double)temporary_community_density[community_index][first_community]+(double)temporary_community_density[community_index][second_community];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution)/((double)2.0*(double)number_edges);
				temp_modularity_density=temp_modularity_density+single_community_contribution;
			}
		}
		
		//now compute the contribution to modularity density given by the
		//new community resulting from the agglomeration
		
		//compute the temporary split penalty term
		
		for (community_index=0;community_index<first_community;community_index++){
			if (flag_community_blocked[community_index]==0){
				temp_contribution=(double)temporary_community_density[first_community][community_index]+(double)temporary_community_density[second_community][community_index];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution;
				external_links=external_links+(double)temporary_community_density[first_community][community_index]+temporary_community_density[second_community][community_index];
			}
		}
		for (community_index=first_community+1;community_index<second_community;community_index++){
			if (flag_community_blocked[community_index]==0){
				temp_contribution=(double)temporary_community_density[first_community][community_index]+(double)temporary_community_density[second_community][community_index];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution;
				external_links=external_links+(double)temporary_community_density[first_community][community_index]+temporary_community_density[second_community][community_index];
			}
		}
		for (community_index=second_community+1;community_index<current_number_communities;community_index++){
			if (flag_community_blocked[community_index]==0){
				temp_contribution=(double)temporary_community_density[first_community][community_index]+(double)temporary_community_density[second_community][community_index];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution;
				external_links=external_links+(double)temporary_community_density[first_community][community_index]+temporary_community_density[second_community][community_index];
			}
		}
	}
	else{
		for (community_index=0;community_index<second_community;community_index++){
			if (flag_community_blocked[community_index]==0){
				old_contribution1=((temporary_community_density[community_index][first_community]*temporary_community_density[community_index][first_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[first_community]));
				old_contribution2=((temporary_community_density[community_index][second_community]*temporary_community_density[community_index][second_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[second_community]));
				temp_contribution=(double)temporary_community_density[community_index][first_community]+(double)temporary_community_density[community_index][second_community];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution)/((double)2.0*(double)number_edges);
				temp_modularity_density=temp_modularity_density+single_community_contribution;
			}
		}
		for (community_index=second_community+1;community_index<first_community;community_index++){
			if (flag_community_blocked[community_index]==0){
				old_contribution1=((temporary_community_density[community_index][first_community]*temporary_community_density[community_index][first_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[first_community]));
				old_contribution2=((temporary_community_density[community_index][second_community]*temporary_community_density[community_index][second_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[second_community]));
				temp_contribution=(double)temporary_community_density[community_index][first_community]+(double)temporary_community_density[community_index][second_community];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution)/((double)2.0*(double)number_edges);
				temp_modularity_density=temp_modularity_density+single_community_contribution;
			}
		}
		for (community_index=first_community+1;community_index<current_number_communities;community_index++){
			if (flag_community_blocked[community_index]==0){
				old_contribution1=((temporary_community_density[community_index][first_community]*temporary_community_density[community_index][first_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[first_community]));
				old_contribution2=((temporary_community_density[community_index][second_community]*temporary_community_density[community_index][second_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[second_community]));
				temp_contribution=(double)temporary_community_density[community_index][first_community]+(double)temporary_community_density[community_index][second_community];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				single_community_contribution=contributions_to_modularity_density[community_index]-(split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution)/((double)2.0*(double)number_edges);
				temp_modularity_density=temp_modularity_density+single_community_contribution;
			}
		}
		
		//now compute the contribution to modularity density given by the
		//new community resulting from the agglomeration
		
		//compute the temporary split penalty term
		
		for (community_index=0;community_index<second_community;community_index++){
			if (flag_community_blocked[community_index]==0){
				temp_contribution=(double)temporary_community_density[first_community][community_index]+(double)temporary_community_density[second_community][community_index];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution;
				external_links=external_links+(double)temporary_community_density[first_community][community_index]+temporary_community_density[second_community][community_index];
			}
		}
		for (community_index=second_community+1;community_index<first_community;community_index++){
			if (flag_community_blocked[community_index]==0){
				temp_contribution=(double)temporary_community_density[first_community][community_index]+(double)temporary_community_density[second_community][community_index];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution;
				external_links=external_links+(double)temporary_community_density[first_community][community_index]+temporary_community_density[second_community][community_index];
			}
		}
		for (community_index=first_community+1;community_index<current_number_communities;community_index++){
			if (flag_community_blocked[community_index]==0){
				temp_contribution=(double)temporary_community_density[first_community][community_index]+(double)temporary_community_density[second_community][community_index];
				new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]));
				temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution;
				external_links=external_links+(double)temporary_community_density[first_community][community_index]+temporary_community_density[second_community][community_index];
			}
		}
	}
	
	//then compute the first term in the modularity density formula
	temp_contribution=(double)temporary_community_density[first_community][first_community]+(double)temporary_community_density[second_community][second_community]+(double)temporary_community_density[first_community][second_community];
	temp_contribution1=((double)2.0*temp_contribution)/(((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community])*((double)temporary_community_size[first_community]+(double)temporary_community_size[second_community]-1));
	temp_contribution2=(((double)2.0*temp_contribution+external_links)/((double)2.0*(double)number_edges))*temp_contribution1;
	new_contribution=(temp_contribution*temp_contribution1)/((double)number_edges)-temp_contribution2*temp_contribution2;
	//and then add them to the temporary value of the modularity density
	temp_modularity_density=temp_modularity_density+new_contribution-temp_split_penalty_contribution/((double)2.0*(double)number_edges);
	//difference between the modularity density value when we move
	//node_to_move and the current modularity density value
	double temp_local=temp_modularity_density-results.modularity_density;
	*local_dq=temp_modularity_density;
	return;
}

/* This function merges two communities into a single one. It receives
 * as input the labels of the two communities. Note that the second
 * input is always corresponding to the community with largest index.
 */
void merge_communities(int first_community_to_merge, int second_community_to_merge)
{	
	int index;
	
	//update the temporary community list
	for(index=0;index<N;index++){
		if (results.community_list[index]==second_community_to_merge+1){
			results.community_list[index]=first_community_to_merge+1;
		}
	}
	
	//Update the community density matrix
	
	//to split the for loops and avoid too many if statements, check
	//which community has the largest index and separate the two cases
	if ( first_community_to_merge < second_community_to_merge){
		for (index=0;index<first_community_to_merge;index++){
			//update the connections between the agglomerated community
			//and all the other ones
			community_density[index][first_community_to_merge]=community_density[index][first_community_to_merge]+community_density[index][second_community_to_merge];
			community_density[first_community_to_merge][index]=community_density[index][first_community_to_merge];
			//set to zero the entries corresponding to the
			//community which "disappears"
			community_density[index][second_community_to_merge]=0;
			community_density[second_community_to_merge][index]=0;
		}
		for (index=first_community_to_merge+1;index<second_community_to_merge;index++){
			//update the connections between the agglomerated community
			//and all the other ones
			community_density[index][first_community_to_merge]=community_density[index][first_community_to_merge]+community_density[index][second_community_to_merge];
			community_density[first_community_to_merge][index]=community_density[index][first_community_to_merge];
			//set to zero the entries corresponding to the
			//community which "disappears"
			community_density[index][second_community_to_merge]=0;
			community_density[second_community_to_merge][index]=0;
		}
		for (index=second_community_to_merge;index<current_number_communities;index++){
			//update the connections between the agglomerated community
			//and all the other ones
			community_density[index][first_community_to_merge]=community_density[index][first_community_to_merge]+community_density[index][second_community_to_merge];
			community_density[first_community_to_merge][index]=community_density[index][first_community_to_merge];
			//set to zero the entries corresponding to the
			//community which "disappears"
			community_density[index][second_community_to_merge]=0;
			community_density[second_community_to_merge][index]=0;
		}
	}
	else{
		for (index=0;index<second_community_to_merge;index++){
			//update the connections between the agglomerated community
			//and all the other ones
			community_density[index][first_community_to_merge]=community_density[index][first_community_to_merge]+community_density[index][second_community_to_merge];
			community_density[first_community_to_merge][index]=community_density[index][first_community_to_merge];
			//set to zero the entries corresponding to the
			//community which "disappears"
			community_density[index][second_community_to_merge]=0;
			community_density[second_community_to_merge][index]=0;
		}
		for (index=second_community_to_merge+1;index<first_community_to_merge;index++){
			//update the connections between the agglomerated community
			//and all the other ones
			community_density[index][first_community_to_merge]=community_density[index][first_community_to_merge]+community_density[index][second_community_to_merge];
			community_density[first_community_to_merge][index]=community_density[index][first_community_to_merge];
			//set to zero the entries corresponding to the
			//community which "disappears"
			community_density[index][second_community_to_merge]=0;
			community_density[second_community_to_merge][index]=0;
		}
		for (index=first_community_to_merge;index<current_number_communities;index++){
			//update the connections between the agglomerated community
			//and all the other ones
			community_density[index][first_community_to_merge]=community_density[index][first_community_to_merge]+community_density[index][second_community_to_merge];
			community_density[first_community_to_merge][index]=community_density[index][first_community_to_merge];
			//set to zero the entries corresponding to the
			//community which "disappears"
			community_density[index][second_community_to_merge]=0;
			community_density[second_community_to_merge][index]=0;
		}
	}
			
	//update the internal connections of the agglomerated
	//community
	community_density[first_community_to_merge][first_community_to_merge]=community_density[first_community_to_merge][first_community_to_merge]+community_density[second_community_to_merge][second_community_to_merge]+community_density[first_community_to_merge][second_community_to_merge];
	//set to zero the remaining entries to the community which
	//is "disappearing"
	community_density[first_community_to_merge][second_community_to_merge]=0;
	community_density[second_community_to_merge][first_community_to_merge]=0;
	community_density[second_community_to_merge][second_community_to_merge]=0;
	
	//update the edge community matrix
			
	for(index=0;index<N;index++){
		edge_community_matrix[index][first_community_to_merge]=edge_community_matrix[index][first_community_to_merge]+edge_community_matrix[index][second_community_to_merge];
		edge_community_matrix[index][second_community_to_merge]=0;
	}
			
	//update the community size vector
	community_size[first_community_to_merge]=community_size[first_community_to_merge]+community_size[second_community_to_merge];
	community_size[second_community_to_merge]=0;
			
	
	return;
}

void merge_memory_after_agglomeration_step(int communities_cancelled[current_number_communities], int new_number_communities)
{
	int i,j;
	//create temporary copies of the variables
	
	int *temp_community_size;
	long int **temp_edge_community_matrix, **temp_community_density;
	
	temp_edge_community_matrix=malloc(N*sizeof(long int*));		
	for (i=0;i<N;i++){
		temp_edge_community_matrix[i]=(long int *)malloc(sizeof(long int)*current_number_communities);
		for (j=0;j<current_number_communities;j++){
			temp_edge_community_matrix[i][j]=edge_community_matrix[i][j];
		}
	}
	

	temp_community_size=malloc(sizeof(int)*current_number_communities);
	for (i=0;i<current_number_communities;i++) temp_community_size[i]=community_size[i];

	temp_community_density=(long int **)malloc(sizeof(long int *)*current_number_communities);
	for (i=0;i<current_number_communities;i++){
		temp_community_density[i]=(long int *)malloc(sizeof(long int)*current_number_communities);
		for (j=0;j<current_number_communities;j++){
			temp_community_density[i][j]=community_density[i][j];
		}
	}
	
	//free the old memory used
	for (i=0;i<current_number_communities;i++) free(community_density[i]);
	free(community_density);
	free(community_size);
	for (i=0;i<N;i++) free(edge_community_matrix[i]);
	free(edge_community_matrix);
	
	//allocate new memory for the global variables with the correct size
	//given by the new number of communities
	int index=0;
	edge_community_matrix=malloc(N*sizeof(long int*));		
	for (i=0;i<N;i++){
		edge_community_matrix[i]=(long int *)malloc(sizeof(long int)*new_number_communities);
		for (j=0;j<current_number_communities;j++){
			if(communities_cancelled[j]==0){
				edge_community_matrix[i][index]=temp_edge_community_matrix[i][j];
				index=index+1;
			}
		}
		index=0;
	}

	index=0;
	community_size=malloc(sizeof(int)*new_number_communities);
	for (i=0;i<current_number_communities;i++){
		if (communities_cancelled[i]==0){
			community_size[index]=temp_community_size[i];
			index=index+1;
		}
	}

	community_density=malloc(sizeof(long int *)*new_number_communities);
	for (i=0;i<new_number_communities;i++){
		community_density[i]=malloc(sizeof(long int)*new_number_communities);
	}

	int index1=0, temp_flag=0;
	index=0;
	for (i=0;i<current_number_communities;i++){
		for (j=0;j<current_number_communities;j++){
			if (communities_cancelled[i]==0 && communities_cancelled[j]==0){
				community_density[index][index1]=temp_community_density[i][j];
				index1=index1+1;
				temp_flag=1;
			}	
		}
		if (temp_flag){
			index=index+1;
		}
		temp_flag=0;
		index1=0;
	}
	index=1;
	//fix the labels in the community list
	int flag_label=0;
	for (i=0;i<current_number_communities;i++){
		for (j=0;j<N;j++){
			if (results.community_list[j]==i+1){
				results.community_list[j]=index;
				
				flag_label=1;
			}
		}
		if (flag_label==1){
			index=index+1;
		}
		flag_label=0;
	}
	
	//free the temporary memory used
	free(temp_community_size);
	for (i=0;i<N;i++) free(temp_edge_community_matrix[i]);
	free(temp_edge_community_matrix);
	for (i=0;i<current_number_communities;i++) free(temp_community_density[i]);
	free(temp_community_density);
	
	//update the global variable storing the number of communities
	current_number_communities=new_number_communities;
	
	return;
}
 
/* This function implements the agglomeration step of the algorithm
 * 
 */
void agglomeration()
{
	//initialise the global variable that stores the increase in
	//modularity density obtained in the agglomeration step
	dq_agglom=0.0;
	
	//variable storing the old value of modularity density initialised
	//to the global value of modularity density obtained so far
	double old_modularity_density;
	
	int memory_index;
	//array storing the sequence of communities to merge
	int **sequence_of_communities_to_merge;
	sequence_of_communities_to_merge=malloc((current_number_communities-1)*sizeof(int *));
	for(memory_index=0;memory_index<current_number_communities-1;memory_index++) sequence_of_communities_to_merge[memory_index]=malloc(2*sizeof(int));
	//array storing the increase in modularity density at each step
	double *dq_agglomeration_tree;
	dq_agglomeration_tree=(double *)malloc((current_number_communities-1)*sizeof(double));
	//matrix storing the potential increase in modularity density
	//resulting from temporarily merging two communities
	double **dq_steps;
	dq_steps=(double **)malloc(current_number_communities*sizeof(double *));
	for(memory_index=0;memory_index<current_number_communities;memory_index++) dq_steps[memory_index]=(double *)malloc(current_number_communities*sizeof(double));
	//array containing the labels of communities corresponding to the
	//maximum modularity density increase (at any given step)
	int **community_maximum_modularity;
	community_maximum_modularity=malloc(current_number_communities*current_number_communities*sizeof(int *));
	for (memory_index=0;memory_index<current_number_communities*current_number_communities;memory_index++) community_maximum_modularity[memory_index]=malloc(2*sizeof(int));
	//array containing a flag to indicate whether a community is blocked
	//(i.e. it has been merged and does not exist)
	int *flag_community_blocked;
	flag_community_blocked=malloc(current_number_communities*sizeof(int));
	//local variable that will represent a temporary copy of the
	//community density matrix.
	long int **temporary_community_density;
	temporary_community_density=malloc(current_number_communities*sizeof(long int *));
	for(memory_index=0;memory_index<current_number_communities;memory_index++) temporary_community_density[memory_index]=malloc(current_number_communities*sizeof(long int));
	//local variable that will represent a temporary copy of the
	//community size vector
	int *temporary_community_size;
	temporary_community_size=malloc(current_number_communities*sizeof(int));
	//create split penalty vector, first contribution to
	//modularity density and number of external links arrays
	double *first_contribution_to_modularity_density;
	double *split_penalty;
	double *external_links;
	first_contribution_to_modularity_density=(double *)malloc(current_number_communities*sizeof(double));
	split_penalty=(double *)malloc(current_number_communities*sizeof(double));
	external_links=(double *)malloc(current_number_communities*sizeof(double));
	//local variable that will represent a temporary copy of the edge
	//community matrix.
	long int **temporary_edge_community_matrix;
	temporary_edge_community_matrix=malloc(N*sizeof(long int *));
	for(memory_index=0;memory_index<N;memory_index++) temporary_edge_community_matrix[memory_index]=malloc(current_number_communities*sizeof(long int));
	//local copy of the community assignments.
	int *temporary_community_list;
	temporary_community_list=malloc(N*sizeof(int));
	
	
	//repeat the agglomeration step as long as it results in an increase
	//in the value of modularity density. We use a flag variable to
	//indicate whether the step has achieved an increase (within
	//tolerance) or not. If there is an increase, we repeat.
	int flag_increase=0;
	
	do{		
	
		//set the flag to 0 at the start of every repetition
		flag_increase=0;
		
		//initialise the variable storing the old value of the
		//modularity density
		old_modularity_density=results.modularity_density;
		
		//variable counting how many steps have been performed
		int number_of_steps=0;
		
		//Initialise the nodes as all being unblocked
		int i;
		for (i=0;i<current_number_communities;i++){ flag_community_blocked[i]=0;}
		
		//temporary variable to store the modularity density change that
		//would result from the move of a node
		double temp_dq;
		temp_dq=0.0;
		
		//local variable that will represent a temporary copy of the
		//community density matrix.
		int index1,index2,index3,index4;
		//create the local copy of the community density matrix
		for (index1=0;index1<current_number_communities;index1++){
			for (index2=0;index2<current_number_communities;index2++){
				temporary_community_density[index1][index2]=community_density[index1][index2];
			}
		}
		
		//local variable that will represent a temporary copy of the
		//community size vector
		//create the local copy of the community size vector
		for (index1=0;index1<current_number_communities;index1++){
			temporary_community_size[index1]=community_size[index1];
		}
		
		//create split penalty vector, first contribution to
		//modularity density and number of external links arrays
		for (index1=0;index1<current_number_communities;index1++){
			split_penalty[index1]=0;
			external_links[index1]=0;
			//split the for loop to avoid the indices being equal and
			//to avoid using an if statement
			for (index2=0;index2<index1;index2++){
				split_penalty[index1]=split_penalty[index1]+((double)community_density[index1][index2]*(double)community_density[index1][index2])/((double)community_size[index1]*(double)community_size[index2]);
				external_links[index1]=external_links[index1]+(double)community_density[index1][index2];
			}
			for (index2=index1+1;index2<current_number_communities;index2++){
				split_penalty[index1]=split_penalty[index1]+((double)community_density[index1][index2]*(double)community_density[index1][index2])/((double)community_size[index1]*(double)community_size[index2]);
				external_links[index1]=external_links[index1]+(double)community_density[index1][index2];
			}
		}
		double temp1,temp2,temp3;
		for (index1=0;index1<current_number_communities;index1++){
			temp1=((double)community_density[index1][index1])/((double)number_edges);
			temp2=(2.0*(double)community_density[index1][index1])/((double)community_size[index1]*((double)community_size[index1]-1.0));
			temp3=(2.0*(double)community_density[index1][index1]+external_links[index1])/(2.0*(double)number_edges);
			first_contribution_to_modularity_density[index1]=temp1*temp2-(temp3*temp2)*(temp3*temp2);
		}
		
		//local variable that will represent a temporary copy of the edge
		//community matrix.
		//create the local copy of the edge community matrix
		for (index1=0;index1<N;index1++){
			for (index2=0;index2<current_number_communities;index2++){
				temporary_edge_community_matrix[index1][index2]=edge_community_matrix[index1][index2];
			}
		}
		
		//local copy of the community assignments.
		for (index1=0;index1<N;index1++){
			temporary_community_list[index1]=results.community_list[index1];
		}
		
		//temporary number of communities
		int temporary_number_communities;
		temporary_number_communities=current_number_communities;		
		
		//try and merge each pair of communities, compute the potential
		//change in modularity density and then merge the two which
		//results in the highest increase. If more than one pair achieves
		//the highest increase (within a tolerance limit set as a global
		//parameter), then pick randomly.
		for (index1=0;index1<current_number_communities-1;index1++){
			
			//try to merge every pair of communities and store the change
			//in modularity density that would result from the merging
			#pragma omp parallel private(index3,temp_dq)
			{
				#pragma omp for
				for (index2=0;index2<current_number_communities;index2++){
						if (flag_community_blocked[index2]==0){
						//start this for loop with the appropriate index so that
						//we do not consider merging a pair of communities twice.
						//This will also automatically make sure that we do not
						//try to merge a community with itself
						for(index3=index2+1;index3<current_number_communities;index3++){
							//if the communities are not blocked, try merging them
							if (flag_community_blocked[index3]==0 && temporary_community_density[index2][index3]>0){
								compute_change_modularity_density_agglomeration(index2,index3, flag_community_blocked, &temp_dq, temporary_community_density, temporary_edge_community_matrix, temporary_community_size, first_contribution_to_modularity_density, split_penalty);
								dq_steps[index2][index3]=temp_dq-old_modularity_density;
								//also set the lower diagonal part of the matrix
								//equal to a large negative number so that when
								//we look for the maximum we do not select any
								//of these.
								dq_steps[index3][index2]=-1000.0;
							}
							//otherwise, assign a large negative value to the
							//potential increase in modularity density
							else{
								dq_steps[index2][index3]=-1000.0;
							}
						}
					}
					//set the diagonal equal to a large negative number
					//since it would correspond to a community merged to
					//itself
					dq_steps[index2][index2]=-1000.0;
				}
			}
			//initialise the maximum increase in modularity to a very
			//large negative number
			double max_dq=-1000.0;
			//find the maximum modularity density change
			for (index2=0;index2<current_number_communities;index2++){
				if (flag_community_blocked[index2]==0){
					for (index3=index2+1;index3<current_number_communities;index3++){
						if (dq_steps[index2][index3]>max_dq){
							max_dq=dq_steps[index2][index3];
						}
					}
				}
			}
			if (max_dq==-1000.0){
				number_of_steps=index1;
				break;
			}
			else{
				number_of_steps=index1;
			}
			
			//variable representing the multeplicity of communities achieving
			//the highest change in modularity density (within tolerance).
			int multeplicity=0;
			//find all the communities whose moves result in the maximum
			//change (within the tolerance parameter)
			for (index2=0;index2<current_number_communities;index2++){
				if (flag_community_blocked[index2]==0){
					for (index3=index2+1;index3<current_number_communities;index3++){
						if (max_dq-dq_steps[index2][index3]<toler && flag_community_blocked[index3]==0){
							community_maximum_modularity[multeplicity][0]=index2;
							community_maximum_modularity[multeplicity][1]=index3;
							multeplicity=multeplicity+1;
						}
					}
				}
			}
			
			//choose the communities to merge randomly between the ones
			//achieving the highest possible change in modularity
			int first_merging_community, second_merging_community;
			int random_index=((int)(rand()%multeplicity));
			first_merging_community=community_maximum_modularity[random_index][0];
			second_merging_community=community_maximum_modularity[random_index][1];
			//store the sequence of communities to merge
			sequence_of_communities_to_merge[index1][0]=first_merging_community;
			sequence_of_communities_to_merge[index1][1]=second_merging_community;
			//variables storing the community with the largest index which
			//will be merged and the community with the smallest index
			//which will be the label after the merging
			int community_merged, agglomerated_community;
			//block the community with the highest index that has been
			//merged to another one. Note that the highest index is
			//always stored in the variable labeled by second_merging_community
			flag_community_blocked[second_merging_community]=1;
			agglomerated_community=first_merging_community+1;
			community_merged=second_merging_community+1;
			
			//find the increase in modularity density obtained so far
			if (index1==0){
				dq_agglomeration_tree[0]=dq_steps[first_merging_community][second_merging_community];
			}
			else {
				dq_agglomeration_tree[index1]=dq_agglomeration_tree[index1-1]+dq_steps[first_merging_community][second_merging_community];
			}
	
			//update the temporary community list
			for(index2=0;index2<N;index2++){
					if (temporary_community_list[index2]==community_merged){
						temporary_community_list[index2]=agglomerated_community;
					}
			}
			
			//update the temporary number of communities
			temporary_number_communities=temporary_number_communities-1;			
			
			//the next section performs the required updates to the
			//various terms needed to compute the modularity density
			
			int community_index;
			double single_community_contribution=0.0;
			double old_contribution1, old_contribution2, temp_contribution, temp_contribution1, temp_contribution2, new_contribution;
			//contribution to modularity density given by all the communities
			//which are not the two we are trying to merge
			#pragma omp parallel
			{
				#pragma omp for private(old_contribution1, old_contribution2, temp_contribution, new_contribution)
				for (community_index=0;community_index<current_number_communities;community_index++){
					if (community_index!=first_merging_community && community_index !=second_merging_community && flag_community_blocked[community_index]==0){
						old_contribution1=((temporary_community_density[community_index][first_merging_community]*temporary_community_density[community_index][first_merging_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[first_merging_community]));
						old_contribution2=((temporary_community_density[community_index][second_merging_community]*temporary_community_density[community_index][second_merging_community])/((double)temporary_community_size[community_index]*(double)temporary_community_size[second_merging_community]));
						temp_contribution=(double)temporary_community_density[community_index][first_merging_community]+(double)temporary_community_density[community_index][second_merging_community];
						new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_merging_community]+(double)temporary_community_size[second_merging_community]));
						split_penalty[community_index]=split_penalty[community_index]-old_contribution1-old_contribution2+new_contribution;
					}
				}
			}
			
			//now compute the split penalty for the community which
			//results from the merging and also the number of external
			//links
			double temp_split_penalty_contribution=0.0;
			double temp_external_links=0.0;
			for (community_index=0;community_index<current_number_communities;community_index++){
				if (community_index!=first_merging_community && community_index!=second_merging_community && flag_community_blocked[community_index]==0){
					temp_contribution=(double)temporary_community_density[first_merging_community][community_index]+(double)temporary_community_density[second_merging_community][community_index];
					new_contribution=(temp_contribution*temp_contribution)/((double)temporary_community_size[community_index]*((double)temporary_community_size[first_merging_community]+(double)temporary_community_size[second_merging_community]));
					temp_split_penalty_contribution=temp_split_penalty_contribution+new_contribution;
					temp_external_links=temp_external_links+(double)temporary_community_density[first_merging_community][community_index]+temporary_community_density[second_merging_community][community_index];
				}
			}
			external_links[agglomerated_community-1]=temp_external_links;
			split_penalty[agglomerated_community-1]=temp_split_penalty_contribution;
			
			//set to zero the split penalty term and the number of
			//external links for the "disappearing" community
			external_links[community_merged-1]=0.0;
			split_penalty[community_merged-1]=0.0;
			
			
			//The next section of code performs an update to the temporary
			//community density by merging the two identified communities

			for (community_index=0;community_index<current_number_communities;community_index++){
				//update the connections between the agglomerated community
				//and all the other ones
				if (community_index!=first_merging_community && community_index!=second_merging_community && flag_community_blocked[community_index]==0){
					temporary_community_density[community_index][agglomerated_community-1]=temporary_community_density[community_index][agglomerated_community-1]+temporary_community_density[community_index][community_merged-1];
					temporary_community_density[agglomerated_community-1][community_index]=temporary_community_density[community_index][agglomerated_community-1];
					//set to zero the entries corresponding to the
					//community which "disappears"
					temporary_community_density[community_index][community_merged-1]=0;
					temporary_community_density[community_merged-1][community_index]=0;
				}
			}
			
			//update the internal connections of the agglomerated
			//community
			temporary_community_density[agglomerated_community-1][agglomerated_community-1]=temporary_community_density[agglomerated_community-1][agglomerated_community-1]+temporary_community_density[community_merged-1][community_merged-1]+temporary_community_density[agglomerated_community-1][community_merged-1];
			//set to zero the remaining entries to the community which
			//is "disappearing"
			temporary_community_density[agglomerated_community-1][community_merged-1]=0;
			temporary_community_density[community_merged-1][agglomerated_community-1]=0;
			temporary_community_density[community_merged-1][community_merged-1]=0;
			
			//the following section performs a local update to the temporary
			//edge community matrix. 
			
			for(index2=0;index2<N;index2++){
				temporary_edge_community_matrix[index2][agglomerated_community-1]=temporary_edge_community_matrix[index2][agglomerated_community-1]+temporary_edge_community_matrix[index2][community_merged-1];
				temporary_edge_community_matrix[index2][community_merged-1]=0;
			}
			
			//the following section performs an update to the temporary
			//community size vector
			temporary_community_size[agglomerated_community-1]=temporary_community_size[agglomerated_community-1]+temporary_community_size[community_merged-1];
			temporary_community_size[community_merged-1]=0;
			

			//the following section updates the vector storing the first
			//part of the modularity density. Note that only the two
			//entries corresponding to the merging communities are
			//involved in this update
			
			//update of the entry relative to the agglomerated community
			temp1=((double)temporary_community_density[agglomerated_community-1][agglomerated_community-1])/((double)number_edges);
			temp2=(2.0*(double)temporary_community_density[agglomerated_community-1][agglomerated_community-1])/((double)temporary_community_size[agglomerated_community-1]*((double)temporary_community_size[agglomerated_community-1]-1.0));
			temp3=(2.0*(double)temporary_community_density[agglomerated_community-1][agglomerated_community-1]+external_links[agglomerated_community-1])/(2.0*(double)number_edges);
			first_contribution_to_modularity_density[agglomerated_community-1]=temp1*temp2-(temp3*temp2)*(temp3*temp2);
			
			//update of the entry relative to the "disappearing" community
			first_contribution_to_modularity_density[community_merged-1]=0.0;

			
			//update the value of modularity density to compare
			old_modularity_density=0.0;
			for (index2=0;index2<current_number_communities;index2++){
				if(flag_community_blocked[index2]==0){
					old_modularity_density=old_modularity_density+first_contribution_to_modularity_density[index2]-split_penalty[index2]/((double)2.0*(double)number_edges);
				}
			}
				
		}

		//find the step in the tree which achieves the highest increase
		//in the value of modularity density. Initialise it to a large
		//negative number
		double maximum_increase=-1000.0;
		index2=0;
		do{
			if (dq_agglomeration_tree[index2]>maximum_increase){
				maximum_increase=dq_agglomeration_tree[index2];
			}
			index2=index2+1;
		} while (index2<number_of_steps);
		
		//if the increase in modularity density is larger than zero (within
		//tolerance), then find if there are other steps which achieve
		//the same (within tolerance) increase and pick the one with
		//the smallest number of communities
		if (maximum_increase>toler){
			int multiple_increase=0;
			index2=0;
			do{
				if (maximum_increase-dq_agglomeration_tree[index2]<toler){
					community_maximum_modularity[multiple_increase][0]=index2;
					multiple_increase=multiple_increase+1;
				}
				index2=index2+1;
			} while (index2<number_of_steps);
			
			//pick the largest step which corresponds to the smallest
			//number of communities
			int step_in_tree;
			step_in_tree=community_maximum_modularity[multiple_increase-1][0];
			//update the accepted increase in modularity density
			dq_agglom=dq_agglom+dq_agglomeration_tree[step_in_tree];
			
			//then perform all the necessary merges and also store the
			//labels of communities that have been merged and do not
			//exist anymore
			int communities_cancelled[current_number_communities];
			for (index2=0;index2<current_number_communities;index2++) communities_cancelled[index2]=0;
			//number of communities that have been merged (to obtain the
			//new overall number of communities)
			index2=0;
			int number_communities_merged=0;
			while(index2<=step_in_tree){
				//communities to merge at this step
				int first_community_to_merge, second_community_to_merge;
				first_community_to_merge=sequence_of_communities_to_merge[index2][0];
				second_community_to_merge=sequence_of_communities_to_merge[index2][1];
				//store the label of the community getting cancelled
				communities_cancelled[second_community_to_merge]=1;
				//count the number of communities that have been merged
				number_communities_merged=number_communities_merged+1;
				//merge the two communities
				merge_communities(first_community_to_merge, second_community_to_merge);
				index2=index2+1;
			}
			
			//remove the unnecessary empty communities from memory and
			//correctly reallocate the memory for the global variables
			merge_memory_after_agglomeration_step(communities_cancelled, current_number_communities-number_communities_merged);
			//update the current value of modularity density
			compute_modularity_density(&results.modularity_density);
			//flag that the final tuning step resulted in an increase in
			//the value of modularity density (if there are at least
			//two communities in the current partition).
			if (current_number_communities>1){
				flag_increase=1;
			}
			else{
				flag_increase=0;
			}
		}
	} while (flag_increase);
	
	
	//free memory used for local variables
	for (memory_index=0;memory_index<current_number_communities-1;memory_index++) free(sequence_of_communities_to_merge[memory_index]);
	free(sequence_of_communities_to_merge);
	free(dq_agglomeration_tree);
	for (memory_index=0;memory_index<current_number_communities;memory_index++) free(dq_steps[memory_index]);
	free(dq_steps);
	for (memory_index=0;memory_index<current_number_communities*current_number_communities;memory_index++) free(community_maximum_modularity[memory_index]);
	free(community_maximum_modularity);
	free(flag_community_blocked);
	for (memory_index=0;memory_index<current_number_communities;memory_index++) free(temporary_community_density[memory_index]);
	free(temporary_community_density);
	free(temporary_community_size);
	free(first_contribution_to_modularity_density);
	free(split_penalty);
	free(external_links);
	for (memory_index=0;memory_index<N;memory_index++) free(temporary_edge_community_matrix[memory_index]);
	free(temporary_edge_community_matrix);
	free(temporary_community_list);
	
	return;
}

 
 /* This function frees the unnecessary memory at the end of the
  * algorithm
  */
void free_memory()
{
	int i;
	for (i=0;i<current_number_communities;i++) free(community_density[i]);
	free(community_density);
	for (i=0;i<N;i++) free(edge_community_matrix[i]);
	free(edge_community_matrix);
	free(community_size);
	for (i=0;i<N;i++) free(adj_list[i]);
	free(adj_list);
	for (i=0;i<N;i++) free(adj_mat[i]);
	free(adj_mat);
	free(degree_sequence);
	free(community_blocked);
	free(results.community_list);
	return;
}

/* This function writes to one file the labels of each node, as detected
 * by the algorithm, and to another file the value of modularity density
 * that has been found.
 */
void write_to_file(int flag_index)
{
	//prepare output files
	char communities_file_name[100],modularity_density_file_name[100];
	sprintf(communities_file_name,"communities_%i.txt",flag_index);
	sprintf(modularity_density_file_name,"modularity_density_%i.txt",flag_index);
	
	FILE *communities_file, *modularity_density_file;
	communities_file=fopen(communities_file_name,"w");
	modularity_density_file = fopen(modularity_density_file_name,"w");

	//output to file the communities
	int i;
	for (i=0; i<N;i++){
		fprintf(communities_file, "%i ", results.community_list[i]);
	}
	
	//output the corresponding value of modularity density
	fprintf(modularity_density_file, "%lf", results.modularity_density);

	//close the corresponding files
	fclose(communities_file);
	fclose(modularity_density_file);	
	return;
}

/* This function implements the algorithm that maximises the value of
 * modularity density. It includes the following steps:
 * - initial bisection
 * - fine tuning
 * - repeated bisections and fine tuning steps until possible
 * - final tuning step
 * - agglomeration step
 */
void community_detection_via_modularity_density_maximisation(double * values_modularity_density, int iteration)
{
	//initialise the value of modularity density to 0
	results.modularity_density=0.0;
	
	//create the vector storing the degree sequence
	create_degree_sequence();
	//create adjacency list
	adjlist_from_adjmat();
	
	double modularity_check_for_bisection=0.0;
	
	//initialise the current number of communities to 1
	current_number_communities=1;
	
	//initalise community density matrix with the total number of edges
	community_density[0][0]=number_edges;
	
	//initialise the community structure so that all nodes are in a
	//single community
	int i;
	for (i=0;i<N;i++){
		results.community_list[i]=1;
	}
	
	//initialise the flag variable that block communities
	community_blocked[current_number_communities-1]=0;
	
	//initialise the edge community matrix so that all nodes have a
	//number of connection to the only community equal to their degree
	for (i=0;i<N;i++) edge_community_matrix[i][0]=degree_sequence[i];
	
	working_community=1;
	community_size[working_community-1]=N;

	//store value of modularity density after execution of algorithm to
	//check whether to repeat or not
	double old_modularity_density_value=0.0;
	//flag variable to check whether we increased modularity density
	//or not
	int flag_repeat=0;
	//flag to indicate that we have done at least one step in the
	//algorithm
	int flag_step=0;
	do{
		flag_repeat=0;
		do{
			//current nodes under consideration
			int current_number_nodes;
			current_number_nodes=community_size[working_community-1];

			int current_nodes[current_number_nodes];
			int i,label;
			label=0;
			for (i=0;i<N;i++){
				if (results.community_list[i]==working_community){
					current_nodes[label]=i;
					label=label+1;
				}			
			}

			//construct the modularity matrix for the nodes under consideration
			double **mod_mat;
			int j;
			create_sub_modularity_matrix(&mod_mat, current_nodes, current_number_nodes);

			
			//variables storing the leading eigenvalue and eigenvector
			double *leading_eigenvalue;
			double *leading_eigenvector;
			leading_eigenvalue=(double *)malloc(sizeof(double));
			leading_eigenvector=(double *)malloc(current_number_nodes*sizeof(double));
			
			//calculate leading eigenvalue and leading eigenvector for the
			//modularity matrix relative to the nodes under consideration

			int flag_bisection=0;
			int flag_fine_tuning=0;
			if (current_number_nodes > 2){
				power_method(mod_mat, current_number_nodes, leading_eigenvalue, leading_eigenvector);
				int positive_count=0, negative_count=0;
				for (i=0;i<current_number_nodes;i++){
					if(leading_eigenvector[i]>0){
						positive_count=positive_count+1;
					}
					if(leading_eigenvector[i]<0){
						negative_count=negative_count+1;
					}
					if (positive_count >= 2 && negative_count >= 2){
						flag_bisection=1;
					}
					if (positive_count + negative_count > 4){
						flag_fine_tuning=1;
					}
				}
				
			}
			else{
				flag_bisection=0;
			}
			//free memory of modularity matrix
			for (i=0;i<current_number_nodes;i++) free(mod_mat[i]);
			free(mod_mat);

			//if the leading eigenvalue just found is positive
			// and the resulting communities would have at least size
			//two (each), bisect the current community
			if (*leading_eigenvalue>0 && flag_bisection){
				modularity_check_for_bisection=results.modularity_density;
				bisection(leading_eigenvector, current_nodes, current_number_nodes);
				compute_modularity_density(&results.modularity_density);
				
				//merge the communities together if their split has
				//resulted in a decrease in the value of modularity
				//density larger than that set by the parameter
				//toler_bisec
				if (modularity_check_for_bisection-results.modularity_density>toler_bisec && flag_step){
					
					community_blocked[working_community-1]=1;
					int *flag_merge;
					flag_merge=malloc(current_number_communities*sizeof(int));
					for (i=0;i<current_number_communities;i++) flag_merge[i]=0;
					flag_merge[current_number_communities-1]=1;
					merge_communities(working_community-1,current_number_communities-1);
					merge_memory_after_agglomeration_step(flag_merge, current_number_communities-1);
					free(flag_merge);
					flag_fine_tuning=0;
				}
				//flag that one iteration of the bisection and fine tuning steps
				//has been performed
				flag_step=1;
				//if the (sum of the) size of the two communities is 
				//(strictly) larger than 4, then execute the fine tuning
				//step
				if (flag_fine_tuning){
					fine_tuning(current_nodes, current_number_nodes, working_community, current_number_communities);
				}
				//update the current value of modularity density
				compute_modularity_density(&results.modularity_density);
				//printf("Modularity density after fine tuning: %lf\n", results.modularity_density);
			}
			//else flag the current community as blocked, so that we won't
			//try to bisect it in following steps
			else{
				community_blocked[working_community-1]=1;
			}
			//free the memory allocated for the eigenvalue
			free(leading_eigenvalue);
			//free the memory used by the eigenvector
			free(leading_eigenvector);
			compute_modularity_density(&results.modularity_density);
			//if the current working community is flagged as blocked,
			//increase the working community by 1; otherwise, repeat the
			//steps on the current working community
			if (community_blocked[working_community-1]){
				working_community=working_community+1;
			}
		} while (working_community<=current_number_communities);
		//update value of modularity density after bisection and fine tuning
		compute_modularity_density(&results.modularity_density);
		if (current_number_communities>1){
			//final tuning step
			final_tuning();
			//printf("Modularity density after final tuning: %lf\n", results.modularity_density);
			//update value of modularity density after final tuning
			compute_modularity_density(&results.modularity_density);
			//agglomeration step
			agglomeration();
			//printf("Modularity density after agglomeration: %lf\n", results.modularity_density);
			//update value of modularity density after agglomeration
			compute_modularity_density(&results.modularity_density);
			//if this iteration has increased the value of modularity
			//density, then repeat
			if (results.modularity_density-old_modularity_density_value>0){
				working_community=1;
				old_modularity_density_value=results.modularity_density;
				flag_repeat=1;
			}
		}
		else{
			flag_repeat=0;
		}
	} while (flag_repeat);
	//compute the final value of modularity density obtained
	compute_modularity_density(&results.modularity_density);

	values_modularity_density[iteration]=results.modularity_density;
	//output to file
	write_to_file(iteration);
	//free the memory allocated during the algorithm
	free_memory();
return;
}
