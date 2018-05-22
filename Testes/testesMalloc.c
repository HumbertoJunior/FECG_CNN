
#include "stdlib.h"
#include "stdio.h"

// Matriz de 5 linhas por 3 colunas

#define rows 5
#define cols 3

int a[15] = {1,2,1,0,1,2,1,2,3,1,3,4,1,2,5};


int i = 0;
int j = 0;

int main(int argc, char const *argv[])
{
	

int *mat = (int *)malloc(rows * cols * sizeof(int));

	 mat = {1,2,1,0,1,2,1,2,3,1,3,4,1,2,5};

	printf("%d\n", mat[10]);
	free(mat);
	mat = NULL;
	return 0;
}




   // int *v; 
   // int n;
   // scanf ("%d", &n);
   // v = malloc (n * sizeof (int));
   // for (int i = 0; i < n; ++i) 
   //    scanf ("%d", &v[i]);
   // . . . 
   // free (v);



// int **M; 
// 	   M = malloc (5 * sizeof (int *));
// 	   for (int i = 0; i < 5; ++i)
// 	      M[i] = malloc (3 * sizeof (int));
	
// 	  for (int l = 0; l < 5; ++l)
// 	  {
// 		  	for (int k = 0; k < 3; ++k)
// 		  	{
// 		  		&M[l] = a[l][k]
// 		  	}
// 	  }