#include "stdio.h"

#define new_max(x,y) ((x) >= (y)) ? (x) : (y)

int z [32][1] ={{1},
				{2},
				{1},
				{0}};

int a[3][8] = {0};

int b[3] = {1,1,1};
int k=0;
int i=0;
int j=0;

void first_layer_conv () {

	float sampleAdjusted = 1+(7%2);
	printf("%f\n", sampleAdjusted);

	for( j=0; j<32; j++){
		for( i=0; i<1; i++){
			printf("%d\t", z[j][i]);
		}
		printf("\n");
	}
}

int main()
{
	first_layer_conv();
}











