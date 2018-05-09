#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define	sampleLength 8
#define firstFilterLength 3
#define fully1Lenght 3
#define new_max(x,y) ((x) >= (y)) ? (x) : (y)
//Global Variables ============

int sample [sampleLength] = {1,1,0,1,1,1,0,1};
int sampleAdjusted [sampleLength+2] = {0};

int filter[firstFilterLength][firstFilterLength] = 	{{0,1,1},
													{1,-1,1},
													{1,0,1}};
int fullyParameters[3] = {0,1,2};
int outputParameter = 1;

int firstLayerOutput[firstFilterLength][sampleLength] =  {0};
int secondLayerOutput[firstFilterLength][sampleLength/2] =  {0};
int fully1Output[3] = {0};
int flattenOutput[12] =  {0};
int output = 0;

int biasConv[3] = {1,2,1};
int biasFully[3] = {1,-1,0};
int biasOut = 1;

#define RELU(x)(x>0?x:0)
#define SIGMOID(x) (1.f / (1 + exp(-x)))


float p = 1.123123123e-01;

//Functions====================

void adjust_sample(int samples[])
{

	for (int i=0; i<sampleLength; i++)
	{
		sampleAdjusted[i+1] = samples[i];
	}
}

void conv1()
{
	for(int j=0; j<firstFilterLength; j++)
	{
		for(int i=0; i<sampleLength; i++)
		{
			for(int k=0; k<3; k++)
			{
				firstLayerOutput[j][i] += (sampleAdjusted[i+k]*filter[j][k]);
			}	
			firstLayerOutput[j][i] += biasConv[j];
			firstLayerOutput[j][i] = RELU(firstLayerOutput[j][i]);
			printf("%d\t", firstLayerOutput[j][i]);
		}
		printf("\n");
	}
	printf("\n");
}


void maxpooling()
{
	for(int j=0; j<firstFilterLength; j++)
	{
		for(int i=0; i<sampleLength; i=i+2)
		{
		secondLayerOutput[j][i/2] = new_max(firstLayerOutput[j][i],firstLayerOutput[j][i+1]);
		printf("%d\t", secondLayerOutput[j][i/2]);
		}
		printf("\n");
	}
	printf("\n");
}

void flatten()
{
	int count = 0;
	for(int j=0; j<firstFilterLength; j++)
	{
		for(int i=0; i<sampleLength/2; i++)
		{
			flattenOutput[count] = secondLayerOutput[j][i];
			printf("%d\n", flattenOutput[count]);
			count++;
		}	
	}
	printf("\n");
}

void fully1()
{
	for(int j=0; j<3; j++)
	{
		for (int i=0; i<(sampleLength/2)*firstFilterLength; ++i)
		{
			fully1Output[j] += (flattenOutput[i]*fullyParameters[j]);
		}
		fully1Output[j] += biasFully[j];
		fully1Output[j] = RELU(fully1Output[j]);
		printf("%d\n", fully1Output[j]);
	}
	printf("\n");
}

void outputLayer()
{
	for (int i=0; i<fully1Lenght; ++i)
	{
		output += (fully1Output[i]*outputParameter);
	}
	output += biasOut;
	output = RELU(output);
	printf("%d\n", output);
}


//Main ========================

int main()
{
	adjust_sample(sample);
	
	conv1();
	
	maxpooling();
	
	// conv2();
	
	// conv3();
	
	// conv4();

	flatten();

	fully1();

	// fully2();

	outputLayer();

    

	return 0;
}



