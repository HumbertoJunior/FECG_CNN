#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Data.h"

#define numberOfFilters 32

#define	sampleLength 60
#define firstFilterLength 7
#define poolLength 2
#define secondFilterLength 5
#define thirdFilterLength 3
#define fourthFilterLength 1
#define flattenLenght numberOfFilters*(sampleLength/poolLength)
#define firstFullyLenght 124
#define secondFullyLenght 64

#define NEW_MAX(x,y) ((x) >= (y)) ? (x) : (y)
#define RELU(x)(x>0?x:0)
#define SIGMOID(x) (2.0f / (1 + exp(-x)))


//Global Variables ============
float sample[sampleLength] = SAMPLE_FILE;

float firstConvFilter[numberOfFilters][firstFilterLength] = {0};
float secondConvFilter[numberOfFilters][secondFilterLength] = {0};	
float thirdConvFilter[numberOfFilters][thirdFilterLength] = {0};
float fourthConvFilter[numberOfFilters][fourthFilterLength] = {0};
float firstFullyParameters[firstFullyLenght] = {0};
float secondFullyParameters[secondFullyLenght] = {0};
float outputParameter = {0};

float firstConvBias[numberOfFilters] = {0};
float secondConvBias[numberOfFilters] = {0};
float thirdConvBias[numberOfFilters] = {0};
float fourthConvBias[numberOfFilters] = {0};
float firstFullyBias[firstFullyLenght] = {0};
float secondFullyBias[secondFullyLenght] = {0};
float outputBias = 0;

float firstConvOutput[numberOfFilters][sampleLength] =  {0};
float maxpoolingOutput[numberOfFilters][sampleLength/poolLength] =  {0};
float secondConvOutput[numberOfFilters][sampleLength/poolLength] =  {0};
float thirdConvOutput[numberOfFilters][sampleLength/poolLength] =  {0};
float fourthConvOutput[numberOfFilters][sampleLength/poolLength] =  {0};
float flattenOutput[flattenLenght] =  {0};
float firstFullyOutput[firstFullyLenght] = {0};
float secondFullyOutput[secondFullyLenght] = {0};
float output = 0;

float sampleAdjusted[sampleLength+(firstFilterLength-1)] = {0}; 
float secondConvInput[numberOfFilters][sampleLength/poolLength+(secondFilterLength-1)] = {0}; 
float thirdConvInput[numberOfFilters][sampleLength/poolLength+(thirdFilterLength-1)] = {0}; 


//Functions====================
void adjust_input(float samples[])
{	
	for (int i=0; i<sampleLength; i++)
	{
		sampleAdjusted[i+(firstFilterLength/2)] = samples[i];
	}
}

void conv1()
{
	for(int j=0; j<numberOfFilters; j++)
	{
		for(int i=0; i<sampleLength; i++)
		{
			for(int k=0; k<firstFilterLength; k++)
			{
				firstConvOutput[j][i] += (sampleAdjusted[i+k]*firstConvFilter[j][k]);
			}	
			firstConvOutput[j][i] += firstConvBias[j];
			firstConvOutput[j][i] = RELU(firstConvOutput[j][i]);
			// printf("%d\t", firstLayerOutput[j][i]);
		}
		// printf("\n");
	}
	// printf("\n");
}


void maxpooling()
{
	for(int j=0; j<numberOfFilters; j++)
	{
		for(int i=0; i<sampleLength; i=i+poolLength)
		{
			maxpoolingOutput[j][i/poolLength] = NEW_MAX(firstConvOutput[j][i],firstConvOutput[j][i+1]);
			// printf("%d\t", secondLayerOutput[j][i/2]);
		}
		// printf("\n");
	}
	// printf("\n");
}

void adjustSecondConvInput(float samples[numberOfFilters][sampleLength/poolLength])
{
	for(int j=0; j<numberOfFilters; j++)
	{
		for(int i=0; i<sampleLength/poolLength; i++)
		{
			secondConvInput[j][i+(secondFilterLength/2)] = samples[j][i];
		}
	}
}


void conv2()
{
	for(int j=0; j<numberOfFilters; j++)
	{
		for(int i=0; i<sampleLength/poolLength; i++)
		{
			for(int k=0; k<secondFilterLength; k++)
			{
				secondConvOutput[j][i] += (secondConvInput[j][i+k]*secondConvFilter[j][k]);
			}	
			secondConvOutput[j][i] += secondConvBias[j];
			secondConvOutput[j][i] = RELU(secondConvOutput[j][i]);
			// printf("%d\t", secondLayerOutput[j][i]);
		}
		// printf("\n");
	}
	// printf("\n");
}

void adjustThirdConvInput(float samples[numberOfFilters][sampleLength/poolLength])
{
	for(int j=0; j<numberOfFilters; j++)
	{
		for(int i=0; i<sampleLength/poolLength; i++)
		{
			thirdConvInput[j][i+(thirdFilterLength/2)] = samples[j][i];
		}
	}
}

void conv3()
{
	for(int j=0; j<numberOfFilters; j++)
	{
		for(int i=0; i<sampleLength/poolLength; i++)
		{
			for(int k=0; k<thirdFilterLength; k++)
			{
				thirdConvOutput[j][i] += (thirdConvInput[j][i+k]*thirdConvFilter[j][k]);
			}	
			thirdConvOutput[j][i] += thirdConvBias[j];
			thirdConvOutput[j][i] = RELU(thirdConvOutput[j][i]);
			// printf("%d\t", secondLayerOutput[j][i]);
		}
		// printf("\n");
	}
	// printf("\n");
}

void conv4()
{
	for(int j=0; j<numberOfFilters; j++)
	{
		for(int i=0; i<sampleLength/poolLength; i++)
		{
			fourthConvOutput[j][i] += (thirdConvOutput[j][i]*fourthConvFilter[j][0]) + fourthConvBias[j];	
			fourthConvOutput[j][i] = RELU(fourthConvOutput[j][i]);
			// printf("%d\t", secondLayerOutput[j][i]);
		}

		// printf("\n");
	}
	// printf("\n");
}

void flatten()
{
	int count = 0;
	for(int j=0; j<numberOfFilters; j++)
	{
		for(int i=0; i<sampleLength/poolLength; i++)
		{
			flattenOutput[count] = fourthConvOutput[j][i];
			// printf("%d\n", flattenOutput[count]);
			count++;
		}	
	}
	// printf("\n");
}

void fully1()
{
	for(int j=0; j<firstFullyLenght; j++)
	{
		for (int i=0; i<flattenLenght; ++i)
		{
			firstFullyOutput[j] += (flattenOutput[i]*firstFullyParameters[j]) + firstFullyBias[j];
		}
		firstFullyOutput[j] = RELU(firstFullyOutput[j]);
		// printf("%d\n", fully1Output[j]);
	}
	// printf("\n");
}

void fully2()
{
	for(int j=0; j<secondFullyLenght; j++)
	{
		for (int i=0; i<firstFullyLenght; ++i)
		{
			secondFullyOutput[j] += (firstFullyOutput[i]*secondFullyParameters[j]) + secondFullyBias[j];
		}
		secondFullyOutput[j] = RELU(secondFullyOutput[j]);
		// printf("%d\n", fully1Output[j]);
	}
	// printf("\n");
}

void outputLayer()
{
	for (int i=0; i<secondFullyLenght; ++i)
	{
		output += (secondFullyOutput[i]*outputParameter) + outputBias;
	}
	// printf("%f\n", output);

	output = SIGMOID(output);
}


//Main ========================
int main()
{
		adjust_input(sample);
	
	conv1();
	
	maxpooling();

		adjustSecondConvInput(maxpoolingOutput);
	
	conv2();
	
		adjustThirdConvInput(secondConvOutput);
	
	conv3();
	
		// Not necessary to adjust the sample, for the filterLenght in the fourth conv layer is 1

	conv4();

	flatten();

	fully1();

	fully2();

	outputLayer();

	return 0;
}



