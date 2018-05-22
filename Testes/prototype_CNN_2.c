#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define	sampleLength 8
#define firstFilterLength 3
#define fully1Lenght 3
#define new_max(x,y) ((x) >= (y)) ? (x) : (y)
//Global Variables ============

volatile int sample [sampleLength] = {1,1,0,1,1,1,0,1};
volatile int sampleAdjusted [sampleLength+2] = {0};

volatile int filter[firstFilterLength][firstFilterLength] = 	{{0,1,1},
													{1,-1,1},
													{1,0,1}};
volatile int fullyParameters[3] = {0,1,2};
volatile int outputParameter = 1;

volatile int firstLayerOutput[firstFilterLength][sampleLength] =  {0};
volatile int secondLayerOutput[firstFilterLength][sampleLength/2] =  {0};
volatile int fully1Output[3] = {0};
volatile int flattenOutput[12] =  {0};
volatile int output = 0;

volatile int biasConv[3] = {1,2,1};
volatile int biasFully[3] = {1,-1,0};
volatile int biasOut = 1;

#define RELU(x)(x>0?x:0)
#define SIGMOID(x) (1.f / (1 + exp(-x)))


volatile int *ledr_ptr = ( int*)0xbf800000 ;
		
//Functions====================

void adjust_sample(volatile int samples[])
{

	for (volatile unsigned int i=0; i<sampleLength; i++)
	{
		sampleAdjusted[i+1] = samples[i];
	}
}

void conv1()
{
	for(volatile unsigned int j=0; j<firstFilterLength; j++)
	{
		for(volatile unsigned int i=0; i<sampleLength; i++)
		{
			for(volatile unsigned int k=0; k<3; k++)
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
	for(volatile unsigned int j=0; j<firstFilterLength; j++)
	{
		for(volatile unsigned int i=0; i<sampleLength; i=i+2)
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
	volatile int count = 0;
	for(volatile unsigned int j=0; j<firstFilterLength; j++)
	{
		for(volatile unsigned int i=0; i<sampleLength/2; i++)
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
	for(volatile unsigned int j=0; j<3; j++)
	{
		for (volatile unsigned int i=0; i<(sampleLength/2)*firstFilterLength; ++i)
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
	for (volatile unsigned int i=0; i<fully1Lenght; ++i)
	{
		output += (fully1Output[i]*outputParameter);
	}
	output += biasOut;
	output = RELU(output);
	printf("%d\n", output);
}



void delay () 
{
	volatile unsigned int j ;
	for (j = 0; j < (2500000); j++); 
}

//Main ========================

int main()
{

	adjust_sample(sample);
	
	conv1();

	*ledr_ptr = 1;
	delay();
	
	maxpooling();
	
	*ledr_ptr = 2;
	delay();

	// conv2();
	
	// conv3();
	
	// conv4();

	flatten();

	*ledr_ptr = 3;
	delay();

	fully1();

	*ledr_ptr = 4;
	delay();

	// fully2();

	outputLayer();

	*ledr_ptr = 5;
	delay();

    
		while (1) {
		  *ledr_ptr = output;
		 
		  delay();
		}

	return 0;
}



