#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
using namespace std;

#include "rapidcsv.h"


string train_csv = "./train.csv";
string test_csv = "./test.csv";
string head_line, line;

unsigned char digit_l[42000];  // label
unsigned char digit_a[42000][28][28]; // 28x28
unsigned int  digit_i = 0;

// port the basic CNN functions
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/stat.h>
#include<unistd.h>
#include<dirent.h>
#include<math.h>
#include<time.h>
#include<pthread.h>


// LOOP
int i,j,k,m,n,p;

// IMAGE DISPLAY
int colorize = 1;
double red[8] =   {1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0};
double green[8] = {0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
double blue[8] =  {0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0};
double red2[256],green2[256],blue2[256];
double red3[256],green3[256],blue3[256];
int image[400][600] = {{0}};
int image2[80][120] = {{0}};
// CONFUSION MATRIX DATA
int maxCD = 54;
int cDigits[10][10][54];
int showAcc = 1;
int showEnt = 1;
int showCon = 0;
int showDig[3][55] = {{0}};
float scaleMin = 0.9, scaleMax = 1.0;

// INIT-NET
void initNet(int t);
void initArch(char *str, int x);
// NEURAL-NET
int isDigits(int init);
void randomizeTrainSet();
void *runBackProp(void *arg);
int backProp(int x,float *ent, int ep);
int forwardProp(int x, int dp, int train, int lay);
float ReLU(float x);
float TanH(float x);

// TRAINING AND VALIDATION DATA
float (*trainImages)[784] = 0;
float (*trainImages2)[196] = 0;
int *trainDigits = 0;
int trainSizeI = 0, extraTrainSizeI = 1000;
int trainColumns = 0, trainSizeE = 0;
int *trainSet = 0; int trainSetSize = 0;
int *validSet = 0; int validSetSize = 0;
float *ents = 0, *ents2 = 0;
float *accs = 0, *accs2 = 0;
// TEST DATA
float (*testImages)[784] = 0;
float (*testImages2)[196] = 0;
int *testDigits;
int testSizeI = 0;
int testColumns = 0;
// LOAD DATA
int loadTrain(int ct, double testProp, int sh, float imgScale, float imgBias);
int loadTest(int ct, int sh, int rc, float imgScale, float imgBias);

// NETWORK VARIABLES
int inited = -1;
int activation = 1; //0=Identity, 1=ReLU, 2=TanH
const int randomizeDescent = 1;
float an = 0.01;
int DOconv=1, DOdense=1, DOpool=1;
float dropOutRatio = 0.0, _decay = 1.0;
float augmentRatio = 0.0, weightScale = 1.0;
float augmentScale = 0, imgBias=0.0;
int augmentAngle = 0;
float augmentDx = 0.0, augmentDy = 0.0;
// NETWORK ACTIVATIONS AND ERRORS
float prob = 0.0, prob0 = 0.0;
float prob1 = 0.0, prob2 = 0.0;
float* layers[10] = {0};
int* dropOut[10] = {0};
float*  weights[10] = {0};
float* errors[10] = {0};
// NETWORK ARCHITECTURE
int numLayers = 0;
char layerNames[10][20] = {0};
int layerType[10] = {0}; //0FC, 1C, 2P
int layerSizes[10] = {0};
int layerConv[10] = {0};
int layerPad[10] = {0};
int layerWidth[10] = {0};
int layerChan[10] = {0};
int layerStride[10] = {0};
int layerConvStep[10] = {0};
int layerConvStep2[10] = {0};

// PREDEFINED NET ARCHITECTURES
char nets[8][10][20] =
{{"","","","","","","","","",""},
		{"","","","","","","2","20","20","6"},
		{"","","","784","C5:6","P2","C5:16","P2","128","10"},
		{"","784","C3:10","C3:10","P2","C3:20","C3:20","P2","128","10"},
		//layer-9: 4-86.5, 5-89.5, 6-90.85, 7-90.75, 8-92.50, 9-92.65, 10-93, 15-94.64, 30-96, 48-96.67, 72-96.93, 
		//98-97.27, 127-97.37, 256-97.46, 512-97.55
		{"","","","","","","","784","5000","10"}, 
		// debug nets below
		{"","","","","","784","C5:6","P2","50","10"},
		{"","","","","","","","196","100","10"},
		{"","","","","","16","C3:2","P2","2","2"}};
// THREAD VARIABLES
pthread_t workerThread;
pthread_attr_t 	stackSizeAttribute;
int pass[5] = {0};
int working = 0;
int requiredStackSize = 8*1024*1024;
int requestInit = 0;
char buffer[1024];


// DOT DATA
const int maxDots=250;
float trainDots[250][2];
int trainColors[250];
int trainSizeD = 0;
// DOT PARAMETERS
int useSmall = -1;
int removeMode = -1;
int dotsMode = 4; //6=fluid display 2=slower
// DOT 3D DISPLAY
int use3D = -1;
float heights3D[121][81] = {{0}};
float pa3D[121][81] = {{0}};
float pb3D[121][81] = {{0}};
float pc3D[121][81] = {{0}};
double *red4=0, *green4=0, *blue4=0;
// MISC
const char *weightsFile1 = "weights1.txt";
const char *weightsFile2 = "weights2.txt";


map<string, string> parms = {
		{"net", "4"},
		{"scaleWeights", "1.414"}
};	
//  parms["net"] = "4";
//  parms["scaleWeights"] = "1.414";


int ipGet(string key) {
		string value = parms[key];
		long int l = 0; char* end; l = strtol(value.c_str(), &end,10);
		return (int)l;
}
void ipSet(string key, int value) {
		char buffer[32]; sprintf(buffer, "%d", value);
		parms[key] = buffer; 
}
double rpGet(string key) {
		string value = parms[key];
		double f = 0.0; char* end; f = strtof(value.c_str(), &end);
		return f;
}
char empty[2] = {0,0};
const char* spGet(const char* key) {
		if (parms.count(key)) return parms[key].c_str();
		else return empty;
}
void spSet(char* key, char* value) {
		parms[key] = value;
}

int* ip=0; double* rp=0; char* sp=0;
void webupdate(int* ip, double* rp, char* sp) {
		cout << "webupdate(..)" << endl;
}
void webwriteline(const char * line) {
		cout << line << endl;
}
void displayDigit(int x, int ct, int p, int lay, int chan, int train, int cfy, int big){
		cout << "displayDigit" << endl;
}
void displayDigits(int *dgs, int ct, int pane, int train, int cfy, int wd, int big){
		cout << "displayDigits" << endl;
}
void websetmode(int mode) {
		cout << "websetmode" << endl;
}
unsigned long fsize(const char* file) {
		/* returns file size */

		FILE * f = fopen(file, "r");
		fseek(f, 0, SEEK_END);
		unsigned long len = (unsigned long)ftell(f);
		fclose(f);
		return len;

}

#define DFUNC(x) { if(x)cout << __FUNCTION__ << endl; }

// DISPLAY PROGRESS
		void displayConfusion(int (*confusion)[10]) DFUNC(false)
		void displayCDigits(int x,int y) DFUNC(true)
		void displayEntropy(float *ents, int entSize, float *ents2, int display) DFUNC(false)
void displayAccuracy(float *accs, int accSize,float *accs2, int display) DFUNC(false)
		// DISPLAY DOTS
		void displayClassify(int dd) DFUNC(true)
void displayClassify3D() DFUNC(true)

#define D(x) cout << #x <<":" << x << endl

		/**********************************************************************/
		/*      LOAD DATA                                                     */
		/**********************************************************************/
		int loadTrain(int ct, double testProp, int sh, float imgScale, float imgBias){
				char *data;
				// LOAD TRAINING DATA FROM FILE
				if (ct<=0) ct=1e6;
				int i, len = 0, lines=1, lines2=1;
				float rnd;
				// READ IN TRAIN.CSV
				char buffer[1000000];
				char name[80] = "train.csv";
				//strcpy(name,spGet("dataFile"));
				if (access(name,F_OK)!=0) sprintf(name,"../%s",spGet("dataFile"));
				if (access(name,F_OK)==0){
						data = (char*)malloc((int)fsize(name)+1);
						FILE *fp;
						fp = fopen(name,"r");
						while (fgets(buffer, 1000000, fp)) {
								len += sprintf(data+len,"%s",buffer);
								//lines++;
						}
						fclose(fp);
				}
				else {
						sprintf(buffer,"ERROR: File %s not found.",name);
						webwriteline(buffer);
						return 0;
				}
				// COUNT LINES
				for (i=0;i<len;i++){
						if (data[i]=='\n') lines++;
						if (data[i]=='\r') lines2++;
				}
				if (lines2>lines) lines=lines2;
				D(lines);
				// ALLOCATE MEMORY
				if (trainImages!=NULL){
						free(trainImages);
						free(trainImages2);
						free(trainDigits);
						free(trainSet);
						free(validSet);
						trainImages = NULL;
				}

				trainImages = (float (*)[784])malloc(784 * (lines+extraTrainSizeI) * sizeof(float));  // float [][784]
				trainImages2 =(float (*)[196]) malloc(196 * (lines+extraTrainSizeI) * sizeof(float)); // float [][196]
				trainDigits = (int*)malloc(lines * sizeof(int));  // int [lines]
				trainSet = (int*)malloc(lines * sizeof(int));     // int [lines]
				validSet = (int*)malloc(lines * sizeof(int));     // int [lines]

				// DECODE COMMA SEPARATED ROWS
				int j = 0, k = 0, c = 0, mark = -1;
				int d = 0, j1,j2;
				while (data[j]!='\n' && data[j]!='\r'){
						if (data[j]==',') c++;
						j++;
				}
				if (data[j]!='\n' || data[j]!='\r') j++;
				trainColumns = c;
				c = 0; i = 0;
				if (sh==1) i = j+1;
				while(i<len && k<ct){
						j = i; while (data[j]!=',' && data[j]!='\r' && data[j]!='\n') j++;
						if (data[j]=='\n' || data[j]=='\r') mark = 1;
						data[j] = 0;
						d = atof(data+i); 
						if (mark == -1){
								trainDigits[k] = (int)d; // save label
								mark = 0;
						}
						else if (mark==0) {
								trainImages[k][c] = d/imgScale - imgBias;  // fill each pixel and convert it to float
								c++;
						}
						if (mark>=1){
								trainImages[k][c] = d/imgScale - imgBias;
								if (c>=trainColumns-1) k++;
								c = 0;
								if (j+1<len && (data[j+1]=='\n' || data[j+1]=='\r')) mark++;
								i = j + mark;
								mark = -1; // new row
						}
						else i = j + 1;
				}
				D(k); // k is all rows that saved
				validSetSize = 0;
				trainSetSize = 0;
				// CREATE A SUBSAMPLED VERSION OF IMAGES
				if (trainColumns==784){
						for (i=0;i<k;i++){
								for (j1=0;j1<14;j1++)
										for (j2=0;j2<14;j2++){
												trainImages2[i][14*j1+j2] = (trainImages[i][28*j1*2+j2*2]
																+ trainImages[i][28*j1*2+j2*2+1]
																+ trainImages[i][28*(j1*2+1)+j2*2] 
																+ trainImages[i][28*(j1*2+1)+j2*2+1])/4.0; // convert 28x28 to 14x14, map (0,0),(0,1),(1,0),(1,1) to 1. using 2x2 square to slide on the 28x28. 
										}
						}
				}
				// CREATE TRAIN AND VALIDATION SETS
				D(testProp); // split % between train and validate
				for (i=0;i<k;i++){
						rnd = (float)rand()/(float)RAND_MAX;
						if (rnd<=testProp) validSet[validSetSize++] = i;
						else trainSet[trainSetSize++] = i;
				}
				D(trainSetSize);
				D(validSetSize);
				trainSizeI = k;
				trainSizeE = k;
				free(data);
				return k;
		}
/**********************************************************************/
/*      LOAD DATA                                                     */
/**********************************************************************/
int loadTest(int ct, int sh, int rc, float imgScale, float imgBias){
		char *data;
		// LOAD TEST DATA FROM FILE
		if (ct<=0) ct=1e6;
		int i,len = 0, lines=0, lines2=0;;
		float rnd;
		// READ IN TEST.CSV
		char buffer[1000000];
		char name[80] = "test.csv";
		//strcpy(name,spGet("dataFile"));
		if (access(name,F_OK)!=0) sprintf(name,"../%s",spGet("dataFile"));
		if (access(name,F_OK)==0){
				data = (char*)malloc((int)fsize(name)+1);
				FILE *fp;
				fp = fopen(name,"r");
				while (fgets(buffer, 1000000, fp)){
						len += sprintf(data+len,"%s",buffer);
						//lines++;
				}
				fclose(fp);
		}
		else {
				sprintf(buffer,"ERROR: File %s not found.",name);
				webwriteline(buffer);
				return 0;
		}
		// COUNT LINES
		for (i=0;i<len;i++){
				if (data[i]=='\n') lines++;
				if (data[i]=='\r') lines2++;
		}
		if (lines2>lines) lines=lines2;

		// ALLOCATE MEMORY
		if (testImages!=NULL){
				free(testImages);
				free(testImages2);
				free(testDigits);
				testImages = NULL;
		}
		testImages = (float (*)[784])malloc(784 * lines * sizeof(float));
		testImages2 =(float (*)[196]) malloc(196 * lines * sizeof(float));
		testDigits = (int*)malloc(lines * sizeof(int));
		// DECODE COMMA SEPARATED ROWS
		int j = 0, k = 0, c = 0, mark = 0;
		int d = 0, j1,j2;
		while (data[j]!='\n' && data[j]!='\r'){
				if (data[j]==',') c++;
				j++;
		}
		if (data[j]!='\n' || data[j]!='\r') j++;
		testColumns = c+1;
		if (rc==1) {
				testColumns--;
				mark = -1;
		}
		//printf("len=%d lines=%d columns=%d\n",len,lines,testColumns);
		c = 0; i = 0;
		if (sh==1) i = j+1;
		while(i<len && k<ct){
				j = i; while (data[j]!=',' && data[j]!='\r' && data[j]!='\n') j++;
				if (data[j]=='\n' || data[j]=='\r') mark = 1;
				data[j] = 0;
				d = atof(data+i);
				if (mark==-1){
						mark = 0;
				}
				else if (mark==0) {
						testImages[k][c] = d/imgScale - imgBias;
						c++;
				}
				if (mark>=1){
						testImages[k][c] = d/imgScale - imgBias;
						if (c>=testColumns-1) k++;
						c = 0;
						if (j+1<len && (data[j+1]=='\n' || data[j+1]=='\r')) mark++;
						i = j + mark;
						mark = 0;
						if (rc==1) mark = -1;
				}
				else i = j + 1;
		}
		// CREATE A SUBSAMPLED VERSION OF IMAGES
		if (testColumns==784){
				for (i=0;i<k;i++){
						for (j1=0;j1<14;j1++)
								for (j2=0;j2<14;j2++){
										testImages2[i][14*j1+j2] = (testImages[i][28*j1*2+j2*2]
														+ testImages[i][28*j1*2+j2*2+1]
														+ testImages[i][28*(j1*2+1)+j2*2]
														+ testImages[i][28*(j1*2+1)+j2*2+1])/4.0;
								}
				}
		}
		testSizeI = k;
		free(data);
		return k;
}


/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
int isDigits(int init){
		// DETERMINES WHETHER TO TRAIN DOTS OR LOADED DATA
		int in = 10-numLayers;
		if (layerSizes[in]==196 || layerSizes[in]==784 || layerSizes[in]==trainColumns) return 1;
		else return 0;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
void randomizeTrainSet(){
		// RANDOMIZES INDICES IN TRAINING SET
		int i, temp, x;
		for (i=0;i<trainSetSize;i++){
				x = (int)(trainSetSize * ((float)rand()/(float)RAND_MAX) - 1);
				temp = trainSet[i];
				trainSet[i] = trainSet[x];
				trainSet[x] = temp;
		}
}

void updateImage() {
		cout << "updateImage()" << endl;
}
/**********************************************************************/
/*      INIT NET                                                      */
/**********************************************************************/
void initNet(int t){
		// ALLOCATION MEMORY AND INITIALIZE NETWORK WEIGHTS
		int i,j, same=1, LL, dd=9;
		char buf[10], buf2[20];
		if (t==0){
				for (i=0;i<10;i++) {
						strcpy(nets[0][i],"0");
						layerType[i] = 0;
				}
				for (i=9;i>=0;i--){
						sprintf(buf,"L%d",i);
						strcpy (buf2,spGet(buf));
						buf2[19]=0;
						if (buf2[0]!=0 && buf2[0]!='0'){
								if (strcmp(buf2,nets[0][dd])!=0) same=0;
								strcpy(nets[0][dd--],buf2);
						}
				}
				if (numLayers!=9-dd) same=0;
		}
		// FREE OLD NET
		if ( (t!=inited && layers[0]!=NULL) || (t==0 && same==0) ){
				free(layers[0]);
				free(errors[0]);
				for (i=1;i<10;i++){
						free(layers[i]);
						free(dropOut[i]);
						free(errors[i]);
						free(weights[i]);
				} 
				layers[0] = NULL;
		}
		// SET NEW NET ARCHITECTURE
		numLayers = 0;
		for (i=0;i<10;i++) {
				initArch(nets[t][i],i);
				sprintf(buf,"L%d",i);
				spSet(buf,nets[t][i]);
				if (numLayers==0 && layerSizes[i]!=0) numLayers = 10-i;
		}
		webupdate(ip,rp,sp);
		//printf("\n");

		// ALOCATE MEMORY
		if (layers[0]==NULL){
				layers[0] = (float*)malloc((layerSizes[0]+1) * sizeof(float));
				errors[0] = (float*)malloc(layerSizes[0] * sizeof(float));
				for (i=1;i<10;i++){
						layers[i] = (float*)malloc((layerSizes[i] * layerChan[i] + 1) * sizeof(float));
						dropOut[i] = (int*)malloc((layerSizes[i] * layerChan[i] + 1) * sizeof(int));
						//printf("setting dropOut i=%d to %d\n",i,(layerSizes[i] * layerChan[i] + 1));
						errors[i] = (float*)malloc((layerSizes[i] * layerChan[i] + 1) * sizeof(float));
						if (layerType[i]==0) {// FULLY CONNECTED
								weights[i] = (float*)malloc(layerSizes[i] * (layerSizes[i-1]*layerChan[i-1]+1) * sizeof(float));
								// ANDY 
								/*
								   weights[7] 784(layersizes[i]) * (0*1+1) (layerSizes[i-1]*layerChan[i-1]+1) = 784
								   weights[8] 4(layersizes[i]) * (784*1+1) (layerSizes[i-1]*layerChan[i-1]+1) = 3140
								   weights[9] 10(layersizes[i]) * (4*1+1) (layerSizes[i-1]*layerChan[i-1]+1) = 50
								   Initialized NN=4 with Xavier init scaled=1.414
								   Architecture (0-0-0-0-0-0-0-784-4-10)
								   Beginning 1000 epochs with lr=0.010000 and decay=0.950000
								   */
								printf("weights[%d] %d(layersizes[i]) * (%d*%d+1) (layerSizes[i-1]*layerChan[i-1]+1) = %d\n", i, layerSizes[i], layerSizes[i-1], layerChan[i-1], layerSizes[i] * (layerSizes[i-1]*layerChan[i-1]+1));
						}else if (layerType[i]==1) // CONVOLUTION
								weights[i] = (float*)malloc((layerConvStep[i]+1) * layerChan[i] * sizeof(float));
						else if (layerType[i]>=2) // POOLING (2=max, 3=avg)
								weights[i] = (float*)malloc( sizeof(float));
				}
		}
		// RANDOMIZE WEIGHTS AND BIAS
		float scale;
		for (i=0;i<10;i++) layers[i][layerSizes[i] * layerChan[i]]=1.0;
		for (j=1;j<10;j++){
				scale = 1.0;
				if (layerSizes[j-1]!=0){
						// XAVIER INITIALIZATION (= SQRT( 6/(N_in + N_out) ) ) What is N_out to MaxPool ??
						if (layerType[j]==0){ // FC LAYER
								if (layerType[j+1]==0)
										scale = 2.0 * sqrt(6.0/ ( layerSizes[j-1]*layerChan[j-1] + layerSizes[j] ));
						}
						//printf("Init layer %d: LS=%d LC=%d LCS=%d Scale=%f\n",j,layerSizes[j],layerChan[j],layerConvStep[j],scale);
						if (j!=9) scale *= weightScale;
				}
				if (layerType[j]==0){ // FULLY CONNECTED
						if(layerSizes[j]) {
								printf("scale[%d] %f\n", j, scale);
								for (i=0;i<layerSizes[j] * (layerSizes[j-1]*layerChan[j-1]+1);i++)
										weights[j][i] = scale * ( (float)rand()/(float)RAND_MAX - 0.5 );
								//weights[j][i] = scale * ( (float)rand()/(float)RAND_MAX - 0.5 );
								// weights[j][i] = -0.01;
						}
				}
		}

		inited = t;
		if (isDigits(inited)!=1) {
				showCon = 0;
				showDig[0][0] = 0;
				updateImage();
		}
}

/**********************************************************************/
/*      INIT NET                                                      */
/**********************************************************************/
void initArch(char *str, int x){
		// PARSES USER INPUT TO CREATE DESIRED NETWORK ARCHITECTURE
		//TODO: remove all spaces, check for invalid characters
		int i;
		char *idx = str, *idx2;
		while (idx[0]==' ' && idx[0]!=0) idx++;
		for (i=0;i<strlen(idx);i++) str[i]=idx[i];
		if (str[0]==0) {str[0]='0'; str[1]=0;}
		if (str[0]>='0' && str[0]<='9'){
				layerSizes[x] = atoi(str);
				layerConv[x] = 0;
				layerChan[x] = 1;
				layerPad[x] = 0;
				layerWidth[x] = (int)sqrt(layerSizes[x]);
				if (layerWidth[x]*layerWidth[x]!=layerSizes[x]) layerWidth[x]=1;
				layerStride[x] = 1;
				layerConvStep[x] = 0;
				layerConvStep2[x] = 0;
				layerType[x] = 0;
		}
		else if (str[0]=='c' || str[0]=='C'){
				int more = 1;
				str[0]='C';
				idx = str+1;
				while(*idx!=':' && *idx!='-' && *idx!=0) idx++;
				if (*idx==0) more = 0; *idx = 0;
				layerConv[x] = atoi(str+1);
				layerChan[x] = 1;
				layerPad[x] = 0;
				//layerWidth[x] = layerWidth[x-1];
				layerWidth[x] = layerWidth[x-1]-layerConv[x]+1;
				if (more==1){
						*idx = ':';
						idx++; idx2 = idx;
						while(*idx!=':' && *idx!='-' && *idx!=0) idx++;
						if (*idx==0) more = 0; *idx = 0;
						layerChan[x] = atoi(idx2);
						if (more==1){
								*idx = ':';
								idx++; idx2 = idx;
								while(*idx!=':' && *idx!='-' && *idx!=0) idx++;
								if (*idx==0) more = 0; *idx = 0;
								layerPad[x] = atoi(idx2);
								if (layerPad[x]==1)
										layerWidth[x] = layerWidth[x-1];
								//layerWidth[x] = layerWidth[x-1]-layerConv[x]+1;
						}
				}
				layerSizes[x] = layerWidth[x] * layerWidth[x];
				layerConvStep2[x] = layerConv[x] * layerConv[x];
				layerConvStep[x] = layerConvStep2[x] * layerChan[x-1];
				layerStride[x] = 1;
				layerType[x] = 1;
		}
		else if (str[0]=='p' || str[0]=='P'){
				int more = 1;
				if (activation!=0) str[0]='P'; // allow avg pool if identity act
				idx = str+1;
				while(*idx!=':' && *idx!='-' && *idx!=0) idx++;
				if (*idx==0) more = 0; *idx = 0;
				layerConv[x] = atoi(str+1);
				layerStride[x] = layerConv[x];
				if (more==1){
						*idx = ':';
						idx++; idx2 = idx;
						while(*idx!=':' && *idx!='-' && *idx!=0) idx++;
						if (*idx==0) more = 0; *idx = 0;
						layerStride[x] = atoi(idx2);
				}
				int newWidth = layerWidth[x-1]/layerStride[x];
				if (layerStride[x]!=layerConv[x])
						newWidth = (layerWidth[x-1]-layerConv[x]+layerStride[x])/layerStride[x];
				layerSizes[x] = newWidth * newWidth;
				layerChan[x] = layerChan[x-1];
				layerPad[x] = 0;
				layerWidth[x] = newWidth;
				layerConvStep2[x] = layerConv[x] * layerConv[x];
				layerConvStep[x] = layerConvStep2[x];
				layerType[x] = 2; // MAX POOLING
				if (str[0]=='p') layerType[x] = 3; // AVG POOLING
		}
		strcpy(layerNames[x],str);
}


void init_net() {
		int t = ipGet("net");  // type.. 4 is 768-1000-1000-10
		weightScale = rpGet("scaleWeights"); // 1.414 sqrt(2)
		if (working==1) requestInit = 1; else initNet(t);
		sprintf(buffer,"Initialized NN=%d with Xavier init scaled=%.3f",t,weightScale);
		webwriteline(buffer);
		int len = sprintf(buffer,"Architecture (%s",layerNames[0]);
		for (i=1;i<10;i++) len += sprintf(buffer+len,"-%s",layerNames[i]);
		sprintf(buffer+len,")");
		webwriteline(buffer);
}

void load_data() {
		int ct = 0; //ipGet("rows");
		double v = 0.24; //rpGet("validRatio");
		int t = 1; //ipGet("trainSet");
		int sh = 1; //ipGet("removeHeader");
		int rc = 0; //ipGet("removeCol1");
		float imgScale = 255.0; //rpGet("divideBy");
		imgBias = 0.0; //rpGet("subtractBy");
		if (t==1){
				webwriteline("Loading training images, please wait...");
				int x = loadTrain(ct,v,sh,imgScale,imgBias);
				sprintf(buffer,"Loaded %d rows training, %d features, vSetSize=%d",x,trainColumns,validSetSize);
				webwriteline(buffer);
		}
		else{
				webwriteline("Loading test images, please wait...");
				int x = loadTest(ct,sh,rc,imgScale,imgBias);
				sprintf(buffer,"Loaded %d rows test, %d features",x,testColumns);
				webwriteline(buffer);
		}
}

void train_and_validate() {
		an = 0.01;//rpGet("learn");
		scaleMin = 0.9; //rpGet("minY");
		scaleMax = 1.0; //rpGet("maxY");
		_decay = 0.95; //rpGet("decay");
		if (working==1){
				sprintf(buffer,"wait until learning ends, learn=%f",an);
				webwriteline(buffer);
		}
		else{
				int x = 1000; //ipGet("epochs");
				int y = 1; //ipGet("displayFreq");
				dotsMode = 4; //ipGet("mode");
				sprintf(buffer,"Beginning %d epochs with lr=%f and decay=%f",x,an,_decay);
				webwriteline(buffer);
				pass[0]=x; pass[1]=y; pass[2]=1; working=1;
				pthread_create(&workerThread,&stackSizeAttribute,runBackProp,NULL);
		}

}

// https://github.com/cdeotte/MNIST-CNN-99.5/blob/master/CNN.c

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
int ftime_cnt = 0;
int ftime_all = 0;
int btime0_all = 0;
int btime1_all = 0;
int btime2_all = 0;
int btime_cnt = 0;
void *runBackProp(void *arg){
		// TRAINS NEURAL NETWORK WITH TRAINING DATA
		time_t start,stop;
		showEnt = 1; showAcc = 1;
		char buffer[80];
		int i, x = pass[0], y = pass[1], z = pass[2];
		int p, confusion[10][10]={{0}};
		// LEARN DIGITS
		int trainSize = trainSetSize;
		int testSize = validSetSize;
		if (isDigits(inited)==1) {
				websetmode(2);
				showCon=1;
		}
		// ALLOCATE MEMORY FOR ENTORPY AND ACCURACY HISTORY
		if (ents!=NULL){
				free(ents); free(ents2); free(accs); free(accs2);
				ents = NULL;
		}
		ents = (float*)malloc( (int)(x/y+1) * sizeof(float) );
		ents2 = (float*)malloc( (int)(x/y+1) * sizeof(float) );
		accs = (float*)malloc( (int)(x/y+1) * sizeof(float) );
		accs2 = (float*)malloc( (int)(x/y+1) * sizeof(float) );
		int entSize = 0, accSize = 0, ent2Size = 0, acc2Size = 0;
		int j,j2,k,s,s2,b;
		float entropy,entropy2,ent;
		time(&start);
		setbuf(stdout, NULL); // make sure no buffer on stdout
		// PERFORM X TRAINING EPOCHS
		for (j=0;j<x;j++){
				s = 0; entropy = 0.0; int skip=0;
				if (isDigits(inited)!=1) trainSize = trainSizeD;
				for (i=0;i<trainSize;i++){
						//if (i%100==0) printf(" EPoch x=%d, i=%d\n",j,i);
						if ( i % (trainSize/100) == 0) printf(".");
						if (rand() % 100 < 90) {skip++;continue;} // simulate mini-batch, ANDY
						if (isDigits(inited)==1) b = backProp(trainSet[i],&ent,j); // LEARN DIGITS
						else b = backProp(i,&ent,0); // LEARN DOTS
						if (b==-1) {
								if (z==1) webwriteline("Exploded. Lower learning rate.");
								else printf("Exploded. Lower learning rate.\n");
								working=0; websetmode(2);
								return NULL;
						}
						s += b;
						entropy += ent;
						if (working==0){
								webwriteline("learning stopped early");
								pthread_exit(NULL);
						}
				}
				printf("> training done\n");//printf(" skip: %d/%d\n", skip, trainSize);
				printf("forward: %d %d  backward: %d %d %d %d\n", ftime_cnt, ftime_all, btime_cnt, btime0_all, btime1_all, btime2_all);
				ftime_cnt = ftime_all = btime_cnt = btime0_all = btime1_all = btime2_all = 0;
				entropy = entropy / trainSize;
				s2 = 0; entropy2 = 0.0;
				for (i=0;i<10;i++) for (k=0;k<10;k++) confusion[i][k]=0;
				for (i=0;i<10;i++) for (j2=0;j2<10;j2++) for (k=0;k<maxCD;k++) cDigits[i][j2][k]= -1;
				if (j%5 != 0) continue; // validate every 5 epoch? ========================================> go back to  training
				/***********************
				 *  Validation
				 **********************/
				for (i=0;i<testSize;i++){
						p = forwardProp(validSet[i],0,1,0);
						if (p==-1) {
								if (z==1) webwriteline("Test exploded.");
								else printf("Test exploded.\n");
								working=0; websetmode(2);
								return NULL;
						}
						if (p==trainDigits[validSet[i]]) s2++;
						cDigits[trainDigits[validSet[i]]][p][ confusion[trainDigits[validSet[i]]][p]%maxCD ] = validSet[i];
						confusion[trainDigits[validSet[i]]][p]++;
						if (layers[9][p]==0){
								if (z==1) webwriteline("Test vanished.");
								else printf("Test vanished.\n");
								working=0; websetmode(2);
								return NULL;
						}
						entropy2 -= log(layers[9][p]);
						if (working==0){
								webwriteline("learning stopped early");
								pthread_exit(NULL);
						}
						if ( i % (testSize/100) == 0) printf(".");
				}
				printf("> validation done\n");
				printf("forward: %d %d \n", ftime_cnt, ftime_all); ftime_cnt = ftime_all = 0;
				entropy2 = entropy2 / testSize;
				if (j==0 || (j+1)%y==0){
						ents[entSize++] = entropy;
						accs[accSize++] = (float)s/trainSize;
						if (isDigits(inited)==1) {
								accs2[acc2Size++] = (float)s2/testSize;
								ents2[ent2Size++] = entropy2;
						}
						time(&stop);
						sprintf(buffer,"i=%d acc=%d/%d, ent=%.4f, lr=%.1e",j+1,s,trainSize,entropy,an*pow(_decay,j));
						if (isDigits(inited)==1 && testSize>0) sprintf(buffer,"i=%d train=%.2f ent=%.4f,valid=%.2f ent=%.4f (%.0fsec) lr=%.1e",
										j+1,100.0*s/trainSize,entropy,100.0*s2/testSize,entropy2,difftime(stop,start),an*pow(_decay,j));
						else if (isDigits(inited)==1 && testSize==0) sprintf(buffer,"i=%d train=%.2f ent=%.4f (%.0fsec) lr=%.1e",
										j+1,100.0*s/trainSize,entropy,difftime(stop,start),an*pow(_decay,j));
						time(&start);
						if (z==1) webwriteline(buffer);
						else printf("%s\n",buffer);
						if (z==1 && isDigits(inited)!=1) {
								if (use3D==1) displayClassify3D();
								else displayClassify(0);
						}
						if (z==1 && showEnt==1) displayEntropy(ents,entSize,ents2,y);
						if (z==1 && showAcc==1) displayAccuracy(accs,accSize,accs2,y);
						if (z==1 && isDigits(inited)==1 && showCon==1)  displayConfusion(confusion);
				}
				if (requestInit==1){
						initNet(ipGet("net"));
						requestInit = 0;
				}
				if (working==0){
						webwriteline("learning stopped early");
						pthread_exit(NULL);
				}
				if (isDigits(inited)==1 && randomizeDescent==1) randomizeTrainSet();
		}
		webwriteline("Done");
		working=0; websetmode(2);
		return NULL;
}

bool slowmotion = false;
#define SLOW(X) { if(slowmotion){ printf X; getchar(); }}
#define SLOW2(X){  printf X; getchar(); }

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/*  x- image index, ent- entopy, ep- epoch number                     */
/**********************************************************************/
int backProp(int x, float *ent, int ep){  // ANDY
		// BACK PROPAGATION WITH 1 TRAINING IMAGE
		int i = 0, j, k, r = 0, d=0, rot=0, hres=0, lres=1;
		float der=1.0, xs=0.0, ys=0.0, extra=0.0, sc=1.0, sum;
		int dc, a, a2, i2, j2, i3, j3, pmax, imax, jmax;
		int temp, temp2;
		time_t start,stop;
		// FORWARD PROP FIRST
		time(&start);
		int p = forwardProp(x,1,1,0);   // p is the predict value
		time(&stop); btime0_all += difftime(stop, start);
		if (p==-1) return -1; // GRADIENT EXPLODED
		// CORRECT PREDICTION?
		int y;
		if (isDigits(inited)==1) y = trainDigits[x];
		else y = trainColors[x];
		if (p==y) r=1;
		else SLOW(("SLOW: %d predict %d expected %d\n", x, p, y))
				// OUTPUT LAYER - CALCULATE ERRORS
				for (i=0;i<layerSizes[9];i++){
						errors[9][i] = an * (0 - layers[9][i]) * pow(_decay,ep);  /// wrong with -1e3: layers[9][i]
						//SLOW2(("errors[9][%d] = %f\n", i, errors[9][i]));
						if (i==y) {
								errors[9][i] = an * (1  - layers[9][i]) * pow(_decay,ep); /// correct with 1e3.
								if (layers[9][i]==0) return -1; // GRADIENT VANISHED
								*ent = -log(layers[9][i]);
								//SLOW2(("%d errors[9][%d] = %f\n", y, i, errors[9][i]));
						}
				}
		// HIDDEN LAYERS - CALCULATE ERRORS
		time(&start);
		for (k=8;k>10-numLayers;k--){
				if (layerType[k+1]==0) // FEEDS INTO FULLY CONNECTED
						for (i=0;i<layerSizes[k]*layerChan[k];i++){
								errors[k][i] = 0.0;
								if (dropOutRatio==0.0 || DOdense==0 || dropOut[k][i]==1){ // dropout
										if (activation==2) der = (layers[k][i]+1)*(1-layers[k][i]); //TanH derivative
										if (activation==0 || activation==2 || layers[k][i]>0){ //this is ReLU derivative
												temp = layerSizes[k]*layerChan[k]+1;
												for (j=0;j<layerSizes[k+1];j++)
														errors[k][i] += errors[k+1][j]*weights[k+1][j*temp+i]*der;  /// use backProp to enhance the correct path with higher delta_E (errors[k][i] +/- enhance/decrese)
										}
								}
						}
		}
		time(&stop); btime1_all += difftime(stop, start);

		// UPDATE WEIGHTS - GRADIENT DESCENT
		time(&start);
		int count = 0;
		for (k=11-numLayers;k<10;k++){
				if (layerType[k]==0){ // FULLY CONNECTED LAYER
						for (i=0;i<layerSizes[k];i++){
								temp = i*(layerSizes[k-1]*layerChan[k-1]+1);
								for (j=0;j<layerSizes[k-1]*layerChan[k-1]+1;j++)
										weights[k][temp+j] += errors[k][i]*layers[k-1][j];   // adjust the weights by Delta_E.. enhance the correct weights on the path and decrease the weights on others..
						}
				}
		}
		time(&stop); btime2_all += difftime(stop, start);
		btime_cnt++; 
		return r;
}

// https://stackoverflow.com/questions/16272384/parallel-sum-of-elements-in-a-large-array  -2013
/*auto J = interval(0, layerSizes[k-1]*layerChan[k-1]+1);
  std::for_each(std::par, J.begin(), J.end(), [=](int j) {
  }); */

void forwardProp_inner1(int k, int dp, int temp, float sum) {
		for (i=0;i<layerSizes[k];i++){
				if (dropOutRatio==0.0 || dp==0 || DOdense==0 || dropOut[k][i]==1){
						temp = i*(layerSizes[k-1]*layerChan[k-1]+1); //starting weights for each node
						sum = 0.0;
						for (j=0;j<layerSizes[k-1]*layerChan[k-1]+1;j++)
								sum += layers[k-1][j]*weights[k][temp+j]; // Sum(all weights)

						if (activation==0) layers[k][i] = sum;
						else if (activation==1) layers[k][i] = ReLU(sum); // activation(layers[k][i]) node
						else layers[k][i] = TanH(sum);
						//if (dropOutRatio>0.0 && dp==1) layers[k][i] = layers[k][i]  / (1-dropOutRatio);
						if (dropOutRatio>0.0 && dp==0 && DOdense==1) layers[k][i] = layers[k][i]  * (1-dropOutRatio);
				}
				else layers[k][i] = 0.0;
		}

}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
//        index,1(train)/0(valid),    1          0
int forwardProp(int x, int dp, int train, int lay){ // ANDY
		// FORWARD PROPAGATION WITH 1 IMAGE
		int i,j,k,imax,dc;
		int a, a2, i2, j2, i3, j3;
		float sum, esum, max, rnd, pmax;
		int temp, temp2;
		time_t start,stop;

		// INPUT LAYER (input_layer = 10-numLayers)  ==> assign image, accept input
		if (isDigits(inited)==1 && layerSizes[10-numLayers]==784){
				if (train==1) for (i=0;i<784;i++) layers[10-numLayers][i] = trainImages[x][i];   // input layer 28x28-> 784
				else for (i=0;i<784;i++) layers[10-numLayers][i] = testImages[x][i];
		}

		// input_layer+1 = 10-numLayers+1 = 11-numLayers, output_layer = 9	
		// HIDDEN LAYERS - k=input+1,k<output;k++ 
		time(&start);  
		for (k=11-numLayers;k<9;k++){
				if (lay!=0 && k>lay) return -1;
				// CALCULATE DROPOUT
				//if (dropOutRatio>0.0) // ALWAYS SET TO 1 TO BE SAFE
				for (i=0;i<layerSizes[k]*layerChan[k];i++) {
						dropOut[k][i] = 1;
						if (dropOutRatio>0.0 && dp==1) {
								rnd = (float)rand()/(float)RAND_MAX;
								if (rnd<dropOutRatio) dropOut[k][i] = 0;
						}
				}

				// 	
				if (layerType[k]==0) // FULLY CONNECTED LAYER
						forwardProp_inner1(k, dp, temp, sum);
		}
		time(&stop); 
		ftime_all += difftime(stop, start);
		ftime_cnt ++;
		//- so all nodes in hidden layers store the temporary/intermediate values which will be thrown away.
		// so the nodes are useless and the 'weights' are the 'memories' which are just bunch of 'trained' parameters to 'fit' the training set.
		// this model in logic is wrong. it is just a toy and it should be called 'deep fitting' not 'learning'. 


		// OUTPUT LAYER - SOFTMAX ACTIVATION
		esum = 0.0;
		for (i=0;i<layerSizes[9];i++){
				sum = 0.0;
				for (j=0;j<layerSizes[8]+1;j++)
						sum += layers[8][j]*weights[9][i*(layerSizes[8]+1)+j];
				layers[9][i] = exp(sum);  // softmax, mapping (1,2,3,4) -> (2.7, 7.4, 20.0, 54.6)
				if (layers[9][i]>1e30) return -1; //GRADIENTS EXPLODED
				esum += layers[9][i];     // softmax, mapping sum (10)  -> (95)
		}

		// SOFTMAX FUNCTION
		max = layers[9][0]; imax=0;
		for (i=0;i<layerSizes[9];i++){
				if (layers[9][i]>max){
						max = layers[9][i];
						imax = i;
				}
				layers[9][i] = layers[9][i] / esum; // softmax, mapping (10%,20%,30%,40%) -> ( 3%, 9%, 23%, 65% ), bump the max by 50% and compress the min by 30%
		}
		prob = layers[9][imax]; // ugly use of global variable :-(
		prob0 = layers[9][0];
		prob1 = layers[9][2];
		prob2 = layers[9][4];
		return imax;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
float ReLU(float x){
		if (x>0) return x;
		else return 0;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
float TanH(float x){
		return 2.0/(1.0+exp(-2*x))-1.0;
}


int main(int argc, char** argv)
{
		load_data();
		init_net();
		train_and_validate();
		while(1) sleep(5);
		return 0;
}
