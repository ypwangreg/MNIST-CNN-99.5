#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <math.h>

#include "pthpool.h"

static const size_t num_threads = 20;
static const size_t num_items   = 100;

int net = 4; // 784x1000x10
int INPUT=7; 
int OUTPUT=9;

float* layers[10]  = {0,};
float* weights[10] = {0,};
float* errors[10] = {0,};
int layersizes[10] = {0,};
// PREDEFINED NET ARCHITECTURES
char nets[8][10][20] =
          {{"","","","","","","","","",""},
           {"","","","","","","2","20","20","6"},
           {"","","","784","C5:6","P2","C5:16","P2","128","10"},
           {"","784","C3:10","C3:10","P2","C3:20","C3:20","P2","128","10"},
           {"","","","","","","","784","1000","10"},
           // debug nets below
           {"","","","","","784","C5:6","P2","50","10"},
           {"","","","","","","","196","100","10"},
           {"","","","","","16","C3:2","P2","2","2"}};

// data set
float (*trainImages)[784] = 0;
float (*trainImages2)[196] = 0;
int *trainDigits = 0;
int trainSizeI = 0, extraTrainSizeI = 1000;
int trainColumns = 0, trainSizeE = 0;
int *trainSet = 0; int trainSetSize = 0;
int *validSet = 0; int validSetSize = 0;

unsigned long fsize(const char* file) {
    /* returns file size */
    FILE * f = fopen(file, "r");
    fseek(f, 0, SEEK_END);
    unsigned long len = (unsigned long)ftell(f);
    fclose(f);
    return len;
}
void webwriteline(const char * line) {
	printf("%s\n", line);
}
#define D(x) printf("%s: %d\n", #x, x )

int load_train_data(int ct=0/*max samples*/, double testProp=0.24/*save 24% for validation*/, int sh=1/*remove header*/, float imgScale=255.0, float imgBias=0.0){
    char *data;
    // LOAD TRAINING DATA FROM FILE
    if (ct<=0) ct=1e6;
    int i, len = 0, lines=1, lines2=1;
    float rnd;
    // READ IN TRAIN.CSV
    char buffer[1000000];
    char name[80] = "train.csv";
    //strcpy(name,spGet("dataFile"));
    //if (access(name,F_OK)!=0) sprintf(name,"../%s",spGet("dataFile"));
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
	//D(testProp); // split % between train and validate
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

void init_net() {
	//init layersizes
	for (int i =0; i < 10; i++) {
		layersizes[i] = atoi(nets[net][i]);
		printf("%d %d\n", i, layersizes[i]);	
	}
	// malloc layers/weights/errors
	for (int i =0; i< 10; i++) {
		layers[i] = (float*)malloc((layersizes[i]+1) * sizeof(float));
		errors[i] = (float*)malloc((layersizes[i]+0) * sizeof(float));
		if(i > 0) 
       weights[i] = (float*)malloc(layersizes[i]*(layersizes[i-1]+1)*sizeof(float));
	}
	// init layers/weights
	float scale;
	for (int i=0; i<10; i++ )layers[i][layersizes[i]] = 1.0; //bias
	for (int j=0; j<10; j++) {
		scale = 1.0;	
		if(layersizes[j-1]) {
			// XAVIER INITIALIZATION SQRT(6/(N_in+N_out))
			scale = 2.0 * sqrt(6.0/(layersizes[j-1]+layersizes[j]));
		}
		if(layersizes[j]) {
			// init weights
			for (int i=0; i<layersizes[j]*(layersizes[j-1]+1); i++) 
				weights[j][i] = scale*( (float)rand()/(float)RAND_MAX -0.5);
		}
	}	
		
}

float ReLU(float x){
    if (x>0) return x;
    else return 0;
}
void forward(int x=0) {
	for (int i=0; i< 784; i++) layers[INPUT][i] = trainImages[x][i];
	for (int k=INPUT+1; k < OUTPUT; k++) {        // for each HIDDEN layer k,
		for (int i = 0; i < layersizes[k]; i++) { // for each HIDDEN layer node i, do
			int psz = layersizes[k-1]+1; // get previous layer size, psz
			int t = i*psz;    // get the start position of weights of w[k][t+0]
			int sum = 0.0;
			for (int j = 0; j<psz; j++) {
				sum += layers[k-1][j]*weights[k][t+j];  // sum (layer[k-1][j]*w[k][t+j]
			}
			layers[k][i] = ReLU(sum);
		}
	}

}

void test_net(int argc, char** argv) {
	load_train_data();
	init_net();
	for(int i = 0; i < 10000; i++) forward(i);
}

void worker1(void *arg)
{
    int *val = (int*)arg;
    int  old = *val;

    *val += 1000;
    printf("tid=%p, old=%d, val=%d\n", (void*)pthread_self(), old, *val);

    if (*val%2)
        usleep(100000);
}
void test1(int argc, char**argv) {
    tpool_t *tm;
    int     *vals;
    size_t   i;

    tm   = tpool_create(num_threads);
    vals = (int*)calloc(num_items, sizeof(*vals));

    for (i=0; i<num_items; i++) {
        vals[i] = i;
        tpool_add_work(tm, worker1, vals+i);
    }

    tpool_wait(tm);

    for (i=0; i<num_items; i++) {
        printf("%d\n", vals[i]);
    }

    free(vals);
    tpool_destroy(tm);
}

int main(int argc, char **argv)
{
	//test1(argc, argv);
	test_net(argc, argv);
    return 0;
}
