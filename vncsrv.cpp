#include <rfb/rfb.h>

int width=1600;
int height=900;
int bpp= 4;

#define WHITE 0xFFFFFFFF
// Little-Endian, x86,  should be CR.. counting from the righ edge
#define CRRED   0x000000FF
#define CRGREEN 0x0000FF00
#define CRBLUE  0x00FF0000
#define BLACK 0x00000000

#define COL(y,x) (y*maxx+x)*bpp
#define ROW(x,y) (x*mayy+y)*bpp

#define MODE1() {\
         buffer[COL(y,x)+0]=(x+y)*256/(maxx+maxy); /* red */ \
         buffer[COL(y,x)+1]=(x+y)*256/maxx; /* green */ \
         buffer[COL(y,x)+2]=(x+y)*256/maxy; /* blue */ \
}
         //buffer[COL(y,x)+3]=(x+y)%256; /* alpha? */ \


// 5D rendering 1024,1024,r,g,b. we have 1M points with 16M coloring. 
// we have snapshot of 1 frame of such system using 3MB memory. (1M*3bytes 24Mbit) but the whole potential space is 16T. (focus on every possible point with color)
// so we have 2D at sqr(1024) and 3d at cub(256). totally 16T combinations.
// for the 2K HD world.. (2048x1024).. it is 32T. if that is what we see and process.. then how does the computer could process and have the same knowledge and understanding? we must be filtering most of the information out and only keep what is important/focus one at a time(in speed of thoughts which is 50ms?) and accumulate those as 'knowledge and understanding'. the AI should do the same

// maybe one of the naive way to do it is to 
// 1. detect and keep tracking the change.
// 2. recoginze such change and form a concept.
// 3. connect the concepts together to form the logic/order.
// 4. use the logic/order to predict the unknown.
// 5. use the error to correct the logic/order(step3) and sometimes extend the concept(step2 - trigger discovery/invention/breakthrough).

static void initBuffer(unsigned char* buffer, int minx, int miny, int maxx, int maxy, unsigned int border)
{
  int x,y;
  for(y=miny;y<maxy;++y) {
    for(x=minx;x<maxx;++x) {
	  if(y==miny || x == minx ||  y == maxy-1 || x == maxx-1) { //boader
		 *(unsigned int*)&buffer[COL(y,x)]=border;
	  } else {
		 MODE1()
	  }
    }
  }
}

static int run_vncsrv(int argc, char**argv) {
  rfbScreenInfoPtr server=rfbGetScreen(&argc,argv,width,height,8,3,bpp);
  server->frameBuffer=(char*)malloc(width*height*bpp);
  initBuffer((unsigned char*)server->frameBuffer, 0, 0, width, height, CRBLUE);
  rfbInitServer(server);
  rfbRunEventLoop(server,-1,FALSE);
  return(0);
}

#ifdef TEST_VNCSRV
int main(int argc,char** argv)
{
	run_vncsrv(argc, argv);
	return 0;
}
#endif
