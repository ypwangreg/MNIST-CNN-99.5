//https://stackoverflow.com/questions/32657609/pinning-a-thread-to-a-core-in-a-cpuset-through-c
   #include <pthread.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <errno.h>

   //ANDY: I have 20-cores/i9-10900F/2020, based on Moore's Law, I will get 1K cores in 2030(1.5y*6). 9 years, 2^5/64.
   //      but I know it is a dream, 2011, Sandy-Bridge. i7-3930K 12-cores. So the spirit is long gone.
   //      by comparision, Nvidia    2011,  GTX 590(40nm,3B), 512core/DDR5-1.5G/FP32-1T. 2020 GTX 3070(8nm,17B), 5888cores/DDR6-8G/FP32-20T.
   //      Ref: https://www.techpowerup.com/gpu-specs/geforce-gtx-590.c281 https://www.techpowerup.com/gpu-specs/geforce-rtx-3070.c3674
   //
   const int cpucore= 1024; //1024 cores.. 
   #define handle_error_en(en, msg) \
           do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)


   int
   main(int argc, char *argv[])
   {
       int s, j;
       cpu_set_t cpuset;
       pthread_t thread;

       thread = pthread_self();

       /* Set affinity mask to include CPUs 0 to 7 */

       CPU_ZERO(&cpuset);
       for (j = 0; j < cpucore; j++)
           CPU_SET(j, &cpuset);

       s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
       if (s != 0)
           handle_error_en(s, "pthread_setaffinity_np");

       /* Check the actual affinity mask assigned to the thread */

       s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
       if (s != 0)
           handle_error_en(s, "pthread_getaffinity_np");

       printf("Set returned by pthread_getaffinity_np() contained:\n");
       for (j = 0; j < CPU_SETSIZE; j++)
           if (CPU_ISSET(j, &cpuset))
               printf("    CPU %d\n", j);

       exit(EXIT_SUCCESS);
   }
