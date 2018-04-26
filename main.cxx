#include "gpu_plasma.h"
#include <stdlib.h>
#include "mpi_shortcut.h"
//TODO: gpu cell in the global array at copy from there appears to be not initialized

int main(int argc,char*argv[])
{
   GPUPlasma<GPUCell> *plasma;
  // Cell<Particle> *c;
   
  // InitMPI(argc,argv);

      size_t sizeP;

      printf("oarticle size %d %d \n",sizeof(Particle),sizeof(Particle)/sizeof(double));
      cudaDeviceGetLimit(&sizeP,cudaLimitPrintfFifoSize);

      printf("printf default limit %d \n",sizeP/1024/1024);

      sizeP *= 10;
      sizeP *= 10;
      sizeP *= 10;
      sizeP *= 10;
      cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sizeP);

      cudaDeviceGetLimit(&sizeP,cudaLimitPrintfFifoSize);



      printf("printf limit set to %d \n",sizeP/1024/1024);

   
      int err = cudaSetDevice(0);
   
      printf("err %d \n",err);

   //plasma = new GPUPlasma<GPUCell>(100,4,4,1.2566,0.05,0.05,1.0,100,1.0,0.001);
   plasma = new GPUPlasma<GPUCell>(100,4,4,1.1424,0.05,0.05,1.0,2000,1.0,0.001);
 
   plasma->Initialize();


   //double3 x,x1,v;
   //int3 i1;
   //int i11,l1,k1;
  // double m,tau,Lx,Ly,Lz;
 //  int Nx =100,Ny = 4,Nz = 4;
   
   
//   Lx = 1.2566;
//   Ly = 0.05;
//   Lz = 0.05;
  
   
/*   x.x = 0.736684309427232;
   x.y = 0.57112790596731056;
   x.z = 6.13108106486108528e-003;
   x1.x = 0.737483386691278;
   x1.y = 0.57113309163301351;
   x1.z = 6.11825940675901278e-003;
   i11 = 7;
   l1 = 11;
   k1 = 1; */
   
/*   i11 = 3;
   l1  = 10;
   k1  = 9;

   x.x = 0.28557456840145212;
   x.y = 0.49365342158073622;
   x.z = 0.46697638877236813;
   x1.x = 0.28637595592617204;
   x1.y = 0.49366399547068435;
   x1.z = 0.46698066504950825; */

/*   i11 = 8;
   l1  = 4;
   k1  = 2;

   x.x = 0.76160586978632705;
   x.y = 0.1428094216032761;
   x.z = 5.36496770865793129e-2;
   x1.x = 0.76240212575208588;
   x1.y = 0.14279803596608148;
   x1.z = 5.36569248825250547e-2;
*/

  /*i11 = 5;
   l1  = 8;
   k1  = 6;

   x.x = 0.51364786407779905;
   x.y = 0.42839953930823238;
   x.z = 0.29162965663951163;
   x1.x = 0.51444760742020168;
   x1.y = 0.42840177167379256;
   x1.z = 0.29163288220586198;*/
/*
   i11 = 2;
   l1  = 10;
   k1  = 5;

   x.x = 0.17131769999556001;
   x.y = 0.49914072160623163;
   x.z = 0.19993118164018731;
   x1.x = 0.17212585470024191;
   x1.y = 0.49913591048408684;
   x1.z = 0.19991771382289567;   
   
   
   i1.x = i11;
   i1.y = l1;
   i1.z = k1;
   
   m = -0.2;
   tau = 0.001;
   
   v.x = 0.999;
   v.y = 0.0;
   v.z = 0.0;
*/   
//    c = new Cell(i11,l1,k1,Lx,Ly,Lz,Nx,Ny,Nz,tau);
   
//   c->CurrentToMesh(x,x1,v,i1,m,tau);;

   double t = plasma->compareCPUtoGPU();
   printf("----------------------------------------------------------- plasma check before move %.5f\n",t);
   size_t m_free,m_total;

   cudaMemGetInfo(&m_free,&m_total);
//#ifdef MEMORY_PRINTS
//   printf("GPU memory total %d free %d\n",m_total/1024/1024,m_free/1024/1024);
//#endif
   struct sysinfo info;


   for(int nt = START_STEP_NUMBER;nt <= TOTAL_STEPS;nt++)
   {
	   cudaMemGetInfo(&m_free,&m_total);
	   sysinfo(&info);
#ifdef MEMORY_PRINTS
       printf("before Step  %10d CPU memory free %10u GPU memory total %10d free %10d\n",
    		   nt,info.freeram/1024/1024,m_total/1024/1024,m_free/1024/1024);
#endif

       plasma->Step(nt);

       cudaMemGetInfo(&m_free,&m_total);
       sysinfo(&info);
#ifdef MEMORY_PRINTS
       printf("after  Step  %10d CPU memory free %10u GPU memory total %10d free %10d\n",
    		   nt,info.freeram/1024/1024/1024,m_total/1024/1024/1024,m_free/1024/1024/1024);
#endif
//#ifdef MEMORY_PRINTS
//       printf("GPU memory total %d free %d\n",m_total/1024/1024,m_free/1024/1024);
//       puts("============================================================================================");
//#endif
   }
 //  plasma->Step(6);
//   plasma->Step(7);
//   plasma->Step(8);
//   plasma->Step(9);
//   plasma->Step(10);

   t = plasma->compareCPUtoGPU();
   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ plasma check after move %.5f\n",t);

   delete plasma;
   
 //  CloseMPI();

   return 0;
}
