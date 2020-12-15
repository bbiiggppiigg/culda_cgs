#include "hip/hip_runtime.h"
#ifndef _SingleChunkSingleGPU_H_
#define _SingleChunkSingleGPU_H_


void static SingleChunkSingleGPU(Document &doc, Vocabulary &vocab, Argument &arg,
                                 ModelPhi &modelPhi, ModelTheta &modelTheta)
{

    /* data preparation and transfer */

    printf("Call SingleChunkSingleGPU() ...\n");
    

    printf("alloc gpu for doc ...\n");
    doc.docChunkVec[0]->allocGPU(0);
    printf("to gpu for doc ...\n");
    doc.docChunkVec[0]->toGPU();

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    /* model phi */
    printf("Prepare model phi ...\n");
    modelPhi.InitData(doc);
    modelPhi.UpdatePhiGPU(doc, 0);
    modelPhi.UpdatePhiHead(arg.beta);
    //modelPhi.MasterGPUToCPU();
    //modelPhi.validPhi(doc);


    /* model theta */
    printf("Prepare model theta ...\n");
    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    modelTheta.InitData(doc);

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    modelTheta.UpdateThetaGPU(doc);

    //exit(0);

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    //modelTheta.toCPU();
    //modelTheta.validTheta(doc);

    //exit(0);

    
    /* prepare the randstate */
    int randStateSize = 256*20;
    hiprandState *deviceRandState[MaxNumGPU];
    hipMalloc(&deviceRandState[0], sizeof(hiprandState)*randStateSize);
    hipLaunchKernelGGL(initRandState, dim3(randStateSize/256), dim3(256), 0, 0, deviceRandState[0]);
    
    hipStream_t extraStream;
    hipStreamCreate(&extraStream);

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    struct timespec begin, end;
    double elapsed = 0, stamp = 0;

    //launch train kernels
    for(int ite = 0;ite < arg.iteration; ite++)
    {
        clock_gettime(CLOCK_MONOTONIC, &begin);

        //numBlocks = 100;
        hipLaunchKernelGGL(LDAKernelTrain, dim3(doc.docChunkVec[0]->numSlots), dim3(TrainBlockSize), 0, 0, 
            arg.k,
            arg.alpha,
            arg.beta,
            doc.numDocs,
            doc.numWords,
            doc.docChunkVec[0]->chunkNumTokens,
            doc.docChunkVec[0]->deviceWordIndices,
            doc.docChunkVec[0]->deviceSlotIdToWordId,
            doc.docChunkVec[0]->deviceSlotIndices,
            doc.docChunkVec[0]->deviceWordTokens,
            doc.docChunkVec[0]->deviceWordTopics,
            modelTheta.thetaChunkVec[0]->deviceThetaA,
            modelTheta.thetaChunkVec[0]->deviceThetaMaxIA,
            modelTheta.thetaChunkVec[0]->deviceThetaCurIA,
            modelTheta.thetaChunkVec[0]->deviceThetaJA,
            modelTheta.thetaChunkVec[0]->docIdStart,
            modelPhi.phiChunkVec[0]->devicePhiTopicWordShort,
            modelPhi.phiChunkVec[0]->devicePhiTopic,
            modelPhi.phiChunkVec[0]->devicePhiHead,
            deviceRandState[0],
            randStateSize, //arg.numWorkers,
            0,
            doc.docChunkVec[0]->deviceWordPerplexity,
            doc.docChunkVec[0]->deviceDocRevIndices
            );

        //hipDeviceSynchronize();
        //gpuErr(hipPeekAtLastError());

        double logLike = LDATrainPerplexity(doc);
        //hipDeviceSynchronize();
        //gpuErr(hipPeekAtLastError());

        //doc.docChunkVec[0]->toCPU();

        modelPhi.UpdatePhiGPU(doc, 0);
        modelPhi.UpdatePhiHead(arg.beta);
        //modelPhi.MasterGPUToCPU();
        //modelPhi.validPhi(doc);

        modelTheta.UpdateThetaGPU(doc);
        //modelTheta.toCPU();
        //modelTheta.validTheta(doc);

        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end);
        stamp = end.tv_sec - begin.tv_sec;
        stamp += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        elapsed += stamp;

        printf("Iteration %3d: %6.2f sec, %3.2f sec, logLikelyhood = %.8f, %5.3f M\n", ite+1,elapsed, stamp, logLike,  doc.numTokens/stamp/1000000);
        hipDeviceSynchronize();
        gpuErr(hipPeekAtLastError());

//        if((ite + 1)%30 == 0)sleep(120);

    }

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++)
        doc.docChunkVec[chunkId]->toCPU();
    printf("\n");

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    modelTheta.toCPU();
    //modelTheta.validTheta(doc);

    modelPhi.MasterGPUToCPU();
    //hipDeviceSynchronize();
    //modelPhi.validPhi(doc);

    

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());
    //modelPhi.savePhi("phi.data");
}


#endif
