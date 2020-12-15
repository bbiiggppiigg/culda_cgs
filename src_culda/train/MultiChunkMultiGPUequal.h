#include "hip/hip_runtime.h"
#ifndef _MultiChunkMultiGPUequal_H_
#define _MultiChunkMultiGPUequal_H_


void static MultiChunkMultiGPUequal(Document &doc, Vocabulary &vocab, Argument &arg,
                                    ModelPhi &modelPhi, ModelTheta &modelTheta)
{

    /* data preparation and transfer */
    gpuErr(hipPeekAtLastError());
    for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++){
        doc.docChunkVec[chunkId]->allocGPU(chunkId);
        doc.docChunkVec[chunkId]->toGPU();
    }
    printf("\n");
   
    /* modelphi */    
    modelPhi.InitData(doc); 
    printf("doc.numChunks:%d\n", doc.numChunks);
    for(int i = 0; i < doc.numChunks; i++)
        modelPhi.UpdatePhiGPU(doc, i);

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());
        
    for(int i = 1;i < arg.numGPUs; i++){
        modelPhi.MasterGPUCollect(i);
        modelPhi.MasterGPUReduce();
    }
    for(int i = 1;i < arg.numGPUs; i++) 
        modelPhi.MasterGPUDistribute(i);

    modelPhi.MasterGPUToCPU();
    modelPhi.validPhi(doc);
    modelPhi.UpdatePhiHead(arg.beta);

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    /* model theta */
    modelTheta.InitData(doc); //alloc GPU+CPU memory space.
    modelTheta.UpdateThetaGPU(doc);
    modelTheta.toCPU();
    //modelTheta.validTheta(doc);
    
    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());


    /* prepare the randstate, used for random sampling. */
    
    
    int randStateSize = 256;
    hiprandState *deviceRandState[MaxNumGPU];
    for(int i = 0;i < arg.numGPUs;i++){
        hipSetDevice(i);
        hipMalloc(&deviceRandState[i], sizeof(hiprandState)*randStateSize);
        hipLaunchKernelGGL(initRandState, dim3(randStateSize/256), dim3(256), 0, 0, deviceRandState[i]);
    }

    hipStream_t mainStream[MaxNumGPU];
    hipStream_t branStream[MaxNumGPU];
    pthreadArgTheta thetaArgs[MaxNumGPU];
    pthread_t threads[MaxNumGPU];
    for(int i = 0;i < arg.numGPUs;i++){
        hipSetDevice(i);    
        hipStreamCreate(&mainStream[i]);
        hipStreamCreate(&branStream[i]);

        thetaArgs[i].mainStream = mainStream[i];
        thetaArgs[i].branStream = branStream[i];
        thetaArgs[i].thetaPtr = &modelTheta;
        thetaArgs[i].id = i;
        thetaArgs[i].docPtr = &doc;
    }
    
    

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());

    //launch kernels
    
    struct timespec begin, end;
    double elapsed = 0, stamp = 0;

    printf("Launching Sampling Part ...\n");
    for(int ite = 0;ite < arg.iteration; ite++){
        
        //printf("Iteration %3d:", ite + 1);

        clock_gettime(CLOCK_MONOTONIC, &begin);

        
        for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++){
            
            hipSetDevice(chunkId);

            hipDeviceSynchronize();
            gpuErr(hipPeekAtLastError());

            //hipLaunchKernelGGL(LDAKernelTrain, dim3(doc.numWords), dim3(TrainBlockSize), 0, mainStream[chunkId], 
            hipLaunchKernelGGL(LDAKernelTrain, dim3(doc.docChunkVec[chunkId]->numSlots), dim3(TrainBlockSize), 0, mainStream[chunkId], 
                arg.k,
                arg.alpha,
                arg.beta,
                doc.numDocs,
                doc.numWords,
                doc.docChunkVec[chunkId]->chunkNumTokens,
                doc.docChunkVec[chunkId]->deviceWordIndices,
                doc.docChunkVec[chunkId]->deviceSlotIdToWordId,
                doc.docChunkVec[chunkId]->deviceSlotIndices,
                doc.docChunkVec[chunkId]->deviceWordTokens,
                doc.docChunkVec[chunkId]->deviceWordTopics,
                modelTheta.thetaChunkVec[chunkId]->deviceThetaA,
                modelTheta.thetaChunkVec[chunkId]->deviceThetaMaxIA,
                modelTheta.thetaChunkVec[chunkId]->deviceThetaCurIA,
                modelTheta.thetaChunkVec[chunkId]->deviceThetaJA,
                modelTheta.thetaChunkVec[chunkId]->docIdStart,
                modelPhi.phiChunkVec[chunkId]->devicePhiTopicWordShort,
                modelPhi.phiChunkVec[chunkId]->devicePhiTopic,
                modelPhi.phiChunkVec[chunkId]->devicePhiHead,
                deviceRandState[chunkId],
                randStateSize,
                chunkId,
                doc.docChunkVec[chunkId]->deviceWordPerplexity,
                doc.docChunkVec[chunkId]->deviceDocRevIndices
                );
        }
        for(int i = 0;i < arg.numGPUs; i++)
            modelPhi.UpdatePhiGPU(doc, i, mainStream[i]);
        

        double logLike = LDATrainPerplexity(doc, mainStream);
        //printf("log likelyhood :%.8f\n", logLike);
        
        for(int i = 0;i < arg.numGPUs;i++){
            pthread_create(&(threads[i]), 
                    NULL, 
                    UpdateThetaThread, 
                    (void*)(&(thetaArgs[i])));
            //pthread_join(threads[i], NULL);
        }

        
        for(int i = 1;i < arg.numGPUs; i++){ 
            hipStreamSynchronize(mainStream[i]);
            modelPhi.MasterGPUCollect(i, mainStream[0]);
            modelPhi.MasterGPUReduce(mainStream[0]);
        }

        hipStreamSynchronize(mainStream[0]);

        for(int i = 1;i < arg.numGPUs; i++) 
            modelPhi.MasterGPUDistribute(i, mainStream[i]);

        modelPhi.UpdatePhiHead(arg.beta, mainStream);
        
        for(int i = 0;i < arg.numGPUs;i++)
            pthread_join(threads[i], NULL);

        hipDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &end);
        stamp = end.tv_sec - begin.tv_sec;
        stamp += (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
        elapsed += stamp;

        printf("Iteration, %d,%.3f sec,%.3f sec, %.8f, %.3f M\n", ite+1,elapsed, stamp, logLike, doc.numTokens/stamp/1000000);

        if((ite + 1)%20 == 0) sleep(60);
    }
    
    /*
    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());
    
    
    for(int chunkId = 0; chunkId < doc.numChunks; chunkId ++)
        doc.docChunkVec[chunkId]->toCPU();
    printf("\n");
    modelTheta.toCPU();
    modelTheta.validTheta(doc);

    //modelPhi.MasterGPUToCPU();
    //modelPhi.validPhi(doc);
    
    for(int i = 0;i < arg.numChunks; i++)hipFree(deviceRandState[i]);
    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());
    */
}

#endif
