#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <cstring>      // std::memset
#include <fstream>

#include <hip/hip_runtime_api.h>
#include "model_phi_gpu.h"
#include "vocab.h"



ModelPhiGPU::ModelPhiGPU():
                k(0),
                numGPUs(1),
                GPUid(0),
                numDocs(0),
                numWords(0),
                devicePhiTopicWordShort(NULL),
                devicePhiTopicWordSub(NULL),
                devicePhiTopic(NULL),
                devicePhiHead(NULL),
                devicePhiTopicWordShortCopy(NULL),
                devicePhiTopicCopy(NULL)
{
}

ModelPhiGPU::ModelPhiGPU(
                int argk, 
                int argNumGPUs,
                int argid, 
                int argdoc, 
                int argword):
                k(argk),
                numGPUs(argNumGPUs),
                GPUid(argid),
                numDocs(argdoc),
                numWords(argword),
                devicePhiTopicWordShort(NULL),
                devicePhiTopicWordSub(NULL),
                devicePhiTopic(NULL),
                devicePhiHead(NULL),
                devicePhiTopicWordShortCopy(NULL),
                devicePhiTopicCopy(NULL)
{
}

void ModelPhiGPU::allocGPU()
{
    hipSetDevice(GPUid);
    hipMalloc((void**)&devicePhiTopicWordShort, sizeof(PHITYPE)*k*numWords);
    hipMalloc((void**)&devicePhiTopicWordSub,   sizeof(int)*k*UpdateNumWorkers);
    hipMalloc((void**)&devicePhiTopic,          sizeof(int)*k);
    hipMalloc((void**)&devicePhiHead,           sizeof(half)*k*numWords);

    if(GPUid == 0 && numGPUs > 1){
        hipMalloc((void**)&devicePhiTopicWordShortCopy, sizeof(PHITYPE)*k*numWords);
        hipMalloc((void**)&devicePhiTopicCopy,          sizeof(int)*k);
    }
    long long totalByte = sizeof(PHITYPE)*k*numWords + 
                          sizeof(int)*k*UpdateNumWorkers + 
                          sizeof(int)*k +
                          sizeof(half)*k*numWords;

    printf("phi sizeof:%.3f GB\n", totalByte/(1024.0*1024.0*1024.0));

    hipDeviceSynchronize();
    gpuErr(hipPeekAtLastError());
}

void ModelPhiGPU::UpdatePhiGPU(Document &doc, int chunkId, hipStream_t stream)
{
    hipSetDevice(GPUid);
    hipMemsetAsync(devicePhiTopic,          0, k*sizeof(int),            stream);

    LDAUpdatePhiAPI(
        k,
        numWords,
        doc.docChunkVec[chunkId]->deviceWordIndices,
        doc.docChunkVec[chunkId]->deviceWordTopics,
        devicePhiTopicWordShort,
        devicePhiTopicWordSub,
        devicePhiTopic,
        stream
    );    
}

void ModelPhiGPU::UpdatePhiHead(float beta, hipStream_t stream)
{

    hipSetDevice(GPUid);
    //printf("ModelPhiGPU::UpdatePhiHead() ... id:%d\n", GPUid);
    LDAComputePhiHeadAPI(
        k, 
        beta, 
        numWords,
        devicePhiTopicWordShort, 
        devicePhiTopic,
        devicePhiHead,
        stream);
}

void ModelPhiGPU::clearPtr()
{
    
    if(devicePhiTopicWordShort != NULL) hipFree(devicePhiTopicWordShort);
    if(devicePhiTopicWordSub   != NULL) hipFree(devicePhiTopicWordSub);
    if(devicePhiTopic          != NULL) hipFree(devicePhiTopic);
    if(devicePhiHead           != NULL) hipFree(devicePhiHead);

    devicePhiTopicWordSub   = NULL;
    devicePhiTopicWordShort = NULL;
    devicePhiTopic          = NULL;
    devicePhiHead           = NULL;
}



