#ifndef _DOC_CHUNK_
#define _DOC_CHUNK_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <sstream>

#include <hip/hip_runtime_api.h>

#include "culda_argument.h"

using namespace std;

class DocChunk
{
public:
    int chunkId;
    int docIdStart;
    int docIdEnd;

    int numWorkers;
    int numDocs;
    int numWords;
    int numSlots;
    int numChunks;

    long long chunkNumTokens;
    int       chunkNumDocs;

    /* original input data */
    long long    *wordIndices;         // numberWords + 1
    int          *slotIdToWordId;      // numSlots
    long long    *slotIndices;         // numSlots*2
    int          *wordTokens;          // chunkNumTokens
    short        *wordTopics;          // chunkNumTokens
    double       *wordPerplexity;      // chunkNumTokens

    long long    *deviceWordIndices;      // numWords + 1
    int          *deviceSlotIdToWordId;   // numSlots
    long long    *deviceSlotIndices;      // numSlots*2
    int          *deviceWordTokens;       // chunkNumTokens
    short        *deviceWordTopics;       // chunkNumTokens
    double       *deviceWordPerplexity;   // chunkNumTokens

    double       *deviceWordPerplexityMid;

    /* reverse doc data */
    long long    *docRevIndices;       // numDocs + 1
    TokenIdxType *docRevIdx;           // chunkTokenSize

    long long    *deviceDocRevIndices; // numDocs + 1
    TokenIdxType *deviceDocRevIdx;     // chunkTokenSize 

    DocChunk();
    DocChunk(int argChunkId, 
             int argDocIdStart, 
             int argDocIdEnd, 
             int argNumDocs, 
             int argNumChunks);
    ~DocChunk()
    {
        
        if(wordIndices             != NULL)delete []wordIndices;
        if(slotIdToWordId          != NULL)delete []slotIdToWordId;
        if(slotIndices             != NULL)delete []slotIndices;
        if(wordTokens              != NULL)delete []wordTokens;
        if(wordTopics              != NULL)delete []wordTopics;
        if(wordPerplexity          != NULL)delete []wordPerplexity;
        
        if(deviceWordIndices       != NULL)hipFree(deviceWordIndices);
        if(deviceSlotIdToWordId    != NULL)hipFree(deviceSlotIdToWordId);
        if(deviceSlotIndices       != NULL)hipFree(deviceSlotIndices);
        if(deviceWordTokens        != NULL)hipFree(deviceWordTokens);
        if(deviceWordTopics        != NULL)hipFree(deviceWordTokens);
        if(deviceWordPerplexity    != NULL)hipFree(deviceWordPerplexity);
        if(deviceWordPerplexityMid != NULL)hipFree(deviceWordPerplexityMid);

        if(deviceDocRevIndices     != NULL)hipFree(deviceDocRevIndices);
        if(deviceDocRevIdx         != NULL)hipFree(deviceDocRevIdx);

    }

    void loadChunk(string, string, int*);
    void generateTopics(int k);

    void allocGPU(int);
    void toGPU();
    void toCPU();


};

#endif