#ifndef _MODEL_PHI_GPU_H_
#define _MODEL_PHI_GPU_H_

#include <vector>
#include <string>
#include <queue>
#include <set>

#include <hip/hip_runtime_api.h>
#include "culda_argument.h"
#include "vocab.h"
#include "doc.h"

#include "../kernel/lda_train_kernel.h"
#include "../kernel/lda_phi_kernel.h"

class ModelPhiGPU
{
public:
    int k;
    int numGPUs;
    int GPUid;
    int numDocs;
    int numWords;


    PHITYPE *devicePhiTopicWordShort;
    int     *devicePhiTopicWordSub;
    int     *devicePhiTopic;
    half    *devicePhiHead;

    PHITYPE *devicePhiTopicWordShortCopy;
    int     *devicePhiTopicCopy;

    ModelPhiGPU();
    ModelPhiGPU(int, int, int, int, int);
    ~ModelPhiGPU(){clearPtr();}

    void allocGPU();
    void UpdatePhiGPU(Document &, int, hipStream_t stream=0);
    void UpdatePhiHead(float, hipStream_t stream=0);

    void clearPtr();
};

#endif