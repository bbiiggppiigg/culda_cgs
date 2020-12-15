#go!



CXX = hipcc
INCLUDE_PATH = /opt/rocm/include/hiprand -I /opt/rocm/rocrand
LIB_PATH = /opt/rocm/
OBJECTIVE = ./obj
CXX_FLAG = -I /opt/rocm/include/hiprand -I /opt/rocm/rocrand/include
#-maxrregcount=32

#-maxrregcount=32

DEBUG_FLAG = 
#DEBUG_FLAG = -G -g
#DEBUG_FLAG = -DDEBUGPrint

culda:$(OBJECTIVE)/main.o \
	  $(OBJECTIVE)/lda_train_kernel.o \
	  $(OBJECTIVE)/lda_train.o \
	  $(OBJECTIVE)/lda_theta_kernel.o \
	  $(OBJECTIVE)/lda_phi_kernel.o \
	  $(OBJECTIVE)/model_theta.o \
	  $(OBJECTIVE)/model_theta_chunk.o \
	  $(OBJECTIVE)/model_phi_gpu.o \
	  $(OBJECTIVE)/model_phi.o \
	  $(OBJECTIVE)/vocab.o \
	  $(OBJECTIVE)/doc.o \
	  $(OBJECTIVE)/doc_chunk.o
	$(CXX) $(OBJECTIVE)/main.o \
	       $(OBJECTIVE)/model_theta.o \
	       $(OBJECTIVE)/model_theta_chunk.o \
	       $(OBJECTIVE)/model_phi.o \
	       $(OBJECTIVE)/model_phi_gpu.o \
	       $(OBJECTIVE)/vocab.o \
	       $(OBJECTIVE)/doc.o \
	       $(OBJECTIVE)/doc_chunk.o \
	       $(OBJECTIVE)/lda_train_kernel.o \
	       $(OBJECTIVE)/lda_train.o \
	       $(OBJECTIVE)/lda_theta_kernel.o \
	       $(OBJECTIVE)/lda_phi_kernel.o \
	       -o culda \
	       -L $(LIB_PATH)/lib \
	       $(CXX_FLAG) $(DEBUG_FLAG)


$(OBJECTIVE)/main.o:\
	main.cpp \
	./model/model_theta.h \
	./model/vocab.h \
	./model/doc.h \
	./model/doc_chunk.h \
	./model/culda_argument.h \
	./train/lda_train.h 
	$(CXX) -c main.cpp \
	       -o $(OBJECTIVE)/main.o \
	       -I $(INCLUDE_PATH)/include \
	       -L $(LIB_PATH)/lib \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/doc.o:\
	./model/doc.cpp \
	./model/doc.h \
	./model/doc_chunk.h
	$(CXX) -c ./model/doc.cpp \
	       -o $(OBJECTIVE)/doc.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/doc_chunk.o:\
	./model/doc_chunk.cpp \
	./model/doc_chunk.h \
	./model/culda_argument.h
	$(CXX) -c ./model/doc_chunk.cpp \
	       -o $(OBJECTIVE)/doc_chunk.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/vocab.o:\
	./model/vocab.cpp \
	./model/vocab.h
	$(CXX) -c ./model/vocab.cpp \
	       -o $(OBJECTIVE)/vocab.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/lda_theta_kernel.o:\
	./kernel/lda_theta_kernel.hip \
	./kernel/lda_theta_kernel.h \
	./kernel/lda_train_kernel.h \
	./model/culda_argument.h
	$(CXX) -c ./kernel/lda_theta_kernel.hip \
	       -o $(OBJECTIVE)/lda_theta_kernel.o \
	       $(CXX_FLAG) -Xptxas -v $(DEBUG_FLAG)

$(OBJECTIVE)/lda_train_kernel.o:\
	./kernel/lda_train_kernel.hip \
	./kernel/lda_train_kernel.h \
	./model/culda_argument.h
	$(CXX) -c ./kernel/lda_train_kernel.hip \
	       -o $(OBJECTIVE)/lda_train_kernel.o \
	       -I $(INCLUDE_PATH)/include \
	       $(CXX_FLAG) -Xptxas -v $(DEBUG_FLAG)

$(OBJECTIVE)/lda_train.o:\
	./train/lda_train.hip \
	./train/lda_train.h \
	./train/MultiChunkMultiGPUequal.h \
	./train/SingleChunkSingleGPU.h \
	./kernel/lda_train_kernel.h \
	./model/doc.h \
	./model/doc_chunk.h \
	./model/vocab.h \
	./model/model_theta.h \
	./model/model_theta_chunk.h \
	./model/model_phi.h \
	./model/model_phi_gpu.h\
	./model/culda_argument.h
	$(CXX) -c ./train/lda_train.hip \
	       -o $(OBJECTIVE)/lda_train.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/lda_phi_kernel.o:\
	./kernel/lda_phi_kernel.hip \
	./kernel/lda_phi_kernel.h \
	./kernel/lda_train_kernel.h \
	./model/culda_argument.h
	$(CXX) -c ./kernel/lda_phi_kernel.hip \
	       -o $(OBJECTIVE)/lda_phi_kernel.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/model_phi_gpu.o:\
	./model/model_phi_gpu.cpp \
	./model/model_phi_gpu.h \
	./model/doc.h \
	./model/doc_chunk.h \
	./model/vocab.h \
	./model/culda_argument.h \
	./kernel/lda_train_kernel.h
	$(CXX) -c ./model/model_phi_gpu.cpp \
	       -o $(OBJECTIVE)/model_phi_gpu.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/model_phi.o:\
	./model/model_phi.cpp \
	./model/model_phi.h \
	./model/doc.h \
	./model/doc_chunk.h \
	./model/vocab.h \
	./model/culda_argument.h \
	./model/model_phi_gpu.h \
	./kernel/lda_train_kernel.h
	$(CXX) -c ./model/model_phi.cpp \
	       -o $(OBJECTIVE)/model_phi.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/model_theta_chunk.o:\
	./model/model_theta_chunk.cpp \
	./model/model_theta_chunk.h \
	./model/doc.h \
	./model/doc_chunk.h \
	./model/vocab.h \
	./model/culda_argument.h \
	./kernel/lda_train_kernel.h 
	$(CXX) -c ./model/model_theta_chunk.cpp \
	       -o $(OBJECTIVE)/model_theta_chunk.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)

$(OBJECTIVE)/model_theta.o:\
	./model/model_theta.cpp \
	./model/model_theta.h \
	./model/doc.h \
	./model/doc_chunk.h \
	./model/vocab.h \
	./model/culda_argument.h \
	./model/model_theta_chunk.h \
	./model/culda_argument.h \
	./kernel/lda_train_kernel.h
	$(CXX) -c ./model/model_theta.cpp \
	       -o $(OBJECTIVE)/model_theta.o \
	       $(CXX_FLAG) $(DEBUG_FLAG)


clean:
	rm -f culda
	rm -f *.o
	rm -f obj/*
