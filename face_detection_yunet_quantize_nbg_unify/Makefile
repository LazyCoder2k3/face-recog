CC=$(CROSS_COMPILE)gcc
CXX=$(CROSS_COMPILE)g++
DEBUG=0
#VIVANTE_SDK_DIR=../VeriSilicon/VivanteIDE5.4.0/cmdtools/vsimulator
OPENCV_DIR=opencv4.2
OPENCV_INCLUDE=$(OPENCV_DIR)/include
OPENCV_LIB=$(OPENCV_DIR)/lib


INCLUDES=-I. -I$(VIVANTE_SDK_DIR)/include/ \
 -I$(VIVANTE_SDK_DIR)/include/CL \
 -I$(VIVANTE_SDK_DIR)/include/VX \
 -I$(VIVANTE_SDK_DIR)/include/ovxlib \
 -I$(VIVANTE_SDK_DIR)/include/jpeg

INCLUDES += -I$(OPENCV_INCLUDE)

CFLAGS=-Wall -std=c++0x $(INCLUDES) -D__linux__ -DLINUX -fpermissive
ifeq (1,$(DEBUG))
CFLAGS+=-g
LFLAGS+=-g
else
CFLAGS+=-O3
LFLAGS+=-O3
endif
# LFLAGS += -Wl,-rpath-link=$(VIVANTE_SDK_DIR)/drivers
LFLAGS+=-Wl,-rpath-link=$(OPENCV_LIB)
LIBS += -lstdc++ -pthread
LIBS += -L$(VIVANTE_SDK_DIR)/drivers \
 -lOpenVX -lOpenVXU -lCLC -lVSC -lGAL -lovxlib -lArchModelSw -lNNArchPerf -ljpeg -lm
# LIBS += -L$(VIVANTE_SDK_DIR)/drivers \
# 	-lOpenVX -lOpenVXU -lCLC -lVSC -lGAL -lovxlib -ljpeg -lm
File = $(VIVANTE_SDK_DIR)/driver/libjpeg.a
ifeq ($(File),$(wildcard $(File)))
LIBS += $(File)
endif

LIBS += -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_dnn -lopencv_objdetect -lopencv_imgcodecs
LIBS += -L$(OPENCV_LIB)

SRCS=${wildcard *.c}
SRCS+=${wildcard *.cpp}
BIN=main
OBJS=$(addsuffix .o, $(basename $(SRCS)))

.SUFFIXES: .cpp .c

.cpp.o:
	$(CC) $(CFLAGS) -c $<

.cpp:
	$(CXX) $(CFLAGS) $< -o $@ -lm

.c.o:
	$(CC) $(CFLAGS) -c $<

.c:
	$(CC) $(CFLAGS) $< -o $@ -lm

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(LFLAGS) $(EXTRALFLAGS) $(OBJS) $(LIBS) -o $@

clean:
	rm -rf *.o
	rm -rf $(BIN)
	rm -rf *~

