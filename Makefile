NVCC = /usr/local/cuda-12.0/bin/nvcc
TARGET = flash_attention

all: $(TARGET)

$(TARGET): main.cpp flash_attention.cu
	$(NVCC) main.cpp flash_attention.cu -o $(TARGET)

clean:
	rm -f $(TARGET)