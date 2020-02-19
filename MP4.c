#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_SIZE 3
#define SHARE_SIZE (TILE_SIZE + 2 * MASK_RADIUS)

//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  __shared__ float Nc[SHARE_SIZE][SHARE_SIZE][SHARE_SIZE];
  
  float Pvalue = 0.0f;
  
  int outX = bx * TILE_SIZE + tx;
  int outY = by * TILE_SIZE + ty;
  int outZ = bz * TILE_SIZE + tz;
  
  int startX = outX - MASK_RADIUS;
  int startY = outY - MASK_RADIUS;
  int startZ = outZ - MASK_RADIUS;
  
  if ((startX >= 0 && startX < x_size) && (startY >= 0 && startY < y_size) && (startZ >= 0 && startZ < z_size)) {
    Nc[tz][ty][tx] = input[startX + startY * x_size + startZ * x_size * y_size];
  }
  else {
    Nc[tz][ty][tx] = 0;
  }
  __syncthreads();
  
  if (tx < TILE_SIZE && ty < TILE_SIZE && tz < TILE_SIZE) {
    for (int i = 0; i < MASK_WIDTH; ++i) {
      for (int j = 0; j < MASK_WIDTH; ++j) {
        for (int k = 0; k < MASK_WIDTH; ++k) {
          Pvalue += Mc[k][j][i] * Nc[tz + k][ty + j][tx + i];
        }
      }
    }
    
    if (outX < x_size && outY < y_size && outZ < z_size) {
      output[outX + outY * x_size + outZ * x_size * y_size] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void**)&deviceOutput, (inputLength - 3) * sizeof(float));
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, (hostInput + 3), (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size * 1.0 / TILE_SIZE), ceil(y_size * 1.0 / TILE_SIZE), ceil(z_size * 1.0 / TILE_SIZE));
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy((hostOutput + 3), deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
