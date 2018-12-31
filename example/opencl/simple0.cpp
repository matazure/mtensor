// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <array>

namespace chrono = std::chrono;

// OpenCL includes
#include <CL/cl.h>

// OpenCL kernel to perform an element-wise
// add of two arrays
const char* programSource =
"__kernel                                            \n"
"void vecadd(__global int *A, \n"
"            __global int *B, \n"
"            __global int *C)                        \n"
"{ \n"
"                                                    \n"
"   // Get the work-item¡¯s unique ID                 \n"
"   int idx = get_global_id(0);                      \n"
"                                                    \n"
"   // Add the corresponding locations of            \n"
"   // 'A' and 'B', and store the result in 'C'.     \n"
"   C[idx] = A[idx] + B[idx];                        \n"
" }                                                  \n"
;

const char* programMatrixMul =
"void set(__global float *p_data, int width, int height, int i, int j, float v) {	\n"
"	p_data[j * width + i] = v;												\n"
"}																			\n"
"__kernel void myGEMM2(const int M, const int N, const int K,				\n"
"	const __global float* A,												\n"
"	const __global float* B,												\n"
"	__global float* C) {													\n"
"																			\n"
"	// Thread identifiers													\n"
"	const int row = get_local_id(0); // Local row ID (max: 16)				\n"
"	const int col = get_local_id(1); // Local col ID (max: 16)				\n"
"	const int globalRow = 16*get_group_id(0) + row; // Row ID of C (0..M)	\n"
"	const int globalCol = 16*get_group_id(1) + col; // Col ID of C (0..N)	\n"
"																			\n"
"	// Local memory to fit a tile of 16*16 elements of A and B				\n"
"	__local float Asub[16][16];												\n"
"	__local float Bsub[16][16];												\n"
"																			\n"
"	// Initialise the accumulation register									\n"
"	float acc = 0.0f;														\n"
"																			\n"
"	// Loop over all tiles													\n"
"	const int numTiles = K / 16;											\n"
"	for (int t = 0; t<numTiles; t++) {										\n"
"																			\n"
"		// Load one tile of A and B into local memory						\n"
"		const int tiledRow = 16*t + row;									\n"
"		const int tiledCol = 16*t + col;									\n"
"		Asub[col][row] = A[tiledCol*M + globalRow];							\n"
"		Bsub[col][row] = B[globalCol*K + tiledRow];							\n"
"																			\n"
"		// Synchronise to make sure the tile is loaded						\n"
"		barrier(CLK_LOCAL_MEM_FENCE);										\n"
"																			\n"
"		// Perform the computation for a single tile						\n"
"		for (int k = 0; k<16; k++) {										\n"
"			acc += Asub[k][row] * Bsub[col][k];								\n"
"		}																	\n"
"																			\n"
"		// Synchronise before loading the next tile							\n"
"		barrier(CLK_LOCAL_MEM_FENCE);										\n"
"	}																		\n"
"																			\n"
"	// Store the final result in C											\n"
"	//C[globalCol*M + globalRow] = acc;										\n"
"	set(C, M, N, globalRow, globalCol, acc);								\n"
"}																			\n"
;

int main() {
	// This code executes on the OpenCL host

	int M = 1024;
	int N = 1024;
	int K = 1024;

	// Host data
	auto A = new float[M * K];
	auto B = new float[N * K];
	auto C = new float[M * N];
	auto R = new float[M * N];
	// Initialize the input data

	for (int i = 0; i < M * K; ++i) {
		A[i] = 1.0f;
	}
	for (int j = 0; j < N * K; ++j) {
		B[j] = 1.0f;
	}

	for (int i = 0; i < M * N; ++i) {
		R[i] = 0.0f;
	}

	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < K; ++k) {
				R[i * N + j] += A[i * K + k] * B[j * K + k];
			}
		}
	}


	// Use this to check the output of each API call
	cl_int status;

	//-----------------------------------------------------
	// STEP 1: Discover and initialize the platforms
	//-----------------------------------------------------

	cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;

	// Use clGetPlatformIDs() to retrieve the number of
	// platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);

	// Allocate enough space for each platform
	platforms =
		(cl_platform_id*)malloc(
			numPlatforms * sizeof(cl_platform_id));

	// Fill in platforms with clGetPlatformIDs()
	status = clGetPlatformIDs(numPlatforms, platforms,
		NULL);

	//-----------------------------------------------------
	// STEP 2: Discover and initialize the devices
	//-----------------------------------------------------

	cl_uint numDevices = 0;
	cl_device_id *devices = NULL;

	// Use clGetDeviceIDs() to retrieve the number of
	// devices present
	status = clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_ALL,
		0,
		NULL,
		&numDevices);

	// Allocate enough space for each device
	devices =
		(cl_device_id*)malloc(
			numDevices * sizeof(cl_device_id));

	// Fill in devices with clGetDeviceIDs()
	status = clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_ALL,
		numDevices,
		devices,
		NULL);

	char info[100];
	size_t size = 0;
	size_t device_id = 0;
	size_t tmp = 0;
	std::array<size_t, 3> re;
	status = clGetDeviceInfo(devices[device_id], CL_DEVICE_MAX_WORK_GROUP_SIZE, 100, &re[0], &size);
	status = clGetDeviceInfo(devices[device_id], CL_DEVICE_MAX_WORK_ITEM_SIZES, 100, &re[0], &size);
	status = clGetDeviceInfo(devices[device_id], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 100, &tmp, &size);

	status = clGetDeviceInfo(devices[device_id], CL_DEVICE_NAME, 100, info, &size);
	printf("current platform is %s\n", info);
	//-----------------------------------------------------
	// STEP 3: Create a context
	//-----------------------------------------------------

	cl_context context = NULL;

	// Create a context using clCreateContext() and
	// associate it with the devices
	context = clCreateContext(
		NULL,
		numDevices,
		devices,
		NULL,
		NULL,
		&status);

	//-----------------------------------------------------
	// STEP 4: Create a command queue
	//-----------------------------------------------------

	cl_command_queue cmdQueue;

	// Create a command queue using clCreateCommandQueue(),
	// and associate it with the device you want to execute
	// on
	cmdQueue = clCreateCommandQueue(
		context,
		devices[device_id],
		0,
		&status);

	//-----------------------------------------------------
	// STEP 5: Create device buffers
	//-----------------------------------------------------

	cl_mem bufferA;  // Input array on the device
	cl_mem bufferB;  // Input array on the device
	cl_mem bufferC;  // Output array on the device

					 // Use clCreateBuffer() to create a buffer object (d_A)
					 // that will contain the data from the host array A
	bufferA = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		4 * M * K,
		NULL,
		&status);

	// Use clCreateBuffer() to create a buffer object (d_B)
	// that will contain the data from the host array B
	bufferB = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		4 * N * K,
		NULL,
		&status);

	// Use clCreateBuffer() to create a buffer object (d_C)
	// with enough space to hold the output data
	bufferC = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		4 * M * N,
		NULL,
		&status);

	//-----------------------------------------------------
	// STEP 6: Write host data to device buffers
	//-----------------------------------------------------

	// Use clEnqueueWriteBuffer() to write input array A to
	// the device buffer bufferA
	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferA,
		CL_FALSE,
		0,
		4 * M * K,
		A,
		0,
		NULL,
		NULL);

	// Use clEnqueueWriteBuffer() to write input array B to
	// the device buffer bufferB
	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferB,
		CL_FALSE,
		0,
		4 * N * K,
		B,
		0,
		NULL,
		NULL);

	//-----------------------------------------------------
	// STEP 7: Create and compile the program
	//-----------------------------------------------------

	// Create a program using clCreateProgramWithSource()
	cl_program program = clCreateProgramWithSource(
		context,
		1,
		(const char**)&programMatrixMul,
		NULL,
		&status);

	// Build (compile) the program for the devices with
	// clBuildProgram()
	status = clBuildProgram(
		program,
		numDevices,
		devices,
		NULL,
		NULL,
		NULL);

	if (status != 0) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
	}

	//-----------------------------------------------------
	// STEP 8: Create the kernel
	//-----------------------------------------------------

	cl_kernel kernel = NULL;

	// Use clCreateKernel() to create a kernel from the
	// vector addition function (named "vecadd")
	kernel = clCreateKernel(program, "myGEMM2", &status);


	//-----------------------------------------------------
	// STEP 9: Set the kernel arguments
	//-----------------------------------------------------

	// Associate the input and output buffers with the
	// kernel
	// using clSetKernelArg()
	status = clSetKernelArg(
		kernel,
		0,
		sizeof(int),
		&M);
	status |= clSetKernelArg(
		kernel,
		1,
		sizeof(int),
		&N);
	status |= clSetKernelArg(
		kernel,
		2,
		sizeof(int),
		&K);

	status = clSetKernelArg(
		kernel,
		3,
		sizeof(cl_mem),
		&bufferA);
	status |= clSetKernelArg(
		kernel,
		4,
		sizeof(cl_mem),
		&bufferB);
	status |= clSetKernelArg(
		kernel,
		5,
		sizeof(cl_mem),
		&bufferC);

	//-----------------------------------------------------
	// STEP 10: Configure the work-item structure
	//-----------------------------------------------------

	// Define an index space (global work size) of work
	// items for
	// execution. A workgroup size (local work size) is not
	// required,
	// but can be used.
	//size_t globalWorkSize[1];
	//// There are 'elements' work-items
	//globalWorkSize[0] = elements;

	size_t globalWorkSize[2] = { M, N };
	size_t localWorkSize[2] = { 16, 16 };



	auto t0 = chrono::high_resolution_clock::now();

	int count = 100;
	for (int i = 0; i < 100; ++i) {

		//-----------------------------------------------------
		// STEP 11: Enqueue the kernel for execution
		//-----------------------------------------------------

		// Execute the kernel by using
		// clEnqueueNDRangeKernel().
		// 'globalWorkSize' is the 1D dimension of the
		// work-items
		status = clEnqueueNDRangeKernel(
			cmdQueue,
			kernel,
			2,
			NULL,
			globalWorkSize,
			localWorkSize,
			0,
			NULL,
			NULL);

		if (status != 0) {
			printf("failed  to execute. ");
			return -1;
		}

		clFinish(cmdQueue);
	}


	auto t1 = chrono::high_resolution_clock::now();
	printf("kernel cost time with data copy is : %dns \n", (t1 - t0).count() / count);

	//-----------------------------------------------------
	// STEP 12: Read the output buffer back to the host
	//-----------------------------------------------------

	// Use clEnqueueReadBuffer() to read the OpenCL output
	// buffer (bufferC)
	// to the host output array (C)
	clEnqueueReadBuffer(
		cmdQueue,
		bufferC,
		CL_TRUE,
		0,
		4 * M * N,
		C,
		0,
		NULL,
		NULL);

	auto t2 = chrono::high_resolution_clock::now();
	printf(" data copy is : %dns \n", (t2 - t1).count());


	// Verify the output
	bool result = true;
	for (int i = 0; i < M * N; i++) {
		if (std::abs(C[i] - R[i]) > 0.01f) {
			result = false;
			break;
		}
	}
	if (result) {
		printf("Output is correct\n");
	}
	else {
		printf("Output is incorrect\n");
	}

	//-----------------------------------------------------
	// STEP 13: Release OpenCL resources
	//-----------------------------------------------------

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseContext(context);

	// Free host resources
	free(A);
	free(B);
	free(C);
	free(platforms);
	free(devices);

	return 0;
}
