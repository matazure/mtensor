__host__ int geth() { return 2; }

__device__ int getd() { return 3; }

__device__ __host__ int gethd() { return 4; }

#pragma nv_exec_check_disable
__host__ void gethtest() {
    geth();
    getd();
}

#pragma nv_exe_check_disable
__device__ void getdtest() {
    gethd();
    getd();
}

#pragma nv_exec_check_disable
__host__ __device__ void gethdtest() {
    geth();
    getd();
}