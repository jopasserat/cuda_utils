cuda_utils
==========

Helpers and utilities for CUDA

  * **cuda_logger.h** Logging utilities for CUDA inspired from the wb.h from the Coursera's Heterogeneous Computing course.
  * **cuda_util.h**   Error checking macro providing the cudaCutilSafeCall utility from the legacy cutil.h, now deprecated from the CUDA SDK.
  * **index_calculation.cuh** Thread index calculation facilities to use within a kernel.

Usage:
======

*cuda_util.h:*

    cutilSafeCall ( cudaRuntimeCall );

*cuda_logger.h*

    cudaUtilsTime_start( TimeType = < Generic | GPU | Compute | COPY >,  "Start log string" );
    ...
    cudaUtilsTime_start( TimeType = < Generic | GPU | Compute | COPY >,  "End log string" );
    
    cudaUtilsLog ( LogLevel = < OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE >, "Log string" );

*index_calculation.h*

    // global id of the calling thread's parent block
    getGlobalBlockId();

    // global id of the calling thread within its parent block
    getBlockThreadId();

    // global id of the calling thread across the whole kernel
    getGlobalThreadId();

    // total number of threads in the kernel
    getTotalThreadsCount();
