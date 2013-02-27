cuda_utils
==========

Helpers and utilities for CUDA

  * **cuda_logger.h** Logging utilities for CUDA inspired from the wb.h from the Coursera's Heterogeneous Computing course
  * **cuda_util.h**   Error checking macro providing the cudaCutilSafeCall utility from the legacy cutil.h, now deprecated from the CUDA SDK

Usage:
======

*cuda_util.h:*

    cutilSafeCall ( cudaRuntimeCall );

*cuda_logger.h*

    cudaUtilsTime_start( TimeType = < Generic | GPU | Compute | COPY >,  "Start log string" );
    ...
    cudaUtilsTime_start( TimeType = < Generic | GPU | Compute | COPY >,  "End log string" );
    
    cudaUtilsLog ( LogLevel = < OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE >, "Log string" );

