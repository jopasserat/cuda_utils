/*
 * @file cuda_utils.h
 * Extract of deprecated cutil_inline_runtime.h from NVIDIA
 *
 *  Created on: Dec 12, 2012
 *      Author: jH@CKtheRipper
 */

#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#ifdef _WIN32
#ifdef _DEBUG // Do this only in debug mode...
# define WINDOWS_LEAN_AND_MEAN
# include <windows.h>
# include <stdlib.h>
# undef min
# undef max
#endif
#endif

#include <cstdio>
#include <cstring>
#include <cstdlib>

#define cutilSafeCall(err) __cudaSafeCall (err, __FILE__, __LINE__)

// Give a little more for Windows : the console window often disapears before we can read the message
#ifdef _WIN32
# if 1//ndef UNICODE
# ifdef _DEBUG // Do this only in debug mode...
inline void VSPrintf(FILE *file, LPCSTR fmt, ...)
{
	size_t fmt2_sz = 2048;
	char *fmt2 = (char*)malloc(fmt2_sz);
	va_list vlist;
	va_start(vlist, fmt);
	while((_vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
	{
		fmt2_sz *= 2;
		if(fmt2) free(fmt2);
		fmt2 = (char*)malloc(fmt2_sz);
	}
	OutputDebugStringA(fmt2);
	fprintf(file, fmt2);
	free(fmt2);
}
# define FPRINTF(a) VSPrintf a
# else //debug
# define FPRINTF(a) fprintf a
// For other than Win32
# endif //debug
# else //unicode
// Unicode case... let's give-up for now and keep basic printf
# define FPRINTF(a) fprintf a
# endif //unicode
#else //win32
# define FPRINTF(a) fprintf a
#endif //win32


inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
	if (cudaSuccess != err) {
		FPRINTF(
				(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n", file, line, cudaGetErrorString( err)));
		exit(-1);
	}
}

#endif // CUDA_UTIL_H

