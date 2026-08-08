#include "codelet.h"
// This build targets armeabi-v7a/arm64-v8a/x86/x86_64; only arm64-v8a gets
// the -march=...+sve2 flag (CMakeLists.txt, scoped to this file), so guard
// the whole SVE2 section on the compiler-defined feature macro rather than
// assuming aarch64 -- keeps this file portable across all 4 ABI builds.
#if defined(JDSP_SVE2_FFT_X2)
#include <arm_sve.h>

// The SVE2 _x2 twiddle-factor code below (dft_twiddle_bfly4 and its call
// sites) is written with non-fused svmul+svadd/svsub specifically so it
// bit-matches the scalar DFT_N reference. Without this pragma, clang
// silently fuses that into hardware FMA (confirmed empirically, no
// -ffast-math needed to trigger it), which pushed relative error to 1.28e-4
// on a cancellation-prone bin in DFT1024_x2 -- over test_dft's own 1e-4
// tolerance. The project's CMakeLists.txt also has -Ofast on this file's
// build target, which implies fast-math and would silently reintroduce
// this; this pragma is a second, source-level guard against that
// regardless of how the CMake flags end up ordered.
#pragma clang fp contract(off)

// A radix-4 (stride-4) SVE2 butterfly was tried and measured worse than
// scalar on this hardware: on 128-bit SVE2, 4 elements = exactly one
// register, so there's no spare parallelism to harvest, only a
// ~6-instruction shuffle network to pay for (svuzp/svtbl/svtrn/svzip).
// Confirmed via simpleperf on-device: more instructions than plain scalar
// for small/medium transform sizes. All stride-4 stages in this file are
// deliberately plain scalar as a result — see the twiddle-factor stages
// below for the actual profitable vectorization target, which have real
// per-iteration work and no shuffle network needed at all.

// Real vectorization target: processes 4 CONSECUTIVE j values of a
// twiddle-factor butterfly stage for one buffer X, in place. j must be the
// first of a group of 4 (j, j+1, j+2, j+3), all strictly inside the stage's
// valid range so the reversed loads/stores below stay in-bounds.
//
// No shuffle network needed: the "forward" terms (X[j], X[j+half]) are
// already contiguous for consecutive j, and the "mirrored" terms
// (X[half-j], X[full-j]) are contiguous too, just in descending order — a
// single svrev_f32 fixes the lane order, which is far cheaper than a
// general cross-lane permute network.
//
// y0v/y1v are precomputed by the caller ONCE per j-group and shared across
// both A and B calls (sinTab reads are identical for both buffers at the
// same j) — halves the twiddle-table load cost vs. loading it separately
// per buffer.
static inline void dft_twiddle_bfly4(float *X, int half, int full, int j,
                                      svfloat32_t y0v, svfloat32_t y1v)
{
	svbool_t pg = svptrue_b32();
	svfloat32_t alpha = svld1_f32(pg, &X[j]);
	svfloat32_t beta  = svrev_f32(svld1_f32(pg, &X[half - j - 3]));
	svfloat32_t xh    = svld1_f32(pg, &X[j + half]);
	svfloat32_t xf    = svrev_f32(svld1_f32(pg, &X[full - j - 3]));

	svfloat32_t beta1 = svadd_f32_x(pg, svmul_f32_x(pg, xh, y1v), svmul_f32_x(pg, xf, y0v));
	svfloat32_t beta2 = svsub_f32_x(pg, svmul_f32_x(pg, xh, y0v), svmul_f32_x(pg, xf, y1v));

	svst1_f32(pg, &X[j], svadd_f32_x(pg, alpha, beta1));
	svst1_f32(pg, &X[j + half], svsub_f32_x(pg, alpha, beta1));
	svst1_f32(pg, &X[half - j - 3], svrev_f32(svadd_f32_x(pg, beta, beta2)));
	svst1_f32(pg, &X[full - j - 3], svrev_f32(svsub_f32_x(pg, beta, beta2)));
}
#endif // JDSP_SVE2_FFT_X2

void DFT1024(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 1024; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 1024; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 1024; i += 16)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 1024; i += 32)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 1024; i += 64)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 1024; i += 128)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 1024; i += 256)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 1024; i += 512)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[512];
	A[0] = alpha + beta;
	A[512] = alpha - beta;
	alpha = A[256];
	beta = A[768];
	A[256] = alpha + beta;
	A[768] = alpha - beta;
	for (j = 1; j < 256; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 256];
		alpha = A[j];
		beta = A[-j + 512];
		beta1 = A[j + 512] * y1 + A[-j + 1024] * y0;
		beta2 = A[j + 512] * y0 - A[-j + 1024] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 512] = alpha - beta1;
		A[-j + 512] = beta + beta2;
		A[-j + 1024] = beta - beta2;
	}
}
void DFT1048576(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 1048576; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 1048576; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 1048576; i += 16)
	{
		theta = 65536;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 65536;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 32)
	{
		theta = 32768;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 32768;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 64)
	{
		theta = 16384;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 16384;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 128)
	{
		theta = 8192;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 8192;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 256)
	{
		theta = 4096;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 4096;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 512)
	{
		theta = 2048;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 2048;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 1024)
	{
		theta = 1024;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 1024;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 2048)
	{
		theta = 512;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 4096)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 2048];
		A[i] = alpha + beta;
		A[i + 2048] = alpha - beta;
		alpha = A[i + 1024];
		beta = A[i + 3072];
		A[i + 1024] = alpha + beta;
		A[i + 3072] = alpha - beta;
		for (j = 1; j < 1024; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 8192)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 4096];
		A[i] = alpha + beta;
		A[i + 4096] = alpha - beta;
		alpha = A[i + 2048];
		beta = A[i + 6144];
		A[i + 2048] = alpha + beta;
		A[i + 6144] = alpha - beta;
		for (j = 1; j < 2048; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 4096];
			beta1 = A[i + j + 4096] * y1 + A[i - j + 8192] * y0;
			beta2 = A[i + j + 4096] * y0 - A[i - j + 8192] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 4096] = alpha - beta1;
			A[i - j + 4096] = beta + beta2;
			A[i - j + 8192] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 16384)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 8192];
		A[i] = alpha + beta;
		A[i + 8192] = alpha - beta;
		alpha = A[i + 4096];
		beta = A[i + 12288];
		A[i + 4096] = alpha + beta;
		A[i + 12288] = alpha - beta;
		for (j = 1; j < 4096; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 8192];
			beta1 = A[i + j + 8192] * y1 + A[i - j + 16384] * y0;
			beta2 = A[i + j + 8192] * y0 - A[i - j + 16384] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 8192] = alpha - beta1;
			A[i - j + 8192] = beta + beta2;
			A[i - j + 16384] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 32768)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 16384];
		A[i] = alpha + beta;
		A[i + 16384] = alpha - beta;
		alpha = A[i + 8192];
		beta = A[i + 24576];
		A[i + 8192] = alpha + beta;
		A[i + 24576] = alpha - beta;
		for (j = 1; j < 8192; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 16384];
			beta1 = A[i + j + 16384] * y1 + A[i - j + 32768] * y0;
			beta2 = A[i + j + 16384] * y0 - A[i - j + 32768] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 16384] = alpha - beta1;
			A[i - j + 16384] = beta + beta2;
			A[i - j + 32768] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 65536)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 32768];
		A[i] = alpha + beta;
		A[i + 32768] = alpha - beta;
		alpha = A[i + 16384];
		beta = A[i + 49152];
		A[i + 16384] = alpha + beta;
		A[i + 49152] = alpha - beta;
		for (j = 1; j < 16384; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 32768];
			beta1 = A[i + j + 32768] * y1 + A[i - j + 65536] * y0;
			beta2 = A[i + j + 32768] * y0 - A[i - j + 65536] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 32768] = alpha - beta1;
			A[i - j + 32768] = beta + beta2;
			A[i - j + 65536] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 131072)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 65536];
		A[i] = alpha + beta;
		A[i + 65536] = alpha - beta;
		alpha = A[i + 32768];
		beta = A[i + 98304];
		A[i + 32768] = alpha + beta;
		A[i + 98304] = alpha - beta;
		for (j = 1; j < 32768; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 65536];
			beta1 = A[i + j + 65536] * y1 + A[i - j + 131072] * y0;
			beta2 = A[i + j + 65536] * y0 - A[i - j + 131072] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 65536] = alpha - beta1;
			A[i - j + 65536] = beta + beta2;
			A[i - j + 131072] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 262144)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 131072];
		A[i] = alpha + beta;
		A[i + 131072] = alpha - beta;
		alpha = A[i + 65536];
		beta = A[i + 196608];
		A[i + 65536] = alpha + beta;
		A[i + 196608] = alpha - beta;
		for (j = 1; j < 65536; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 131072];
			beta1 = A[i + j + 131072] * y1 + A[i - j + 262144] * y0;
			beta2 = A[i + j + 131072] * y0 - A[i - j + 262144] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 131072] = alpha - beta1;
			A[i - j + 131072] = beta + beta2;
			A[i - j + 262144] = beta - beta2;
		}
	}
	for (i = 0; i < 1048576; i += 524288)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 262144];
		A[i] = alpha + beta;
		A[i + 262144] = alpha - beta;
		alpha = A[i + 131072];
		beta = A[i + 393216];
		A[i + 131072] = alpha + beta;
		A[i + 393216] = alpha - beta;
		for (j = 1; j < 131072; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 262144];
			alpha = A[i + j];
			beta = A[i - j + 262144];
			beta1 = A[i + j + 262144] * y1 + A[i - j + 524288] * y0;
			beta2 = A[i + j + 262144] * y0 - A[i - j + 524288] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 262144] = alpha - beta1;
			A[i - j + 262144] = beta + beta2;
			A[i - j + 524288] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[524288];
	A[0] = alpha + beta;
	A[524288] = alpha - beta;
	alpha = A[262144];
	beta = A[786432];
	A[262144] = alpha + beta;
	A[786432] = alpha - beta;
	for (j = 1; j < 262144; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 262144];
		alpha = A[j];
		beta = A[-j + 524288];
		beta1 = A[j + 524288] * y1 + A[-j + 1048576] * y0;
		beta2 = A[j + 524288] * y0 - A[-j + 1048576] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 524288] = alpha - beta1;
		A[-j + 524288] = beta + beta2;
		A[-j + 1048576] = beta - beta2;
	}
}
void DFT128(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 128; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 128; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 128; i += 16)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 128; i += 32)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 128; i += 64)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[64];
	A[0] = alpha + beta;
	A[64] = alpha - beta;
	alpha = A[32];
	beta = A[96];
	A[32] = alpha + beta;
	A[96] = alpha - beta;
	for (j = 1; j < 32; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 32];
		alpha = A[j];
		beta = A[-j + 64];
		beta1 = A[j + 64] * y1 + A[-j + 128] * y0;
		beta2 = A[j + 64] * y0 - A[-j + 128] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 64] = alpha - beta1;
		A[-j + 64] = beta + beta2;
		A[-j + 128] = beta - beta2;
	}
}
void DFT131072(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 131072; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 131072; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 131072; i += 16)
	{
		theta = 8192;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 8192;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 32)
	{
		theta = 4096;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 4096;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 64)
	{
		theta = 2048;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 2048;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 128)
	{
		theta = 1024;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 1024;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 256)
	{
		theta = 512;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 512)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 1024)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 2048)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 4096)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 2048];
		A[i] = alpha + beta;
		A[i + 2048] = alpha - beta;
		alpha = A[i + 1024];
		beta = A[i + 3072];
		A[i + 1024] = alpha + beta;
		A[i + 3072] = alpha - beta;
		for (j = 1; j < 1024; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 8192)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 4096];
		A[i] = alpha + beta;
		A[i + 4096] = alpha - beta;
		alpha = A[i + 2048];
		beta = A[i + 6144];
		A[i + 2048] = alpha + beta;
		A[i + 6144] = alpha - beta;
		for (j = 1; j < 2048; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 4096];
			beta1 = A[i + j + 4096] * y1 + A[i - j + 8192] * y0;
			beta2 = A[i + j + 4096] * y0 - A[i - j + 8192] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 4096] = alpha - beta1;
			A[i - j + 4096] = beta + beta2;
			A[i - j + 8192] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 16384)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 8192];
		A[i] = alpha + beta;
		A[i + 8192] = alpha - beta;
		alpha = A[i + 4096];
		beta = A[i + 12288];
		A[i + 4096] = alpha + beta;
		A[i + 12288] = alpha - beta;
		for (j = 1; j < 4096; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 8192];
			beta1 = A[i + j + 8192] * y1 + A[i - j + 16384] * y0;
			beta2 = A[i + j + 8192] * y0 - A[i - j + 16384] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 8192] = alpha - beta1;
			A[i - j + 8192] = beta + beta2;
			A[i - j + 16384] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 32768)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 16384];
		A[i] = alpha + beta;
		A[i + 16384] = alpha - beta;
		alpha = A[i + 8192];
		beta = A[i + 24576];
		A[i + 8192] = alpha + beta;
		A[i + 24576] = alpha - beta;
		for (j = 1; j < 8192; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 16384];
			beta1 = A[i + j + 16384] * y1 + A[i - j + 32768] * y0;
			beta2 = A[i + j + 16384] * y0 - A[i - j + 32768] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 16384] = alpha - beta1;
			A[i - j + 16384] = beta + beta2;
			A[i - j + 32768] = beta - beta2;
		}
	}
	for (i = 0; i < 131072; i += 65536)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 32768];
		A[i] = alpha + beta;
		A[i + 32768] = alpha - beta;
		alpha = A[i + 16384];
		beta = A[i + 49152];
		A[i + 16384] = alpha + beta;
		A[i + 49152] = alpha - beta;
		for (j = 1; j < 16384; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 32768];
			alpha = A[i + j];
			beta = A[i - j + 32768];
			beta1 = A[i + j + 32768] * y1 + A[i - j + 65536] * y0;
			beta2 = A[i + j + 32768] * y0 - A[i - j + 65536] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 32768] = alpha - beta1;
			A[i - j + 32768] = beta + beta2;
			A[i - j + 65536] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[65536];
	A[0] = alpha + beta;
	A[65536] = alpha - beta;
	alpha = A[32768];
	beta = A[98304];
	A[32768] = alpha + beta;
	A[98304] = alpha - beta;
	for (j = 1; j < 32768; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 32768];
		alpha = A[j];
		beta = A[-j + 65536];
		beta1 = A[j + 65536] * y1 + A[-j + 131072] * y0;
		beta2 = A[j + 65536] * y0 - A[-j + 131072] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 65536] = alpha - beta1;
		A[-j + 65536] = beta + beta2;
		A[-j + 131072] = beta - beta2;
	}
}
void DFT16(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 16; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 16; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	theta = 1;
	alpha = A[0];
	beta = A[8];
	A[0] = alpha + beta;
	A[8] = alpha - beta;
	alpha = A[4];
	beta = A[12];
	A[4] = alpha + beta;
	A[12] = alpha - beta;
	for (j = 1; j < 4; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 4];
		alpha = A[j];
		beta = A[-j + 8];
		beta1 = A[j + 8] * y1 + A[-j + 16] * y0;
		beta2 = A[j + 8] * y0 - A[-j + 16] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 8] = alpha - beta1;
		A[-j + 8] = beta + beta2;
		A[-j + 16] = beta - beta2;
	}
}
void DFT16384(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 16384; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 16384; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 16384; i += 16)
	{
		theta = 1024;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 1024;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 32)
	{
		theta = 512;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 64)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 128)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 256)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 512)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 1024)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 2048)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 4096)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 2048];
		A[i] = alpha + beta;
		A[i + 2048] = alpha - beta;
		alpha = A[i + 1024];
		beta = A[i + 3072];
		A[i + 1024] = alpha + beta;
		A[i + 3072] = alpha - beta;
		for (j = 1; j < 1024; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
		}
	}
	for (i = 0; i < 16384; i += 8192)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 4096];
		A[i] = alpha + beta;
		A[i + 4096] = alpha - beta;
		alpha = A[i + 2048];
		beta = A[i + 6144];
		A[i + 2048] = alpha + beta;
		A[i + 6144] = alpha - beta;
		for (j = 1; j < 2048; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j];
			beta = A[i - j + 4096];
			beta1 = A[i + j + 4096] * y1 + A[i - j + 8192] * y0;
			beta2 = A[i + j + 4096] * y0 - A[i - j + 8192] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 4096] = alpha - beta1;
			A[i - j + 4096] = beta + beta2;
			A[i - j + 8192] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[8192];
	A[0] = alpha + beta;
	A[8192] = alpha - beta;
	alpha = A[4096];
	beta = A[12288];
	A[4096] = alpha + beta;
	A[12288] = alpha - beta;
	for (j = 1; j < 4096; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 4096];
		alpha = A[j];
		beta = A[-j + 8192];
		beta1 = A[j + 8192] * y1 + A[-j + 16384] * y0;
		beta2 = A[j + 8192] * y0 - A[-j + 16384] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 8192] = alpha - beta1;
		A[-j + 8192] = beta + beta2;
		A[-j + 16384] = beta - beta2;
	}
}
void DFT2(float *A, const float *sinTab)
{
	float y0 = A[0] + A[1];
	A[1] = A[0] - A[1];
	A[0] = y0;
}
void DFT2048(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 2048; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 2048; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 2048; i += 16)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 2048; i += 32)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 2048; i += 64)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 2048; i += 128)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 2048; i += 256)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 2048; i += 512)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 2048; i += 1024)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[1024];
	A[0] = alpha + beta;
	A[1024] = alpha - beta;
	alpha = A[512];
	beta = A[1536];
	A[512] = alpha + beta;
	A[1536] = alpha - beta;
	for (j = 1; j < 512; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 512];
		alpha = A[j];
		beta = A[-j + 1024];
		beta1 = A[j + 1024] * y1 + A[-j + 2048] * y0;
		beta2 = A[j + 1024] * y0 - A[-j + 2048] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 1024] = alpha - beta1;
		A[-j + 1024] = beta + beta2;
		A[-j + 2048] = beta - beta2;
	}
}
void DFT256(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 256; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 256; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 256; i += 16)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 64];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 256; i += 32)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 64];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 256; i += 64)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 64];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 256; i += 128)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 64];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[128];
	A[0] = alpha + beta;
	A[128] = alpha - beta;
	alpha = A[64];
	beta = A[192];
	A[64] = alpha + beta;
	A[192] = alpha - beta;
	for (j = 1; j < 64; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 64];
		alpha = A[j];
		beta = A[-j + 128];
		beta1 = A[j + 128] * y1 + A[-j + 256] * y0;
		beta2 = A[j + 128] * y0 - A[-j + 256] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 128] = alpha - beta1;
		A[-j + 128] = beta + beta2;
		A[-j + 256] = beta - beta2;
	}
}
void DFT262144(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 262144; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 262144; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 262144; i += 16)
	{
		theta = 16384;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 16384;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 32)
	{
		theta = 8192;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 8192;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 64)
	{
		theta = 4096;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 4096;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 128)
	{
		theta = 2048;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 2048;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 256)
	{
		theta = 1024;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 1024;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 512)
	{
		theta = 512;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 1024)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 2048)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 4096)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 2048];
		A[i] = alpha + beta;
		A[i + 2048] = alpha - beta;
		alpha = A[i + 1024];
		beta = A[i + 3072];
		A[i + 1024] = alpha + beta;
		A[i + 3072] = alpha - beta;
		for (j = 1; j < 1024; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 8192)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 4096];
		A[i] = alpha + beta;
		A[i + 4096] = alpha - beta;
		alpha = A[i + 2048];
		beta = A[i + 6144];
		A[i + 2048] = alpha + beta;
		A[i + 6144] = alpha - beta;
		for (j = 1; j < 2048; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 4096];
			beta1 = A[i + j + 4096] * y1 + A[i - j + 8192] * y0;
			beta2 = A[i + j + 4096] * y0 - A[i - j + 8192] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 4096] = alpha - beta1;
			A[i - j + 4096] = beta + beta2;
			A[i - j + 8192] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 16384)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 8192];
		A[i] = alpha + beta;
		A[i + 8192] = alpha - beta;
		alpha = A[i + 4096];
		beta = A[i + 12288];
		A[i + 4096] = alpha + beta;
		A[i + 12288] = alpha - beta;
		for (j = 1; j < 4096; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 8192];
			beta1 = A[i + j + 8192] * y1 + A[i - j + 16384] * y0;
			beta2 = A[i + j + 8192] * y0 - A[i - j + 16384] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 8192] = alpha - beta1;
			A[i - j + 8192] = beta + beta2;
			A[i - j + 16384] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 32768)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 16384];
		A[i] = alpha + beta;
		A[i + 16384] = alpha - beta;
		alpha = A[i + 8192];
		beta = A[i + 24576];
		A[i + 8192] = alpha + beta;
		A[i + 24576] = alpha - beta;
		for (j = 1; j < 8192; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 16384];
			beta1 = A[i + j + 16384] * y1 + A[i - j + 32768] * y0;
			beta2 = A[i + j + 16384] * y0 - A[i - j + 32768] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 16384] = alpha - beta1;
			A[i - j + 16384] = beta + beta2;
			A[i - j + 32768] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 65536)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 32768];
		A[i] = alpha + beta;
		A[i + 32768] = alpha - beta;
		alpha = A[i + 16384];
		beta = A[i + 49152];
		A[i + 16384] = alpha + beta;
		A[i + 49152] = alpha - beta;
		for (j = 1; j < 16384; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 32768];
			beta1 = A[i + j + 32768] * y1 + A[i - j + 65536] * y0;
			beta2 = A[i + j + 32768] * y0 - A[i - j + 65536] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 32768] = alpha - beta1;
			A[i - j + 32768] = beta + beta2;
			A[i - j + 65536] = beta - beta2;
		}
	}
	for (i = 0; i < 262144; i += 131072)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 65536];
		A[i] = alpha + beta;
		A[i + 65536] = alpha - beta;
		alpha = A[i + 32768];
		beta = A[i + 98304];
		A[i + 32768] = alpha + beta;
		A[i + 98304] = alpha - beta;
		for (j = 1; j < 32768; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 65536];
			alpha = A[i + j];
			beta = A[i - j + 65536];
			beta1 = A[i + j + 65536] * y1 + A[i - j + 131072] * y0;
			beta2 = A[i + j + 65536] * y0 - A[i - j + 131072] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 65536] = alpha - beta1;
			A[i - j + 65536] = beta + beta2;
			A[i - j + 131072] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[131072];
	A[0] = alpha + beta;
	A[131072] = alpha - beta;
	alpha = A[65536];
	beta = A[196608];
	A[65536] = alpha + beta;
	A[196608] = alpha - beta;
	for (j = 1; j < 65536; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 65536];
		alpha = A[j];
		beta = A[-j + 131072];
		beta1 = A[j + 131072] * y1 + A[-j + 262144] * y0;
		beta2 = A[j + 131072] * y0 - A[-j + 262144] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 131072] = alpha - beta1;
		A[-j + 131072] = beta + beta2;
		A[-j + 262144] = beta - beta2;
	}
}
void DFT32(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 32; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 32; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 32; i += 16)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[16];
	A[0] = alpha + beta;
	A[16] = alpha - beta;
	alpha = A[8];
	beta = A[24];
	A[8] = alpha + beta;
	A[24] = alpha - beta;
	for (j = 1; j < 8; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 8];
		alpha = A[j];
		beta = A[-j + 16];
		beta1 = A[j + 16] * y1 + A[-j + 32] * y0;
		beta2 = A[j + 16] * y0 - A[-j + 32] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 16] = alpha - beta1;
		A[-j + 16] = beta + beta2;
		A[-j + 32] = beta - beta2;
	}
}
void DFT32768(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 32768; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 32768; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 32768; i += 16)
	{
		theta = 2048;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 2048;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 32)
	{
		theta = 1024;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 1024;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 64)
	{
		theta = 512;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 128)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 256)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 512)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 1024)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 2048)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 4096)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 2048];
		A[i] = alpha + beta;
		A[i + 2048] = alpha - beta;
		alpha = A[i + 1024];
		beta = A[i + 3072];
		A[i + 1024] = alpha + beta;
		A[i + 3072] = alpha - beta;
		for (j = 1; j < 1024; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 8192)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 4096];
		A[i] = alpha + beta;
		A[i + 4096] = alpha - beta;
		alpha = A[i + 2048];
		beta = A[i + 6144];
		A[i + 2048] = alpha + beta;
		A[i + 6144] = alpha - beta;
		for (j = 1; j < 2048; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 4096];
			beta1 = A[i + j + 4096] * y1 + A[i - j + 8192] * y0;
			beta2 = A[i + j + 4096] * y0 - A[i - j + 8192] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 4096] = alpha - beta1;
			A[i - j + 4096] = beta + beta2;
			A[i - j + 8192] = beta - beta2;
		}
	}
	for (i = 0; i < 32768; i += 16384)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 8192];
		A[i] = alpha + beta;
		A[i + 8192] = alpha - beta;
		alpha = A[i + 4096];
		beta = A[i + 12288];
		A[i + 4096] = alpha + beta;
		A[i + 12288] = alpha - beta;
		for (j = 1; j < 4096; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 8192];
			alpha = A[i + j];
			beta = A[i - j + 8192];
			beta1 = A[i + j + 8192] * y1 + A[i - j + 16384] * y0;
			beta2 = A[i + j + 8192] * y0 - A[i - j + 16384] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 8192] = alpha - beta1;
			A[i - j + 8192] = beta + beta2;
			A[i - j + 16384] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[16384];
	A[0] = alpha + beta;
	A[16384] = alpha - beta;
	alpha = A[8192];
	beta = A[24576];
	A[8192] = alpha + beta;
	A[24576] = alpha - beta;
	for (j = 1; j < 8192; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 8192];
		alpha = A[j];
		beta = A[-j + 16384];
		beta1 = A[j + 16384] * y1 + A[-j + 32768] * y0;
		beta2 = A[j + 16384] * y0 - A[-j + 32768] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 16384] = alpha - beta1;
		A[-j + 16384] = beta + beta2;
		A[-j + 32768] = beta - beta2;
	}
}
void DFT4(float *A, const float *sinTab)
{
	float y0, y1, y2, y3;
	y0 = A[0] + A[1];
	y1 = A[0] - A[1];
	y2 = A[2] + A[3];
	y3 = A[2] - A[3];
	A[0] = y0 + y2;
	A[2] = y0 - y2;
	A[1] = y1 + y3;
	A[3] = y1 - y3;
}
void DFT4096(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 4096; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 4096; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 4096; i += 16)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 4096; i += 32)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 4096; i += 64)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 4096; i += 128)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 4096; i += 256)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 4096; i += 512)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 4096; i += 1024)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 4096; i += 2048)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[2048];
	A[0] = alpha + beta;
	A[2048] = alpha - beta;
	alpha = A[1024];
	beta = A[3072];
	A[1024] = alpha + beta;
	A[3072] = alpha - beta;
	for (j = 1; j < 1024; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 1024];
		alpha = A[j];
		beta = A[-j + 2048];
		beta1 = A[j + 2048] * y1 + A[-j + 4096] * y0;
		beta2 = A[j + 2048] * y0 - A[-j + 4096] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 2048] = alpha - beta1;
		A[-j + 2048] = beta + beta2;
		A[-j + 4096] = beta - beta2;
	}
}
void DFT512(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 512; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 512; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 512; i += 16)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 512; i += 32)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 512; i += 64)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 512; i += 128)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 512; i += 256)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[256];
	A[0] = alpha + beta;
	A[256] = alpha - beta;
	alpha = A[128];
	beta = A[384];
	A[128] = alpha + beta;
	A[384] = alpha - beta;
	for (j = 1; j < 128; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 128];
		alpha = A[j];
		beta = A[-j + 256];
		beta1 = A[j + 256] * y1 + A[-j + 512] * y0;
		beta2 = A[j + 256] * y0 - A[-j + 512] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 256] = alpha - beta1;
		A[-j + 256] = beta + beta2;
		A[-j + 512] = beta - beta2;
	}
}
void DFT524288(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 524288; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 524288; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 524288; i += 16)
	{
		theta = 32768;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 32768;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 32)
	{
		theta = 16384;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 16384;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 64)
	{
		theta = 8192;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 8192;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 128)
	{
		theta = 4096;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 4096;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 256)
	{
		theta = 2048;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 2048;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 512)
	{
		theta = 1024;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 1024;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 1024)
	{
		theta = 512;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 2048)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 4096)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 2048];
		A[i] = alpha + beta;
		A[i + 2048] = alpha - beta;
		alpha = A[i + 1024];
		beta = A[i + 3072];
		A[i + 1024] = alpha + beta;
		A[i + 3072] = alpha - beta;
		for (j = 1; j < 1024; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 8192)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 4096];
		A[i] = alpha + beta;
		A[i + 4096] = alpha - beta;
		alpha = A[i + 2048];
		beta = A[i + 6144];
		A[i + 2048] = alpha + beta;
		A[i + 6144] = alpha - beta;
		for (j = 1; j < 2048; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 4096];
			beta1 = A[i + j + 4096] * y1 + A[i - j + 8192] * y0;
			beta2 = A[i + j + 4096] * y0 - A[i - j + 8192] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 4096] = alpha - beta1;
			A[i - j + 4096] = beta + beta2;
			A[i - j + 8192] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 16384)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 8192];
		A[i] = alpha + beta;
		A[i + 8192] = alpha - beta;
		alpha = A[i + 4096];
		beta = A[i + 12288];
		A[i + 4096] = alpha + beta;
		A[i + 12288] = alpha - beta;
		for (j = 1; j < 4096; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 8192];
			beta1 = A[i + j + 8192] * y1 + A[i - j + 16384] * y0;
			beta2 = A[i + j + 8192] * y0 - A[i - j + 16384] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 8192] = alpha - beta1;
			A[i - j + 8192] = beta + beta2;
			A[i - j + 16384] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 32768)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 16384];
		A[i] = alpha + beta;
		A[i + 16384] = alpha - beta;
		alpha = A[i + 8192];
		beta = A[i + 24576];
		A[i + 8192] = alpha + beta;
		A[i + 24576] = alpha - beta;
		for (j = 1; j < 8192; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 16384];
			beta1 = A[i + j + 16384] * y1 + A[i - j + 32768] * y0;
			beta2 = A[i + j + 16384] * y0 - A[i - j + 32768] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 16384] = alpha - beta1;
			A[i - j + 16384] = beta + beta2;
			A[i - j + 32768] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 65536)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 32768];
		A[i] = alpha + beta;
		A[i + 32768] = alpha - beta;
		alpha = A[i + 16384];
		beta = A[i + 49152];
		A[i + 16384] = alpha + beta;
		A[i + 49152] = alpha - beta;
		for (j = 1; j < 16384; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 32768];
			beta1 = A[i + j + 32768] * y1 + A[i - j + 65536] * y0;
			beta2 = A[i + j + 32768] * y0 - A[i - j + 65536] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 32768] = alpha - beta1;
			A[i - j + 32768] = beta + beta2;
			A[i - j + 65536] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 131072)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 65536];
		A[i] = alpha + beta;
		A[i + 65536] = alpha - beta;
		alpha = A[i + 32768];
		beta = A[i + 98304];
		A[i + 32768] = alpha + beta;
		A[i + 98304] = alpha - beta;
		for (j = 1; j < 32768; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 65536];
			beta1 = A[i + j + 65536] * y1 + A[i - j + 131072] * y0;
			beta2 = A[i + j + 65536] * y0 - A[i - j + 131072] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 65536] = alpha - beta1;
			A[i - j + 65536] = beta + beta2;
			A[i - j + 131072] = beta - beta2;
		}
	}
	for (i = 0; i < 524288; i += 262144)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 131072];
		A[i] = alpha + beta;
		A[i + 131072] = alpha - beta;
		alpha = A[i + 65536];
		beta = A[i + 196608];
		A[i + 65536] = alpha + beta;
		A[i + 196608] = alpha - beta;
		for (j = 1; j < 65536; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 131072];
			alpha = A[i + j];
			beta = A[i - j + 131072];
			beta1 = A[i + j + 131072] * y1 + A[i - j + 262144] * y0;
			beta2 = A[i + j + 131072] * y0 - A[i - j + 262144] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 131072] = alpha - beta1;
			A[i - j + 131072] = beta + beta2;
			A[i - j + 262144] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[262144];
	A[0] = alpha + beta;
	A[262144] = alpha - beta;
	alpha = A[131072];
	beta = A[393216];
	A[131072] = alpha + beta;
	A[393216] = alpha - beta;
	for (j = 1; j < 131072; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 131072];
		alpha = A[j];
		beta = A[-j + 262144];
		beta1 = A[j + 262144] * y1 + A[-j + 524288] * y0;
		beta2 = A[j + 262144] * y0 - A[-j + 524288] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 262144] = alpha - beta1;
		A[-j + 262144] = beta + beta2;
		A[-j + 524288] = beta - beta2;
	}
}
void DFT64(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 64; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 64; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 64; i += 16)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 64; i += 32)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[32];
	A[0] = alpha + beta;
	A[32] = alpha - beta;
	alpha = A[16];
	beta = A[48];
	A[16] = alpha + beta;
	A[48] = alpha - beta;
	for (j = 1; j < 16; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 16];
		alpha = A[j];
		beta = A[-j + 32];
		beta1 = A[j + 32] * y1 + A[-j + 64] * y0;
		beta2 = A[j + 32] * y0 - A[-j + 64] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 32] = alpha - beta1;
		A[-j + 32] = beta + beta2;
		A[-j + 64] = beta - beta2;
	}
}
void DFT65536(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 65536; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 65536; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 65536; i += 16)
	{
		theta = 4096;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 4096;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 32)
	{
		theta = 2048;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 2048;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 64)
	{
		theta = 1024;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 1024;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 128)
	{
		theta = 512;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 256)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 512)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 1024)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 2048)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 4096)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 2048];
		A[i] = alpha + beta;
		A[i + 2048] = alpha - beta;
		alpha = A[i + 1024];
		beta = A[i + 3072];
		A[i + 1024] = alpha + beta;
		A[i + 3072] = alpha - beta;
		for (j = 1; j < 1024; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 8192)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 4096];
		A[i] = alpha + beta;
		A[i + 4096] = alpha - beta;
		alpha = A[i + 2048];
		beta = A[i + 6144];
		A[i + 2048] = alpha + beta;
		A[i + 6144] = alpha - beta;
		for (j = 1; j < 2048; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 4096];
			beta1 = A[i + j + 4096] * y1 + A[i - j + 8192] * y0;
			beta2 = A[i + j + 4096] * y0 - A[i - j + 8192] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 4096] = alpha - beta1;
			A[i - j + 4096] = beta + beta2;
			A[i - j + 8192] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 16384)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 8192];
		A[i] = alpha + beta;
		A[i + 8192] = alpha - beta;
		alpha = A[i + 4096];
		beta = A[i + 12288];
		A[i + 4096] = alpha + beta;
		A[i + 12288] = alpha - beta;
		for (j = 1; j < 4096; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 8192];
			beta1 = A[i + j + 8192] * y1 + A[i - j + 16384] * y0;
			beta2 = A[i + j + 8192] * y0 - A[i - j + 16384] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 8192] = alpha - beta1;
			A[i - j + 8192] = beta + beta2;
			A[i - j + 16384] = beta - beta2;
		}
	}
	for (i = 0; i < 65536; i += 32768)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 16384];
		A[i] = alpha + beta;
		A[i + 16384] = alpha - beta;
		alpha = A[i + 8192];
		beta = A[i + 24576];
		A[i + 8192] = alpha + beta;
		A[i + 24576] = alpha - beta;
		for (j = 1; j < 8192; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 16384];
			alpha = A[i + j];
			beta = A[i - j + 16384];
			beta1 = A[i + j + 16384] * y1 + A[i - j + 32768] * y0;
			beta2 = A[i + j + 16384] * y0 - A[i - j + 32768] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 16384] = alpha - beta1;
			A[i - j + 16384] = beta + beta2;
			A[i - j + 32768] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[32768];
	A[0] = alpha + beta;
	A[32768] = alpha - beta;
	alpha = A[16384];
	beta = A[49152];
	A[16384] = alpha + beta;
	A[49152] = alpha - beta;
	for (j = 1; j < 16384; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 16384];
		alpha = A[j];
		beta = A[-j + 32768];
		beta1 = A[j + 32768] * y1 + A[-j + 65536] * y0;
		beta2 = A[j + 32768] * y0 - A[-j + 65536] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 32768] = alpha - beta1;
		A[-j + 32768] = beta + beta2;
		A[-j + 65536] = beta - beta2;
	}
}
void DFT8(float *A, const float *sinTab)
{
	int i;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 8; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	alpha = A[0];
	A[0] = alpha + A[4];
	A[4] = alpha - A[4];
	alpha = A[2];
	A[2] = alpha + A[6];
	A[6] = alpha - A[6];
	beta1 = 0.70710678118654752440084436210485f*(A[5] + A[7]);
	beta2 = 0.70710678118654752440084436210485f*(A[5] - A[7]);
	A[5] = A[1] - beta1;
	A[1] = A[1] + beta1;
	A[7] = A[3] - beta2;
	A[3] = A[3] + beta2;
}
void DFT8192(float *A, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, y2, y3;
	for (i = 0; i < 8192; i += 4)
	{
		alpha = A[i];
		beta = A[i + 1];
		beta1 = A[i + 2];
		beta2 = A[i + 3];
		y0 = alpha + beta;
		y1 = alpha - beta;
		y2 = beta1 + beta2;
		y3 = beta1 - beta2;
		A[i] = y0 + y2;
		A[i + 2] = y0 - y2;
		A[i + 1] = y1 + y3;
		A[i + 3] = y1 - y3;
	}
	for (i = 0; i < 8192; i += 8)
	{
		alpha = A[i];
		beta = A[i + 4];
		A[i] = alpha + beta;
		A[i + 4] = alpha - beta;
		alpha = A[i + 2];
		beta = A[i + 6];
		A[i + 2] = alpha + beta;
		A[i + 6] = alpha - beta;
		alpha = A[i + 1];
		beta1 = 0.70710678118654752440084436210485f*(A[i + 5] + A[i + 7]);
		beta2 = 0.70710678118654752440084436210485f*(A[i + 5] - A[i + 7]);
		A[i + 1] = alpha + beta1;
		A[i + 5] = alpha - beta1;
		alpha = A[i + 3];
		A[i + 3] = alpha + beta2;
		A[i + 7] = alpha - beta2;
	}
	for (i = 0; i < 8192; i += 16)
	{
		theta = 512;
		alpha = A[i];
		beta = A[i + 8];
		A[i] = alpha + beta;
		A[i + 8] = alpha - beta;
		alpha = A[i + 4];
		beta = A[i + 12];
		A[i + 4] = alpha + beta;
		A[i + 12] = alpha - beta;
		for (j = 1; j < 4; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
		}
	}
	for (i = 0; i < 8192; i += 32)
	{
		theta = 256;
		alpha = A[i];
		beta = A[i + 16];
		A[i] = alpha + beta;
		A[i + 16] = alpha - beta;
		alpha = A[i + 8];
		beta = A[i + 24];
		A[i + 8] = alpha + beta;
		A[i + 24] = alpha - beta;
		for (j = 1; j < 8; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
		}
	}
	for (i = 0; i < 8192; i += 64)
	{
		theta = 128;
		alpha = A[i];
		beta = A[i + 32];
		A[i] = alpha + beta;
		A[i + 32] = alpha - beta;
		alpha = A[i + 16];
		beta = A[i + 48];
		A[i + 16] = alpha + beta;
		A[i + 48] = alpha - beta;
		for (j = 1; j < 16; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
		}
	}
	for (i = 0; i < 8192; i += 128)
	{
		theta = 64;
		alpha = A[i];
		beta = A[i + 64];
		A[i] = alpha + beta;
		A[i + 64] = alpha - beta;
		alpha = A[i + 32];
		beta = A[i + 96];
		A[i + 32] = alpha + beta;
		A[i + 96] = alpha - beta;
		for (j = 1; j < 32; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
		}
	}
	for (i = 0; i < 8192; i += 256)
	{
		theta = 32;
		alpha = A[i];
		beta = A[i + 128];
		A[i] = alpha + beta;
		A[i + 128] = alpha - beta;
		alpha = A[i + 64];
		beta = A[i + 192];
		A[i + 64] = alpha + beta;
		A[i + 192] = alpha - beta;
		for (j = 1; j < 64; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
		}
	}
	for (i = 0; i < 8192; i += 512)
	{
		theta = 16;
		alpha = A[i];
		beta = A[i + 256];
		A[i] = alpha + beta;
		A[i + 256] = alpha - beta;
		alpha = A[i + 128];
		beta = A[i + 384];
		A[i + 128] = alpha + beta;
		A[i + 384] = alpha - beta;
		for (j = 1; j < 128; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
		}
	}
	for (i = 0; i < 8192; i += 1024)
	{
		theta = 8;
		alpha = A[i];
		beta = A[i + 512];
		A[i] = alpha + beta;
		A[i + 512] = alpha - beta;
		alpha = A[i + 256];
		beta = A[i + 768];
		A[i + 256] = alpha + beta;
		A[i + 768] = alpha - beta;
		for (j = 1; j < 256; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			theta += 8;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
		}
	}
	for (i = 0; i < 8192; i += 2048)
	{
		theta = 4;
		alpha = A[i];
		beta = A[i + 1024];
		A[i] = alpha + beta;
		A[i + 1024] = alpha - beta;
		alpha = A[i + 512];
		beta = A[i + 1536];
		A[i + 512] = alpha + beta;
		A[i + 1536] = alpha - beta;
		for (j = 1; j < 512; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			theta += 4;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
		}
	}
	for (i = 0; i < 8192; i += 4096)
	{
		theta = 2;
		alpha = A[i];
		beta = A[i + 2048];
		A[i] = alpha + beta;
		A[i + 2048] = alpha - beta;
		alpha = A[i + 1024];
		beta = A[i + 3072];
		A[i + 1024] = alpha + beta;
		A[i + 3072] = alpha - beta;
		for (j = 1; j < 1024; j++)
		{
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j];
			beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			theta += 2;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
		}
	}
	theta = 1;
	alpha = A[0];
	beta = A[4096];
	A[0] = alpha + beta;
	A[4096] = alpha - beta;
	alpha = A[2048];
	beta = A[6144];
	A[2048] = alpha + beta;
	A[6144] = alpha - beta;
	for (j = 1; j < 2048; j++)
	{
		y0 = sinTab[theta];
		y1 = sinTab[theta + 2048];
		alpha = A[j];
		beta = A[-j + 4096];
		beta1 = A[j + 4096] * y1 + A[-j + 8192] * y0;
		beta2 = A[j + 4096] * y0 - A[-j + 8192] * y1;
		theta += 1;
		A[j] = alpha + beta1;
		A[j + 4096] = alpha - beta1;
		A[-j + 4096] = beta + beta2;
		A[-j + 8192] = beta - beta2;
	}
}

// Batched dual-buffer variants: SVE2, Tensor G5 only -- see the top-of-file
// comment and codelet.h's matching guard on these declarations.
#if defined(JDSP_SVE2_FFT_X2)
void DFT256_x2(float *A, float *B, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, aa, ab;

	/* stride-4 -- plain scalar, deliberately not SVE2. See DFT1024_x2's
	 * comment: vectorizing a 4-wide radix-4 butterfly on 128-bit SVE2 costs
	 * more instructions than scalar (shuffle network, no spare parallelism
	 * to pay for it), confirmed via simpleperf on real hardware. */
	for (i = 0; i < 256; i += 4)
	{
		float a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
		float b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
		float ay0 = a0 + a1, ay1 = a0 - a1, ay2 = a2 + a3, ay3 = a2 - a3;
		float by0 = b0 + b1, by1 = b0 - b1, by2 = b2 + b3, by3 = b2 - b3;
		A[i] = ay0 + ay2; A[i + 2] = ay0 - ay2;
		A[i + 1] = ay1 + ay3; A[i + 3] = ay1 - ay3;
		B[i] = by0 + by2; B[i + 2] = by0 - by2;
		B[i + 1] = by1 + by3; B[i + 3] = by1 - by3;
	}

	/* stride-8 batched */
	for (i = 0; i < 256; i += 8) {
		float a1 = A[i], a2 = A[i + 2], a3 = A[i + 1], a4 = A[i + 3];
		float a5 = A[i + 5], a7 = A[i + 7];
		float b1 = B[i], b2 = B[i + 2], b3 = B[i + 1], b4 = B[i + 3];
		float b5 = B[i + 5], b7 = B[i + 7];
		float a_b1 = 0.70710678118654752440084436210485f * (a5 + a7);
		float a_b2 = 0.70710678118654752440084436210485f * (a5 - a7);
		float b_b1 = 0.70710678118654752440084436210485f * (b5 + b7);
		float b_b2 = 0.70710678118654752440084436210485f * (b5 - b7);
		A[i] = a1 + A[i + 4]; A[i + 4] = a1 - A[i + 4];
		B[i] = b1 + B[i + 4]; B[i + 4] = b1 - B[i + 4];
		A[i + 2] = a2 + A[i + 6]; A[i + 6] = a2 - A[i + 6];
		B[i + 2] = b2 + B[i + 6]; B[i + 6] = b2 - B[i + 6];
		A[i + 1] = a3 + a_b1; A[i + 5] = a3 - a_b1;
		B[i + 1] = b3 + b_b1; B[i + 5] = b3 - b_b1;
		A[i + 3] = a4 + a_b2; A[i + 7] = a4 - a_b2;
		B[i + 3] = b4 + b_b2; B[i + 7] = b4 - b_b2;
	}

	/* stride-16 batched */
	for (i = 0; i < 256; i += 16) {
		theta = 16;
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 8]; B[i] = ab + B[i + 8];
		A[i + 8] = aa - A[i + 8]; B[i + 8] = ab - B[i + 8];
		aa = A[i + 4]; ab = B[i + 4];
		A[i + 4] = aa + A[i + 12];
		B[i + 4] = ab + B[i + 12];
		A[i + 12] = aa - A[i + 12];
		B[i + 12] = ab - B[i + 12];
		for (j = 1; j < 4; j++) {
			y0 = sinTab[theta];
			y1 = sinTab[theta + 64];
			alpha = A[i + j]; beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 16;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 8];
			beta1 = B[i + j + 8] * y1 + B[i - j + 16] * y0;
			beta2 = B[i + j + 8] * y0 - B[i - j + 16] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 8] = alpha - beta1;
			B[i - j + 8] = beta + beta2;
			B[i - j + 16] = beta - beta2;
		}
	}

	/* stride-32 batched: theta = 8*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 256; i += 32) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 16]; B[i] = ab + B[i + 16];
		A[i + 16] = aa - A[i + 16]; B[i + 16] = ab - B[i + 16];
		aa = A[i + 8]; ab = B[i + 8];
		A[i + 8] = aa + A[i + 24];
		B[i + 8] = ab + B[i + 24];
		A[i + 24] = aa - A[i + 24];
		B[i + 24] = ab - B[i + 24];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 4; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 8 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 64];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 16, 32, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 16, 32, j, y0v, y1v);
			}
		}
		for (; j < 8; j++) {
			theta = 8 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 64];
			alpha = A[i + j]; beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 16];
			beta1 = B[i + j + 16] * y1 + B[i - j + 32] * y0;
			beta2 = B[i + j + 16] * y0 - B[i - j + 32] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 16] = alpha - beta1;
			B[i - j + 16] = beta + beta2;
			B[i - j + 32] = beta - beta2;
		}
	}

	/* stride-64 batched: theta = 4*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 256; i += 64) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 32]; B[i] = ab + B[i + 32];
		A[i + 32] = aa - A[i + 32]; B[i + 32] = ab - B[i + 32];
		aa = A[i + 16]; ab = B[i + 16];
		A[i + 16] = aa + A[i + 48];
		B[i + 16] = ab + B[i + 48];
		A[i + 48] = aa - A[i + 48];
		B[i + 48] = ab - B[i + 48];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 12; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 4 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 64];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 32, 64, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 32, 64, j, y0v, y1v);
			}
		}
		for (; j < 16; j++) {
			theta = 4 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 64];
			alpha = A[i + j]; beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 32];
			beta1 = B[i + j + 32] * y1 + B[i - j + 64] * y0;
			beta2 = B[i + j + 32] * y0 - B[i - j + 64] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 32] = alpha - beta1;
			B[i - j + 32] = beta + beta2;
			B[i - j + 64] = beta - beta2;
		}
	}

	/* stride-128 batched: theta = 2*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 256; i += 128) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 64]; B[i] = ab + B[i + 64];
		A[i + 64] = aa - A[i + 64]; B[i + 64] = ab - B[i + 64];
		aa = A[i + 32]; ab = B[i + 32];
		A[i + 32] = aa + A[i + 96];
		B[i + 32] = ab + B[i + 96];
		A[i + 96] = aa - A[i + 96];
		B[i + 96] = ab - B[i + 96];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 28; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 2 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 64];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 64, 128, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 64, 128, j, y0v, y1v);
			}
		}
		for (; j < 32; j++) {
			theta = 2 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 64];
			alpha = A[i + j]; beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 64];
			beta1 = B[i + j + 64] * y1 + B[i - j + 128] * y0;
			beta2 = B[i + j + 64] * y0 - B[i - j + 128] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 64] = alpha - beta1;
			B[i - j + 64] = beta + beta2;
			B[i - j + 128] = beta - beta2;
		}
	}

	/* final stage: theta == j, forward twiddle loads (see DFT1024_x2). */
	aa = A[0]; ab = B[0];
	A[0] = aa + A[128]; B[0] = ab + B[128];
	A[128] = aa - A[128]; B[128] = ab - B[128];
	aa = A[64]; ab = B[64];
	A[64] = aa + A[192]; B[64] = ab + B[192];
	A[192] = aa - A[192]; B[192] = ab - B[192];
	{
		svbool_t pg = svptrue_b32();
		for (j = 1; j + 3 <= 60; j += 4) {
			svfloat32_t y0v = svld1_f32(pg, &sinTab[j]);
			svfloat32_t y1v = svld1_f32(pg, &sinTab[j + 64]);
			dft_twiddle_bfly4(A, 128, 256, j, y0v, y1v);
			dft_twiddle_bfly4(B, 128, 256, j, y0v, y1v);
		}
	}
	for (; j < 64; j++) {
		y0 = sinTab[j];
		y1 = sinTab[j + 64];
		alpha = A[j]; beta = A[-j + 128];
		beta1 = A[j + 128] * y1 + A[-j + 256] * y0;
		beta2 = A[j + 128] * y0 - A[-j + 256] * y1;
		A[j] = alpha + beta1;
		A[j + 128] = alpha - beta1;
		A[-j + 128] = beta + beta2;
		A[-j + 256] = beta - beta2;
		alpha = B[j]; beta = B[-j + 128];
		beta1 = B[j + 128] * y1 + B[-j + 256] * y0;
		beta2 = B[j + 128] * y0 - B[-j + 256] * y1;
		B[j] = alpha + beta1;
		B[j + 128] = alpha - beta1;
		B[-j + 128] = beta + beta2;
		B[-j + 256] = beta - beta2;
	}
}

void DFT512_x2(float *A, float *B, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, aa, ab;

	/* stride-4 -- plain scalar, deliberately not SVE2. See DFT1024_x2's
	 * comment: vectorizing a 4-wide radix-4 butterfly on 128-bit SVE2 costs
	 * more instructions than scalar (shuffle network, no spare parallelism
	 * to pay for it), confirmed via simpleperf on real hardware. */
	for (i = 0; i < 512; i += 4)
	{
		float a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
		float b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
		float ay0 = a0 + a1, ay1 = a0 - a1, ay2 = a2 + a3, ay3 = a2 - a3;
		float by0 = b0 + b1, by1 = b0 - b1, by2 = b2 + b3, by3 = b2 - b3;
		A[i] = ay0 + ay2; A[i + 2] = ay0 - ay2;
		A[i + 1] = ay1 + ay3; A[i + 3] = ay1 - ay3;
		B[i] = by0 + by2; B[i + 2] = by0 - by2;
		B[i + 1] = by1 + by3; B[i + 3] = by1 - by3;
	}

	/* stride-8 batched */
	for (i = 0; i < 512; i += 8) {
		float a1 = A[i], a2 = A[i + 2], a3 = A[i + 1], a4 = A[i + 3];
		float a5 = A[i + 5], a7 = A[i + 7];
		float b1 = B[i], b2 = B[i + 2], b3 = B[i + 1], b4 = B[i + 3];
		float b5 = B[i + 5], b7 = B[i + 7];
		float a_b1 = 0.70710678118654752440084436210485f * (a5 + a7);
		float a_b2 = 0.70710678118654752440084436210485f * (a5 - a7);
		float b_b1 = 0.70710678118654752440084436210485f * (b5 + b7);
		float b_b2 = 0.70710678118654752440084436210485f * (b5 - b7);
		A[i] = a1 + A[i + 4]; A[i + 4] = a1 - A[i + 4];
		B[i] = b1 + B[i + 4]; B[i + 4] = b1 - B[i + 4];
		A[i + 2] = a2 + A[i + 6]; A[i + 6] = a2 - A[i + 6];
		B[i + 2] = b2 + B[i + 6]; B[i + 6] = b2 - B[i + 6];
		A[i + 1] = a3 + a_b1; A[i + 5] = a3 - a_b1;
		B[i + 1] = b3 + b_b1; B[i + 5] = b3 - b_b1;
		A[i + 3] = a4 + a_b2; A[i + 7] = a4 - a_b2;
		B[i + 3] = b4 + b_b2; B[i + 7] = b4 - b_b2;
	}

	/* stride-16 batched */
	for (i = 0; i < 512; i += 16) {
		theta = 32;
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 8]; B[i] = ab + B[i + 8];
		A[i + 8] = aa - A[i + 8]; B[i + 8] = ab - B[i + 8];
		aa = A[i + 4]; ab = B[i + 4];
		A[i + 4] = aa + A[i + 12];
		B[i + 4] = ab + B[i + 12];
		A[i + 12] = aa - A[i + 12];
		B[i + 12] = ab - B[i + 12];
		for (j = 1; j < 4; j++) {
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j]; beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 32;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 8];
			beta1 = B[i + j + 8] * y1 + B[i - j + 16] * y0;
			beta2 = B[i + j + 8] * y0 - B[i - j + 16] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 8] = alpha - beta1;
			B[i - j + 8] = beta + beta2;
			B[i - j + 16] = beta - beta2;
		}
	}

	/* stride-32 batched: theta = 16*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 512; i += 32) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 16]; B[i] = ab + B[i + 16];
		A[i + 16] = aa - A[i + 16]; B[i + 16] = ab - B[i + 16];
		aa = A[i + 8]; ab = B[i + 8];
		A[i + 8] = aa + A[i + 24];
		B[i + 8] = ab + B[i + 24];
		A[i + 24] = aa - A[i + 24];
		B[i + 24] = ab - B[i + 24];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 4; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 16 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 128];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 16, 32, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 16, 32, j, y0v, y1v);
			}
		}
		for (; j < 8; j++) {
			theta = 16 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j]; beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 16];
			beta1 = B[i + j + 16] * y1 + B[i - j + 32] * y0;
			beta2 = B[i + j + 16] * y0 - B[i - j + 32] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 16] = alpha - beta1;
			B[i - j + 16] = beta + beta2;
			B[i - j + 32] = beta - beta2;
		}
	}

	/* stride-64 batched: theta = 8*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 512; i += 64) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 32]; B[i] = ab + B[i + 32];
		A[i + 32] = aa - A[i + 32]; B[i + 32] = ab - B[i + 32];
		aa = A[i + 16]; ab = B[i + 16];
		A[i + 16] = aa + A[i + 48];
		B[i + 16] = ab + B[i + 48];
		A[i + 48] = aa - A[i + 48];
		B[i + 48] = ab - B[i + 48];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 12; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 8 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 128];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 32, 64, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 32, 64, j, y0v, y1v);
			}
		}
		for (; j < 16; j++) {
			theta = 8 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j]; beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 32];
			beta1 = B[i + j + 32] * y1 + B[i - j + 64] * y0;
			beta2 = B[i + j + 32] * y0 - B[i - j + 64] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 32] = alpha - beta1;
			B[i - j + 32] = beta + beta2;
			B[i - j + 64] = beta - beta2;
		}
	}

	/* stride-128 batched: theta = 4*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 512; i += 128) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 64]; B[i] = ab + B[i + 64];
		A[i + 64] = aa - A[i + 64]; B[i + 64] = ab - B[i + 64];
		aa = A[i + 32]; ab = B[i + 32];
		A[i + 32] = aa + A[i + 96];
		B[i + 32] = ab + B[i + 96];
		A[i + 96] = aa - A[i + 96];
		B[i + 96] = ab - B[i + 96];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 28; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 4 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 128];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 64, 128, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 64, 128, j, y0v, y1v);
			}
		}
		for (; j < 32; j++) {
			theta = 4 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j]; beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 64];
			beta1 = B[i + j + 64] * y1 + B[i - j + 128] * y0;
			beta2 = B[i + j + 64] * y0 - B[i - j + 128] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 64] = alpha - beta1;
			B[i - j + 64] = beta + beta2;
			B[i - j + 128] = beta - beta2;
		}
	}

	/* stride-256 batched: theta = 2*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 512; i += 256) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 128]; B[i] = ab + B[i + 128];
		A[i + 128] = aa - A[i + 128]; B[i + 128] = ab - B[i + 128];
		aa = A[i + 64]; ab = B[i + 64];
		A[i + 64] = aa + A[i + 192];
		B[i + 64] = ab + B[i + 192];
		A[i + 192] = aa - A[i + 192];
		B[i + 192] = ab - B[i + 192];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 60; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 2 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 128];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 128, 256, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 128, 256, j, y0v, y1v);
			}
		}
		for (; j < 64; j++) {
			theta = 2 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 128];
			alpha = A[i + j]; beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 128];
			beta1 = B[i + j + 128] * y1 + B[i - j + 256] * y0;
			beta2 = B[i + j + 128] * y0 - B[i - j + 256] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 128] = alpha - beta1;
			B[i - j + 128] = beta + beta2;
			B[i - j + 256] = beta - beta2;
		}
	}

	/* final stage: theta == j, forward twiddle loads (see DFT1024_x2). */
	aa = A[0]; ab = B[0];
	A[0] = aa + A[256]; B[0] = ab + B[256];
	A[256] = aa - A[256]; B[256] = ab - B[256];
	aa = A[128]; ab = B[128];
	A[128] = aa + A[384]; B[128] = ab + B[384];
	A[384] = aa - A[384]; B[384] = ab - B[384];
	{
		svbool_t pg = svptrue_b32();
		for (j = 1; j + 3 <= 124; j += 4) {
			svfloat32_t y0v = svld1_f32(pg, &sinTab[j]);
			svfloat32_t y1v = svld1_f32(pg, &sinTab[j + 128]);
			dft_twiddle_bfly4(A, 256, 512, j, y0v, y1v);
			dft_twiddle_bfly4(B, 256, 512, j, y0v, y1v);
		}
	}
	for (; j < 128; j++) {
		y0 = sinTab[j];
		y1 = sinTab[j + 128];
		alpha = A[j]; beta = A[-j + 256];
		beta1 = A[j + 256] * y1 + A[-j + 512] * y0;
		beta2 = A[j + 256] * y0 - A[-j + 512] * y1;
		A[j] = alpha + beta1;
		A[j + 256] = alpha - beta1;
		A[-j + 256] = beta + beta2;
		A[-j + 512] = beta - beta2;
		alpha = B[j]; beta = B[-j + 256];
		beta1 = B[j + 256] * y1 + B[-j + 512] * y0;
		beta2 = B[j + 256] * y0 - B[-j + 512] * y1;
		B[j] = alpha + beta1;
		B[j + 256] = alpha - beta1;
		B[-j + 256] = beta + beta2;
		B[-j + 512] = beta - beta2;
	}
}

void DFT1024_x2(float *A, float *B, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, aa, ab;

	/* stride-4 — plain scalar, deliberately not SVE2 here. Measured via
	 * simpleperf: vectorizing a 4-wide radix-4 butterfly on 128-bit SVE2
	 * costs MORE instructions than scalar (shuffle network with no spare
	 * parallelism to pay for it), not fewer. See the file-header comment. */
	for (i = 0; i < 1024; i += 4)
	{
		float a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
		float b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
		float ay0 = a0 + a1, ay1 = a0 - a1, ay2 = a2 + a3, ay3 = a2 - a3;
		float by0 = b0 + b1, by1 = b0 - b1, by2 = b2 + b3, by3 = b2 - b3;
		A[i] = ay0 + ay2; A[i + 2] = ay0 - ay2;
		A[i + 1] = ay1 + ay3; A[i + 3] = ay1 - ay3;
		B[i] = by0 + by2; B[i + 2] = by0 - by2;
		B[i + 1] = by1 + by3; B[i + 3] = by1 - by3;
	}

	/* stride-8 batched */
	for (i = 0; i < 1024; i += 8) {
		float a1 = A[i], a2 = A[i + 2], a3 = A[i + 1], a4 = A[i + 3];
		float a5 = A[i + 5], a7 = A[i + 7];
		float b1 = B[i], b2 = B[i + 2], b3 = B[i + 1], b4 = B[i + 3];
		float b5 = B[i + 5], b7 = B[i + 7];
		float a_b1 = 0.70710678118654752440084436210485f * (a5 + a7);
		float a_b2 = 0.70710678118654752440084436210485f * (a5 - a7);
		float b_b1 = 0.70710678118654752440084436210485f * (b5 + b7);
		float b_b2 = 0.70710678118654752440084436210485f * (b5 - b7);
		A[i] = a1 + A[i + 4]; A[i + 4] = a1 - A[i + 4];
		B[i] = b1 + B[i + 4]; B[i + 4] = b1 - B[i + 4];
		A[i + 2] = a2 + A[i + 6]; A[i + 6] = a2 - A[i + 6];
		B[i + 2] = b2 + B[i + 6]; B[i + 6] = b2 - B[i + 6];
		A[i + 1] = a3 + a_b1; A[i + 5] = a3 - a_b1;
		B[i + 1] = b3 + b_b1; B[i + 5] = b3 - b_b1;
		A[i + 3] = a4 + a_b2; A[i + 7] = a4 - a_b2;
		B[i + 3] = b4 + b_b2; B[i + 7] = b4 - b_b2;
	}

	/* stride-16 batched */
	for (i = 0; i < 1024; i += 16) {
		theta = 64;
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 8]; B[i] = ab + B[i + 8];
		A[i + 8] = aa - A[i + 8]; B[i + 8] = ab - B[i + 8];
		aa = A[i + 4]; ab = B[i + 4];
		A[i + 4] = aa + A[i + 12];
		B[i + 4] = ab + B[i + 12];
		A[i + 12] = aa - A[i + 12];
		B[i + 12] = ab - B[i + 12];
		for (j = 1; j < 4; j++) {
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j]; beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 64;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 8];
			beta1 = B[i + j + 8] * y1 + B[i - j + 16] * y0;
			beta2 = B[i + j + 8] * y0 - B[i - j + 16] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 8] = alpha - beta1;
			B[i - j + 8] = beta + beta2;
			B[i - j + 16] = beta - beta2;
		}
	}

	/* stride-32 batched: theta = 32*j, only 7 j-iterations per i-block so
	 * only ONE group of 4 fits (tail is 3 of the 7) -- measuring whether
	 * this amortizes before doing it everywhere. */
	for (i = 0; i < 1024; i += 32) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 16]; B[i] = ab + B[i + 16];
		A[i + 16] = aa - A[i + 16]; B[i + 16] = ab - B[i + 16];
		aa = A[i + 8]; ab = B[i + 8];
		A[i + 8] = aa + A[i + 24];
		B[i + 8] = ab + B[i + 24];
		A[i + 24] = aa - A[i + 24];
		B[i + 24] = ab - B[i + 24];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 4; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 32 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 256];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 16, 32, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 16, 32, j, y0v, y1v);
			}
		}
		/* scalar tail: j=5,6,7 */
		for (; j < 8; j++) {
			theta = 32 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j]; beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 16];
			beta1 = B[i + j + 16] * y1 + B[i - j + 32] * y0;
			beta2 = B[i + j + 16] * y0 - B[i - j + 32] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 16] = alpha - beta1;
			B[i - j + 16] = beta + beta2;
			B[i - j + 32] = beta - beta2;
		}
	}

	/* stride-64 batched: theta = 16*j here (strided, not contiguous), so
	 * y0v/y1v are materialized via a 4-wide scalar gather into a scratch
	 * array then one vector load each -- cheaper than a real SVE2 gather
	 * for a small fixed stride (see INSTRUCTIONS.md). dft_twiddle_bfly4
	 * takes X+i as base so its internal half/full offsets stay local to
	 * this i-block, same helper as the final stage, just shifted. */
	for (i = 0; i < 1024; i += 64) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 32]; B[i] = ab + B[i + 32];
		A[i + 32] = aa - A[i + 32]; B[i + 32] = ab - B[i + 32];
		aa = A[i + 16]; ab = B[i + 16];
		A[i + 16] = aa + A[i + 48];
		B[i + 16] = ab + B[i + 48];
		A[i + 48] = aa - A[i + 48];
		B[i + 48] = ab - B[i + 48];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 12; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 16 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 256];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 32, 64, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 32, 64, j, y0v, y1v);
			}
		}
		/* scalar tail: j=13,14,15 (15 values isn't a multiple of 4) */
		for (; j < 16; j++) {
			theta = 16 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j]; beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 32];
			beta1 = B[i + j + 32] * y1 + B[i - j + 64] * y0;
			beta2 = B[i + j + 32] * y0 - B[i - j + 64] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 32] = alpha - beta1;
			B[i - j + 32] = beta + beta2;
			B[i - j + 64] = beta - beta2;
		}
	}

	/* stride-128 batched: theta = 8*j, same scratch-gather approach as
	 * stride-64. */
	for (i = 0; i < 1024; i += 128) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 64]; B[i] = ab + B[i + 64];
		A[i + 64] = aa - A[i + 64]; B[i + 64] = ab - B[i + 64];
		aa = A[i + 32]; ab = B[i + 32];
		A[i + 32] = aa + A[i + 96];
		B[i + 32] = ab + B[i + 96];
		A[i + 96] = aa - A[i + 96];
		B[i + 96] = ab - B[i + 96];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 28; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 8 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 256];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 64, 128, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 64, 128, j, y0v, y1v);
			}
		}
		/* scalar tail: j=29,30,31 */
		for (; j < 32; j++) {
			theta = 8 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j]; beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 64];
			beta1 = B[i + j + 64] * y1 + B[i - j + 128] * y0;
			beta2 = B[i + j + 64] * y0 - B[i - j + 128] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 64] = alpha - beta1;
			B[i - j + 64] = beta + beta2;
			B[i - j + 128] = beta - beta2;
		}
	}

	/* stride-256 batched: theta = 4*j, same scratch-gather approach. */
	for (i = 0; i < 1024; i += 256) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 128]; B[i] = ab + B[i + 128];
		A[i + 128] = aa - A[i + 128]; B[i + 128] = ab - B[i + 128];
		aa = A[i + 64]; ab = B[i + 64];
		A[i + 64] = aa + A[i + 192];
		B[i + 64] = ab + B[i + 192];
		A[i + 192] = aa - A[i + 192];
		B[i + 192] = ab - B[i + 192];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 60; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 4 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 256];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 128, 256, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 128, 256, j, y0v, y1v);
			}
		}
		/* scalar tail: j=61,62,63 */
		for (; j < 64; j++) {
			theta = 4 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j]; beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 128];
			beta1 = B[i + j + 128] * y1 + B[i - j + 256] * y0;
			beta2 = B[i + j + 128] * y0 - B[i - j + 256] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 128] = alpha - beta1;
			B[i - j + 128] = beta + beta2;
			B[i - j + 256] = beta - beta2;
		}
	}

	/* stride-512 batched: theta = 2*j, same scratch-gather approach. */
	for (i = 0; i < 1024; i += 512) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 256]; B[i] = ab + B[i + 256];
		A[i + 256] = aa - A[i + 256]; B[i + 256] = ab - B[i + 256];
		aa = A[i + 128]; ab = B[i + 128];
		A[i + 128] = aa + A[i + 384];
		B[i + 128] = ab + B[i + 384];
		A[i + 384] = aa - A[i + 384];
		B[i + 384] = ab - B[i + 384];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 124; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 2 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 256];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 256, 512, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 256, 512, j, y0v, y1v);
			}
		}
		/* scalar tail: j=125,126,127 */
		for (; j < 128; j++) {
			theta = 2 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 256];
			alpha = A[i + j]; beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 256];
			beta1 = B[i + j + 256] * y1 + B[i - j + 512] * y0;
			beta2 = B[i + j + 256] * y0 - B[i - j + 512] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 256] = alpha - beta1;
			B[i - j + 256] = beta + beta2;
			B[i - j + 512] = beta - beta2;
		}
	}

	/* final stage */
	theta = 1;
	aa = A[0]; ab = B[0];
	A[0] = aa + A[512]; B[0] = ab + B[512];
	A[512] = aa - A[512]; B[512] = ab - B[512];
	aa = A[256]; ab = B[256];
	A[256] = aa + A[768]; B[256] = ab + B[768];
	A[768] = aa - A[768]; B[768] = ab - B[768];
	/* Real SVE2: theta == j in this stage (both start at 1, step by 1), so
	 * the twiddle vectors load forward, no svrev needed for them. Process
	 * j in groups of 4 via dft_twiddle_bfly4; y0v/y1v computed once per
	 * group and shared across the A and B calls (same sinTab reads). */
	{
		svbool_t pg = svptrue_b32();
		for (j = 1; j + 3 <= 252; j += 4) {
			svfloat32_t y0v = svld1_f32(pg, &sinTab[j]);
			svfloat32_t y1v = svld1_f32(pg, &sinTab[j + 256]);
			dft_twiddle_bfly4(A, 512, 1024, j, y0v, y1v);
			dft_twiddle_bfly4(B, 512, 1024, j, y0v, y1v);
		}
	}
	/* scalar tail: j=253,254,255 (255 values isn't a multiple of 4) */
	for (; j < 256; j++) {
		y0 = sinTab[j];
		y1 = sinTab[j + 256];
		alpha = A[j]; beta = A[-j + 512];
		beta1 = A[j + 512] * y1 + A[-j + 1024] * y0;
		beta2 = A[j + 512] * y0 - A[-j + 1024] * y1;
		A[j] = alpha + beta1;
		A[j + 512] = alpha - beta1;
		A[-j + 512] = beta + beta2;
		A[-j + 1024] = beta - beta2;
		alpha = B[j]; beta = B[-j + 512];
		beta1 = B[j + 512] * y1 + B[-j + 1024] * y0;
		beta2 = B[j + 512] * y0 - B[-j + 1024] * y1;
		B[j] = alpha + beta1;
		B[j + 512] = alpha - beta1;
		B[-j + 512] = beta + beta2;
		B[-j + 1024] = beta - beta2;
	}
}

void DFT2048_x2(float *A, float *B, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, aa, ab;

	/* stride-4 -- plain scalar, deliberately not SVE2. See DFT1024_x2's
	 * comment: vectorizing a 4-wide radix-4 butterfly on 128-bit SVE2 costs
	 * more instructions than scalar (shuffle network, no spare parallelism
	 * to pay for it), confirmed via simpleperf on real hardware. */
	for (i = 0; i < 2048; i += 4)
	{
		float a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
		float b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
		float ay0 = a0 + a1, ay1 = a0 - a1, ay2 = a2 + a3, ay3 = a2 - a3;
		float by0 = b0 + b1, by1 = b0 - b1, by2 = b2 + b3, by3 = b2 - b3;
		A[i] = ay0 + ay2; A[i + 2] = ay0 - ay2;
		A[i + 1] = ay1 + ay3; A[i + 3] = ay1 - ay3;
		B[i] = by0 + by2; B[i + 2] = by0 - by2;
		B[i + 1] = by1 + by3; B[i + 3] = by1 - by3;
	}

	/* stride-8 batched */
	for (i = 0; i < 2048; i += 8) {
		float a1 = A[i], a2 = A[i + 2], a3 = A[i + 1], a4 = A[i + 3];
		float a5 = A[i + 5], a7 = A[i + 7];
		float b1 = B[i], b2 = B[i + 2], b3 = B[i + 1], b4 = B[i + 3];
		float b5 = B[i + 5], b7 = B[i + 7];
		float a_b1 = 0.70710678118654752440084436210485f * (a5 + a7);
		float a_b2 = 0.70710678118654752440084436210485f * (a5 - a7);
		float b_b1 = 0.70710678118654752440084436210485f * (b5 + b7);
		float b_b2 = 0.70710678118654752440084436210485f * (b5 - b7);
		A[i] = a1 + A[i + 4]; A[i + 4] = a1 - A[i + 4];
		B[i] = b1 + B[i + 4]; B[i + 4] = b1 - B[i + 4];
		A[i + 2] = a2 + A[i + 6]; A[i + 6] = a2 - A[i + 6];
		B[i + 2] = b2 + B[i + 6]; B[i + 6] = b2 - B[i + 6];
		A[i + 1] = a3 + a_b1; A[i + 5] = a3 - a_b1;
		B[i + 1] = b3 + b_b1; B[i + 5] = b3 - b_b1;
		A[i + 3] = a4 + a_b2; A[i + 7] = a4 - a_b2;
		B[i + 3] = b4 + b_b2; B[i + 7] = b4 - b_b2;
	}

	/* stride-16 batched */
	for (i = 0; i < 2048; i += 16) {
		theta = 128;
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 8]; B[i] = ab + B[i + 8];
		A[i + 8] = aa - A[i + 8]; B[i + 8] = ab - B[i + 8];
		aa = A[i + 4]; ab = B[i + 4];
		A[i + 4] = aa + A[i + 12];
		B[i + 4] = ab + B[i + 12];
		A[i + 12] = aa - A[i + 12];
		B[i + 12] = ab - B[i + 12];
		for (j = 1; j < 4; j++) {
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j]; beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 128;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 8];
			beta1 = B[i + j + 8] * y1 + B[i - j + 16] * y0;
			beta2 = B[i + j + 8] * y0 - B[i - j + 16] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 8] = alpha - beta1;
			B[i - j + 8] = beta + beta2;
			B[i - j + 16] = beta - beta2;
		}
	}

	/* stride-32 batched: theta = 64*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 2048; i += 32) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 16]; B[i] = ab + B[i + 16];
		A[i + 16] = aa - A[i + 16]; B[i + 16] = ab - B[i + 16];
		aa = A[i + 8]; ab = B[i + 8];
		A[i + 8] = aa + A[i + 24];
		B[i + 8] = ab + B[i + 24];
		A[i + 24] = aa - A[i + 24];
		B[i + 24] = ab - B[i + 24];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 4; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 64 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 512];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 16, 32, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 16, 32, j, y0v, y1v);
			}
		}
		for (; j < 8; j++) {
			theta = 64 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j]; beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 16];
			beta1 = B[i + j + 16] * y1 + B[i - j + 32] * y0;
			beta2 = B[i + j + 16] * y0 - B[i - j + 32] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 16] = alpha - beta1;
			B[i - j + 16] = beta + beta2;
			B[i - j + 32] = beta - beta2;
		}
	}

	/* stride-64 batched: theta = 32*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 2048; i += 64) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 32]; B[i] = ab + B[i + 32];
		A[i + 32] = aa - A[i + 32]; B[i + 32] = ab - B[i + 32];
		aa = A[i + 16]; ab = B[i + 16];
		A[i + 16] = aa + A[i + 48];
		B[i + 16] = ab + B[i + 48];
		A[i + 48] = aa - A[i + 48];
		B[i + 48] = ab - B[i + 48];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 12; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 32 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 512];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 32, 64, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 32, 64, j, y0v, y1v);
			}
		}
		for (; j < 16; j++) {
			theta = 32 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j]; beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 32];
			beta1 = B[i + j + 32] * y1 + B[i - j + 64] * y0;
			beta2 = B[i + j + 32] * y0 - B[i - j + 64] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 32] = alpha - beta1;
			B[i - j + 32] = beta + beta2;
			B[i - j + 64] = beta - beta2;
		}
	}

	/* stride-128 batched: theta = 16*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 2048; i += 128) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 64]; B[i] = ab + B[i + 64];
		A[i + 64] = aa - A[i + 64]; B[i + 64] = ab - B[i + 64];
		aa = A[i + 32]; ab = B[i + 32];
		A[i + 32] = aa + A[i + 96];
		B[i + 32] = ab + B[i + 96];
		A[i + 96] = aa - A[i + 96];
		B[i + 96] = ab - B[i + 96];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 28; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 16 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 512];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 64, 128, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 64, 128, j, y0v, y1v);
			}
		}
		for (; j < 32; j++) {
			theta = 16 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j]; beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 64];
			beta1 = B[i + j + 64] * y1 + B[i - j + 128] * y0;
			beta2 = B[i + j + 64] * y0 - B[i - j + 128] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 64] = alpha - beta1;
			B[i - j + 64] = beta + beta2;
			B[i - j + 128] = beta - beta2;
		}
	}

	/* stride-256 batched: theta = 8*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 2048; i += 256) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 128]; B[i] = ab + B[i + 128];
		A[i + 128] = aa - A[i + 128]; B[i + 128] = ab - B[i + 128];
		aa = A[i + 64]; ab = B[i + 64];
		A[i + 64] = aa + A[i + 192];
		B[i + 64] = ab + B[i + 192];
		A[i + 192] = aa - A[i + 192];
		B[i + 192] = ab - B[i + 192];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 60; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 8 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 512];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 128, 256, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 128, 256, j, y0v, y1v);
			}
		}
		for (; j < 64; j++) {
			theta = 8 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j]; beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 128];
			beta1 = B[i + j + 128] * y1 + B[i - j + 256] * y0;
			beta2 = B[i + j + 128] * y0 - B[i - j + 256] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 128] = alpha - beta1;
			B[i - j + 128] = beta + beta2;
			B[i - j + 256] = beta - beta2;
		}
	}

	/* stride-512 batched: theta = 4*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 2048; i += 512) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 256]; B[i] = ab + B[i + 256];
		A[i + 256] = aa - A[i + 256]; B[i + 256] = ab - B[i + 256];
		aa = A[i + 128]; ab = B[i + 128];
		A[i + 128] = aa + A[i + 384];
		B[i + 128] = ab + B[i + 384];
		A[i + 384] = aa - A[i + 384];
		B[i + 384] = ab - B[i + 384];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 124; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 4 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 512];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 256, 512, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 256, 512, j, y0v, y1v);
			}
		}
		for (; j < 128; j++) {
			theta = 4 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j]; beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 256];
			beta1 = B[i + j + 256] * y1 + B[i - j + 512] * y0;
			beta2 = B[i + j + 256] * y0 - B[i - j + 512] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 256] = alpha - beta1;
			B[i - j + 256] = beta + beta2;
			B[i - j + 512] = beta - beta2;
		}
	}

	/* stride-1024 batched: theta = 2*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 2048; i += 1024) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 512]; B[i] = ab + B[i + 512];
		A[i + 512] = aa - A[i + 512]; B[i + 512] = ab - B[i + 512];
		aa = A[i + 256]; ab = B[i + 256];
		A[i + 256] = aa + A[i + 768];
		B[i + 256] = ab + B[i + 768];
		A[i + 768] = aa - A[i + 768];
		B[i + 768] = ab - B[i + 768];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 252; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 2 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 512];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 512, 1024, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 512, 1024, j, y0v, y1v);
			}
		}
		for (; j < 256; j++) {
			theta = 2 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 512];
			alpha = A[i + j]; beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 512];
			beta1 = B[i + j + 512] * y1 + B[i - j + 1024] * y0;
			beta2 = B[i + j + 512] * y0 - B[i - j + 1024] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 512] = alpha - beta1;
			B[i - j + 512] = beta + beta2;
			B[i - j + 1024] = beta - beta2;
		}
	}

	/* final stage: theta == j, forward twiddle loads (see DFT1024_x2). */
	aa = A[0]; ab = B[0];
	A[0] = aa + A[1024]; B[0] = ab + B[1024];
	A[1024] = aa - A[1024]; B[1024] = ab - B[1024];
	aa = A[512]; ab = B[512];
	A[512] = aa + A[1536]; B[512] = ab + B[1536];
	A[1536] = aa - A[1536]; B[1536] = ab - B[1536];
	{
		svbool_t pg = svptrue_b32();
		for (j = 1; j + 3 <= 508; j += 4) {
			svfloat32_t y0v = svld1_f32(pg, &sinTab[j]);
			svfloat32_t y1v = svld1_f32(pg, &sinTab[j + 512]);
			dft_twiddle_bfly4(A, 1024, 2048, j, y0v, y1v);
			dft_twiddle_bfly4(B, 1024, 2048, j, y0v, y1v);
		}
	}
	for (; j < 512; j++) {
		y0 = sinTab[j];
		y1 = sinTab[j + 512];
		alpha = A[j]; beta = A[-j + 1024];
		beta1 = A[j + 1024] * y1 + A[-j + 2048] * y0;
		beta2 = A[j + 1024] * y0 - A[-j + 2048] * y1;
		A[j] = alpha + beta1;
		A[j + 1024] = alpha - beta1;
		A[-j + 1024] = beta + beta2;
		A[-j + 2048] = beta - beta2;
		alpha = B[j]; beta = B[-j + 1024];
		beta1 = B[j + 1024] * y1 + B[-j + 2048] * y0;
		beta2 = B[j + 1024] * y0 - B[-j + 2048] * y1;
		B[j] = alpha + beta1;
		B[j + 1024] = alpha - beta1;
		B[-j + 1024] = beta + beta2;
		B[-j + 2048] = beta - beta2;
	}
}

void DFT4096_x2(float *A, float *B, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, aa, ab;

	/* stride-4 -- plain scalar, deliberately not SVE2. See DFT1024_x2's
	 * comment: vectorizing a 4-wide radix-4 butterfly on 128-bit SVE2 costs
	 * more instructions than scalar (shuffle network, no spare parallelism
	 * to pay for it), confirmed via simpleperf on real hardware. */
	for (i = 0; i < 4096; i += 4)
	{
		float a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
		float b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
		float ay0 = a0 + a1, ay1 = a0 - a1, ay2 = a2 + a3, ay3 = a2 - a3;
		float by0 = b0 + b1, by1 = b0 - b1, by2 = b2 + b3, by3 = b2 - b3;
		A[i] = ay0 + ay2; A[i + 2] = ay0 - ay2;
		A[i + 1] = ay1 + ay3; A[i + 3] = ay1 - ay3;
		B[i] = by0 + by2; B[i + 2] = by0 - by2;
		B[i + 1] = by1 + by3; B[i + 3] = by1 - by3;
	}

	/* stride-8 batched */
	for (i = 0; i < 4096; i += 8) {
		float a1 = A[i], a2 = A[i + 2], a3 = A[i + 1], a4 = A[i + 3];
		float a5 = A[i + 5], a7 = A[i + 7];
		float b1 = B[i], b2 = B[i + 2], b3 = B[i + 1], b4 = B[i + 3];
		float b5 = B[i + 5], b7 = B[i + 7];
		float a_b1 = 0.70710678118654752440084436210485f * (a5 + a7);
		float a_b2 = 0.70710678118654752440084436210485f * (a5 - a7);
		float b_b1 = 0.70710678118654752440084436210485f * (b5 + b7);
		float b_b2 = 0.70710678118654752440084436210485f * (b5 - b7);
		A[i] = a1 + A[i + 4]; A[i + 4] = a1 - A[i + 4];
		B[i] = b1 + B[i + 4]; B[i + 4] = b1 - B[i + 4];
		A[i + 2] = a2 + A[i + 6]; A[i + 6] = a2 - A[i + 6];
		B[i + 2] = b2 + B[i + 6]; B[i + 6] = b2 - B[i + 6];
		A[i + 1] = a3 + a_b1; A[i + 5] = a3 - a_b1;
		B[i + 1] = b3 + b_b1; B[i + 5] = b3 - b_b1;
		A[i + 3] = a4 + a_b2; A[i + 7] = a4 - a_b2;
		B[i + 3] = b4 + b_b2; B[i + 7] = b4 - b_b2;
	}

	/* stride-16 batched */
	for (i = 0; i < 4096; i += 16) {
		theta = 256;
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 8]; B[i] = ab + B[i + 8];
		A[i + 8] = aa - A[i + 8]; B[i + 8] = ab - B[i + 8];
		aa = A[i + 4]; ab = B[i + 4];
		A[i + 4] = aa + A[i + 12];
		B[i + 4] = ab + B[i + 12];
		A[i + 12] = aa - A[i + 12];
		B[i + 12] = ab - B[i + 12];
		for (j = 1; j < 4; j++) {
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j]; beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 256;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 8];
			beta1 = B[i + j + 8] * y1 + B[i - j + 16] * y0;
			beta2 = B[i + j + 8] * y0 - B[i - j + 16] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 8] = alpha - beta1;
			B[i - j + 8] = beta + beta2;
			B[i - j + 16] = beta - beta2;
		}
	}

	/* stride-32 batched: theta = 128*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 4096; i += 32) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 16]; B[i] = ab + B[i + 16];
		A[i + 16] = aa - A[i + 16]; B[i + 16] = ab - B[i + 16];
		aa = A[i + 8]; ab = B[i + 8];
		A[i + 8] = aa + A[i + 24];
		B[i + 8] = ab + B[i + 24];
		A[i + 24] = aa - A[i + 24];
		B[i + 24] = ab - B[i + 24];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 4; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 128 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 1024];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 16, 32, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 16, 32, j, y0v, y1v);
			}
		}
		for (; j < 8; j++) {
			theta = 128 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j]; beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 16];
			beta1 = B[i + j + 16] * y1 + B[i - j + 32] * y0;
			beta2 = B[i + j + 16] * y0 - B[i - j + 32] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 16] = alpha - beta1;
			B[i - j + 16] = beta + beta2;
			B[i - j + 32] = beta - beta2;
		}
	}

	/* stride-64 batched: theta = 64*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 4096; i += 64) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 32]; B[i] = ab + B[i + 32];
		A[i + 32] = aa - A[i + 32]; B[i + 32] = ab - B[i + 32];
		aa = A[i + 16]; ab = B[i + 16];
		A[i + 16] = aa + A[i + 48];
		B[i + 16] = ab + B[i + 48];
		A[i + 48] = aa - A[i + 48];
		B[i + 48] = ab - B[i + 48];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 12; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 64 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 1024];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 32, 64, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 32, 64, j, y0v, y1v);
			}
		}
		for (; j < 16; j++) {
			theta = 64 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j]; beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 32];
			beta1 = B[i + j + 32] * y1 + B[i - j + 64] * y0;
			beta2 = B[i + j + 32] * y0 - B[i - j + 64] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 32] = alpha - beta1;
			B[i - j + 32] = beta + beta2;
			B[i - j + 64] = beta - beta2;
		}
	}

	/* stride-128 batched: theta = 32*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 4096; i += 128) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 64]; B[i] = ab + B[i + 64];
		A[i + 64] = aa - A[i + 64]; B[i + 64] = ab - B[i + 64];
		aa = A[i + 32]; ab = B[i + 32];
		A[i + 32] = aa + A[i + 96];
		B[i + 32] = ab + B[i + 96];
		A[i + 96] = aa - A[i + 96];
		B[i + 96] = ab - B[i + 96];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 28; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 32 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 1024];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 64, 128, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 64, 128, j, y0v, y1v);
			}
		}
		for (; j < 32; j++) {
			theta = 32 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j]; beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 64];
			beta1 = B[i + j + 64] * y1 + B[i - j + 128] * y0;
			beta2 = B[i + j + 64] * y0 - B[i - j + 128] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 64] = alpha - beta1;
			B[i - j + 64] = beta + beta2;
			B[i - j + 128] = beta - beta2;
		}
	}

	/* stride-256 batched: theta = 16*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 4096; i += 256) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 128]; B[i] = ab + B[i + 128];
		A[i + 128] = aa - A[i + 128]; B[i + 128] = ab - B[i + 128];
		aa = A[i + 64]; ab = B[i + 64];
		A[i + 64] = aa + A[i + 192];
		B[i + 64] = ab + B[i + 192];
		A[i + 192] = aa - A[i + 192];
		B[i + 192] = ab - B[i + 192];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 60; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 16 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 1024];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 128, 256, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 128, 256, j, y0v, y1v);
			}
		}
		for (; j < 64; j++) {
			theta = 16 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j]; beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 128];
			beta1 = B[i + j + 128] * y1 + B[i - j + 256] * y0;
			beta2 = B[i + j + 128] * y0 - B[i - j + 256] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 128] = alpha - beta1;
			B[i - j + 128] = beta + beta2;
			B[i - j + 256] = beta - beta2;
		}
	}

	/* stride-512 batched: theta = 8*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 4096; i += 512) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 256]; B[i] = ab + B[i + 256];
		A[i + 256] = aa - A[i + 256]; B[i + 256] = ab - B[i + 256];
		aa = A[i + 128]; ab = B[i + 128];
		A[i + 128] = aa + A[i + 384];
		B[i + 128] = ab + B[i + 384];
		A[i + 384] = aa - A[i + 384];
		B[i + 384] = ab - B[i + 384];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 124; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 8 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 1024];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 256, 512, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 256, 512, j, y0v, y1v);
			}
		}
		for (; j < 128; j++) {
			theta = 8 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j]; beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 256];
			beta1 = B[i + j + 256] * y1 + B[i - j + 512] * y0;
			beta2 = B[i + j + 256] * y0 - B[i - j + 512] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 256] = alpha - beta1;
			B[i - j + 256] = beta + beta2;
			B[i - j + 512] = beta - beta2;
		}
	}

	/* stride-1024 batched: theta = 4*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 4096; i += 1024) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 512]; B[i] = ab + B[i + 512];
		A[i + 512] = aa - A[i + 512]; B[i + 512] = ab - B[i + 512];
		aa = A[i + 256]; ab = B[i + 256];
		A[i + 256] = aa + A[i + 768];
		B[i + 256] = ab + B[i + 768];
		A[i + 768] = aa - A[i + 768];
		B[i + 768] = ab - B[i + 768];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 252; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 4 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 1024];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 512, 1024, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 512, 1024, j, y0v, y1v);
			}
		}
		for (; j < 256; j++) {
			theta = 4 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j]; beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 512];
			beta1 = B[i + j + 512] * y1 + B[i - j + 1024] * y0;
			beta2 = B[i + j + 512] * y0 - B[i - j + 1024] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 512] = alpha - beta1;
			B[i - j + 512] = beta + beta2;
			B[i - j + 1024] = beta - beta2;
		}
	}

	/* stride-2048 batched: theta = 2*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 4096; i += 2048) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 1024]; B[i] = ab + B[i + 1024];
		A[i + 1024] = aa - A[i + 1024]; B[i + 1024] = ab - B[i + 1024];
		aa = A[i + 512]; ab = B[i + 512];
		A[i + 512] = aa + A[i + 1536];
		B[i + 512] = ab + B[i + 1536];
		A[i + 1536] = aa - A[i + 1536];
		B[i + 1536] = ab - B[i + 1536];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 508; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 2 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 1024];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 1024, 2048, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 1024, 2048, j, y0v, y1v);
			}
		}
		for (; j < 512; j++) {
			theta = 2 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 1024];
			alpha = A[i + j]; beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 1024];
			beta1 = B[i + j + 1024] * y1 + B[i - j + 2048] * y0;
			beta2 = B[i + j + 1024] * y0 - B[i - j + 2048] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 1024] = alpha - beta1;
			B[i - j + 1024] = beta + beta2;
			B[i - j + 2048] = beta - beta2;
		}
	}

	/* final stage: theta == j, forward twiddle loads (see DFT1024_x2). */
	aa = A[0]; ab = B[0];
	A[0] = aa + A[2048]; B[0] = ab + B[2048];
	A[2048] = aa - A[2048]; B[2048] = ab - B[2048];
	aa = A[1024]; ab = B[1024];
	A[1024] = aa + A[3072]; B[1024] = ab + B[3072];
	A[3072] = aa - A[3072]; B[3072] = ab - B[3072];
	{
		svbool_t pg = svptrue_b32();
		for (j = 1; j + 3 <= 1020; j += 4) {
			svfloat32_t y0v = svld1_f32(pg, &sinTab[j]);
			svfloat32_t y1v = svld1_f32(pg, &sinTab[j + 1024]);
			dft_twiddle_bfly4(A, 2048, 4096, j, y0v, y1v);
			dft_twiddle_bfly4(B, 2048, 4096, j, y0v, y1v);
		}
	}
	for (; j < 1024; j++) {
		y0 = sinTab[j];
		y1 = sinTab[j + 1024];
		alpha = A[j]; beta = A[-j + 2048];
		beta1 = A[j + 2048] * y1 + A[-j + 4096] * y0;
		beta2 = A[j + 2048] * y0 - A[-j + 4096] * y1;
		A[j] = alpha + beta1;
		A[j + 2048] = alpha - beta1;
		A[-j + 2048] = beta + beta2;
		A[-j + 4096] = beta - beta2;
		alpha = B[j]; beta = B[-j + 2048];
		beta1 = B[j + 2048] * y1 + B[-j + 4096] * y0;
		beta2 = B[j + 2048] * y0 - B[-j + 4096] * y1;
		B[j] = alpha + beta1;
		B[j + 2048] = alpha - beta1;
		B[-j + 2048] = beta + beta2;
		B[-j + 4096] = beta - beta2;
	}
}

void DFT8192_x2(float *A, float *B, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, aa, ab;

	/* stride-4 -- plain scalar, deliberately not SVE2. See DFT1024_x2's
	 * comment: vectorizing a 4-wide radix-4 butterfly on 128-bit SVE2 costs
	 * more instructions than scalar (shuffle network, no spare parallelism
	 * to pay for it), confirmed via simpleperf on real hardware. */
	for (i = 0; i < 8192; i += 4)
	{
		float a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
		float b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
		float ay0 = a0 + a1, ay1 = a0 - a1, ay2 = a2 + a3, ay3 = a2 - a3;
		float by0 = b0 + b1, by1 = b0 - b1, by2 = b2 + b3, by3 = b2 - b3;
		A[i] = ay0 + ay2; A[i + 2] = ay0 - ay2;
		A[i + 1] = ay1 + ay3; A[i + 3] = ay1 - ay3;
		B[i] = by0 + by2; B[i + 2] = by0 - by2;
		B[i + 1] = by1 + by3; B[i + 3] = by1 - by3;
	}

	/* stride-8 batched */
	for (i = 0; i < 8192; i += 8) {
		float a1 = A[i], a2 = A[i + 2], a3 = A[i + 1], a4 = A[i + 3];
		float a5 = A[i + 5], a7 = A[i + 7];
		float b1 = B[i], b2 = B[i + 2], b3 = B[i + 1], b4 = B[i + 3];
		float b5 = B[i + 5], b7 = B[i + 7];
		float a_b1 = 0.70710678118654752440084436210485f * (a5 + a7);
		float a_b2 = 0.70710678118654752440084436210485f * (a5 - a7);
		float b_b1 = 0.70710678118654752440084436210485f * (b5 + b7);
		float b_b2 = 0.70710678118654752440084436210485f * (b5 - b7);
		A[i] = a1 + A[i + 4]; A[i + 4] = a1 - A[i + 4];
		B[i] = b1 + B[i + 4]; B[i + 4] = b1 - B[i + 4];
		A[i + 2] = a2 + A[i + 6]; A[i + 6] = a2 - A[i + 6];
		B[i + 2] = b2 + B[i + 6]; B[i + 6] = b2 - B[i + 6];
		A[i + 1] = a3 + a_b1; A[i + 5] = a3 - a_b1;
		B[i + 1] = b3 + b_b1; B[i + 5] = b3 - b_b1;
		A[i + 3] = a4 + a_b2; A[i + 7] = a4 - a_b2;
		B[i + 3] = b4 + b_b2; B[i + 7] = b4 - b_b2;
	}

	/* stride-16 batched */
	for (i = 0; i < 8192; i += 16) {
		theta = 512;
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 8]; B[i] = ab + B[i + 8];
		A[i + 8] = aa - A[i + 8]; B[i + 8] = ab - B[i + 8];
		aa = A[i + 4]; ab = B[i + 4];
		A[i + 4] = aa + A[i + 12];
		B[i + 4] = ab + B[i + 12];
		A[i + 12] = aa - A[i + 12];
		B[i + 12] = ab - B[i + 12];
		for (j = 1; j < 4; j++) {
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 512;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 8];
			beta1 = B[i + j + 8] * y1 + B[i - j + 16] * y0;
			beta2 = B[i + j + 8] * y0 - B[i - j + 16] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 8] = alpha - beta1;
			B[i - j + 8] = beta + beta2;
			B[i - j + 16] = beta - beta2;
		}
	}

	/* stride-32 batched: theta = 256*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 8192; i += 32) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 16]; B[i] = ab + B[i + 16];
		A[i + 16] = aa - A[i + 16]; B[i + 16] = ab - B[i + 16];
		aa = A[i + 8]; ab = B[i + 8];
		A[i + 8] = aa + A[i + 24];
		B[i + 8] = ab + B[i + 24];
		A[i + 24] = aa - A[i + 24];
		B[i + 24] = ab - B[i + 24];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 4; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 256 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 2048];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 16, 32, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 16, 32, j, y0v, y1v);
			}
		}
		for (; j < 8; j++) {
			theta = 256 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 16];
			beta1 = B[i + j + 16] * y1 + B[i - j + 32] * y0;
			beta2 = B[i + j + 16] * y0 - B[i - j + 32] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 16] = alpha - beta1;
			B[i - j + 16] = beta + beta2;
			B[i - j + 32] = beta - beta2;
		}
	}

	/* stride-64 batched: theta = 128*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 8192; i += 64) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 32]; B[i] = ab + B[i + 32];
		A[i + 32] = aa - A[i + 32]; B[i + 32] = ab - B[i + 32];
		aa = A[i + 16]; ab = B[i + 16];
		A[i + 16] = aa + A[i + 48];
		B[i + 16] = ab + B[i + 48];
		A[i + 48] = aa - A[i + 48];
		B[i + 48] = ab - B[i + 48];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 12; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 128 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 2048];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 32, 64, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 32, 64, j, y0v, y1v);
			}
		}
		for (; j < 16; j++) {
			theta = 128 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 32];
			beta1 = B[i + j + 32] * y1 + B[i - j + 64] * y0;
			beta2 = B[i + j + 32] * y0 - B[i - j + 64] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 32] = alpha - beta1;
			B[i - j + 32] = beta + beta2;
			B[i - j + 64] = beta - beta2;
		}
	}

	/* stride-128 batched: theta = 64*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 8192; i += 128) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 64]; B[i] = ab + B[i + 64];
		A[i + 64] = aa - A[i + 64]; B[i + 64] = ab - B[i + 64];
		aa = A[i + 32]; ab = B[i + 32];
		A[i + 32] = aa + A[i + 96];
		B[i + 32] = ab + B[i + 96];
		A[i + 96] = aa - A[i + 96];
		B[i + 96] = ab - B[i + 96];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 28; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 64 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 2048];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 64, 128, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 64, 128, j, y0v, y1v);
			}
		}
		for (; j < 32; j++) {
			theta = 64 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 64];
			beta1 = B[i + j + 64] * y1 + B[i - j + 128] * y0;
			beta2 = B[i + j + 64] * y0 - B[i - j + 128] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 64] = alpha - beta1;
			B[i - j + 64] = beta + beta2;
			B[i - j + 128] = beta - beta2;
		}
	}

	/* stride-256 batched: theta = 32*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 8192; i += 256) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 128]; B[i] = ab + B[i + 128];
		A[i + 128] = aa - A[i + 128]; B[i + 128] = ab - B[i + 128];
		aa = A[i + 64]; ab = B[i + 64];
		A[i + 64] = aa + A[i + 192];
		B[i + 64] = ab + B[i + 192];
		A[i + 192] = aa - A[i + 192];
		B[i + 192] = ab - B[i + 192];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 60; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 32 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 2048];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 128, 256, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 128, 256, j, y0v, y1v);
			}
		}
		for (; j < 64; j++) {
			theta = 32 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 128];
			beta1 = B[i + j + 128] * y1 + B[i - j + 256] * y0;
			beta2 = B[i + j + 128] * y0 - B[i - j + 256] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 128] = alpha - beta1;
			B[i - j + 128] = beta + beta2;
			B[i - j + 256] = beta - beta2;
		}
	}

	/* stride-512 batched: theta = 16*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 8192; i += 512) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 256]; B[i] = ab + B[i + 256];
		A[i + 256] = aa - A[i + 256]; B[i + 256] = ab - B[i + 256];
		aa = A[i + 128]; ab = B[i + 128];
		A[i + 128] = aa + A[i + 384];
		B[i + 128] = ab + B[i + 384];
		A[i + 384] = aa - A[i + 384];
		B[i + 384] = ab - B[i + 384];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 124; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 16 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 2048];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 256, 512, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 256, 512, j, y0v, y1v);
			}
		}
		for (; j < 128; j++) {
			theta = 16 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 256];
			beta1 = B[i + j + 256] * y1 + B[i - j + 512] * y0;
			beta2 = B[i + j + 256] * y0 - B[i - j + 512] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 256] = alpha - beta1;
			B[i - j + 256] = beta + beta2;
			B[i - j + 512] = beta - beta2;
		}
	}

	/* stride-1024 batched: theta = 8*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 8192; i += 1024) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 512]; B[i] = ab + B[i + 512];
		A[i + 512] = aa - A[i + 512]; B[i + 512] = ab - B[i + 512];
		aa = A[i + 256]; ab = B[i + 256];
		A[i + 256] = aa + A[i + 768];
		B[i + 256] = ab + B[i + 768];
		A[i + 768] = aa - A[i + 768];
		B[i + 768] = ab - B[i + 768];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 252; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 8 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 2048];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 512, 1024, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 512, 1024, j, y0v, y1v);
			}
		}
		for (; j < 256; j++) {
			theta = 8 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 512];
			beta1 = B[i + j + 512] * y1 + B[i - j + 1024] * y0;
			beta2 = B[i + j + 512] * y0 - B[i - j + 1024] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 512] = alpha - beta1;
			B[i - j + 512] = beta + beta2;
			B[i - j + 1024] = beta - beta2;
		}
	}

	/* stride-2048 batched: theta = 4*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 8192; i += 2048) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 1024]; B[i] = ab + B[i + 1024];
		A[i + 1024] = aa - A[i + 1024]; B[i + 1024] = ab - B[i + 1024];
		aa = A[i + 512]; ab = B[i + 512];
		A[i + 512] = aa + A[i + 1536];
		B[i + 512] = ab + B[i + 1536];
		A[i + 1536] = aa - A[i + 1536];
		B[i + 1536] = ab - B[i + 1536];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 508; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 4 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 2048];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 1024, 2048, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 1024, 2048, j, y0v, y1v);
			}
		}
		for (; j < 512; j++) {
			theta = 4 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 1024];
			beta1 = B[i + j + 1024] * y1 + B[i - j + 2048] * y0;
			beta2 = B[i + j + 1024] * y0 - B[i - j + 2048] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 1024] = alpha - beta1;
			B[i - j + 1024] = beta + beta2;
			B[i - j + 2048] = beta - beta2;
		}
	}

	/* stride-4096 batched: theta = 2*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 8192; i += 4096) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 2048]; B[i] = ab + B[i + 2048];
		A[i + 2048] = aa - A[i + 2048]; B[i + 2048] = ab - B[i + 2048];
		aa = A[i + 1024]; ab = B[i + 1024];
		A[i + 1024] = aa + A[i + 3072];
		B[i + 1024] = ab + B[i + 3072];
		A[i + 3072] = aa - A[i + 3072];
		B[i + 3072] = ab - B[i + 3072];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 1020; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 2 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 2048];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 2048, 4096, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 2048, 4096, j, y0v, y1v);
			}
		}
		for (; j < 1024; j++) {
			theta = 2 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 2048];
			alpha = A[i + j]; beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 2048];
			beta1 = B[i + j + 2048] * y1 + B[i - j + 4096] * y0;
			beta2 = B[i + j + 2048] * y0 - B[i - j + 4096] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 2048] = alpha - beta1;
			B[i - j + 2048] = beta + beta2;
			B[i - j + 4096] = beta - beta2;
		}
	}

	/* final stage: theta == j, forward twiddle loads (see DFT1024_x2). */
	aa = A[0]; ab = B[0];
	A[0] = aa + A[4096]; B[0] = ab + B[4096];
	A[4096] = aa - A[4096]; B[4096] = ab - B[4096];
	aa = A[2048]; ab = B[2048];
	A[2048] = aa + A[6144]; B[2048] = ab + B[6144];
	A[6144] = aa - A[6144]; B[6144] = ab - B[6144];
	{
		svbool_t pg = svptrue_b32();
		for (j = 1; j + 3 <= 2044; j += 4) {
			svfloat32_t y0v = svld1_f32(pg, &sinTab[j]);
			svfloat32_t y1v = svld1_f32(pg, &sinTab[j + 2048]);
			dft_twiddle_bfly4(A, 4096, 8192, j, y0v, y1v);
			dft_twiddle_bfly4(B, 4096, 8192, j, y0v, y1v);
		}
	}
	for (; j < 2048; j++) {
		y0 = sinTab[j];
		y1 = sinTab[j + 2048];
		alpha = A[j]; beta = A[-j + 4096];
		beta1 = A[j + 4096] * y1 + A[-j + 8192] * y0;
		beta2 = A[j + 4096] * y0 - A[-j + 8192] * y1;
		A[j] = alpha + beta1;
		A[j + 4096] = alpha - beta1;
		A[-j + 4096] = beta + beta2;
		A[-j + 8192] = beta - beta2;
		alpha = B[j]; beta = B[-j + 4096];
		beta1 = B[j + 4096] * y1 + B[-j + 8192] * y0;
		beta2 = B[j + 4096] * y0 - B[-j + 8192] * y1;
		B[j] = alpha + beta1;
		B[j + 4096] = alpha - beta1;
		B[-j + 4096] = beta + beta2;
		B[-j + 8192] = beta - beta2;
	}
}

void DFT16384_x2(float *A, float *B, const float *sinTab)
{
	int i, j, theta;
	float alpha, beta, beta1, beta2, y0, y1, aa, ab;

	/* stride-4 -- plain scalar, deliberately not SVE2. See DFT1024_x2's
	 * comment: vectorizing a 4-wide radix-4 butterfly on 128-bit SVE2 costs
	 * more instructions than scalar (shuffle network, no spare parallelism
	 * to pay for it), confirmed via simpleperf on real hardware. */
	for (i = 0; i < 16384; i += 4)
	{
		float a0 = A[i], a1 = A[i + 1], a2 = A[i + 2], a3 = A[i + 3];
		float b0 = B[i], b1 = B[i + 1], b2 = B[i + 2], b3 = B[i + 3];
		float ay0 = a0 + a1, ay1 = a0 - a1, ay2 = a2 + a3, ay3 = a2 - a3;
		float by0 = b0 + b1, by1 = b0 - b1, by2 = b2 + b3, by3 = b2 - b3;
		A[i] = ay0 + ay2; A[i + 2] = ay0 - ay2;
		A[i + 1] = ay1 + ay3; A[i + 3] = ay1 - ay3;
		B[i] = by0 + by2; B[i + 2] = by0 - by2;
		B[i + 1] = by1 + by3; B[i + 3] = by1 - by3;
	}

	/* stride-8 batched */
	for (i = 0; i < 16384; i += 8) {
		float a1 = A[i], a2 = A[i + 2], a3 = A[i + 1], a4 = A[i + 3];
		float a5 = A[i + 5], a7 = A[i + 7];
		float b1 = B[i], b2 = B[i + 2], b3 = B[i + 1], b4 = B[i + 3];
		float b5 = B[i + 5], b7 = B[i + 7];
		float a_b1 = 0.70710678118654752440084436210485f * (a5 + a7);
		float a_b2 = 0.70710678118654752440084436210485f * (a5 - a7);
		float b_b1 = 0.70710678118654752440084436210485f * (b5 + b7);
		float b_b2 = 0.70710678118654752440084436210485f * (b5 - b7);
		A[i] = a1 + A[i + 4]; A[i + 4] = a1 - A[i + 4];
		B[i] = b1 + B[i + 4]; B[i + 4] = b1 - B[i + 4];
		A[i + 2] = a2 + A[i + 6]; A[i + 6] = a2 - A[i + 6];
		B[i + 2] = b2 + B[i + 6]; B[i + 6] = b2 - B[i + 6];
		A[i + 1] = a3 + a_b1; A[i + 5] = a3 - a_b1;
		B[i + 1] = b3 + b_b1; B[i + 5] = b3 - b_b1;
		A[i + 3] = a4 + a_b2; A[i + 7] = a4 - a_b2;
		B[i + 3] = b4 + b_b2; B[i + 7] = b4 - b_b2;
	}

	/* stride-16 batched */
	for (i = 0; i < 16384; i += 16) {
		theta = 1024;
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 8]; B[i] = ab + B[i + 8];
		A[i + 8] = aa - A[i + 8]; B[i + 8] = ab - B[i + 8];
		aa = A[i + 4]; ab = B[i + 4];
		A[i + 4] = aa + A[i + 12];
		B[i + 4] = ab + B[i + 12];
		A[i + 12] = aa - A[i + 12];
		B[i + 12] = ab - B[i + 12];
		for (j = 1; j < 4; j++) {
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 8];
			beta1 = A[i + j + 8] * y1 + A[i - j + 16] * y0;
			beta2 = A[i + j + 8] * y0 - A[i - j + 16] * y1;
			theta += 1024;
			A[i + j] = alpha + beta1;
			A[i + j + 8] = alpha - beta1;
			A[i - j + 8] = beta + beta2;
			A[i - j + 16] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 8];
			beta1 = B[i + j + 8] * y1 + B[i - j + 16] * y0;
			beta2 = B[i + j + 8] * y0 - B[i - j + 16] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 8] = alpha - beta1;
			B[i - j + 8] = beta + beta2;
			B[i - j + 16] = beta - beta2;
		}
	}

	/* stride-32 batched: theta = 512*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 32) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 16]; B[i] = ab + B[i + 16];
		A[i + 16] = aa - A[i + 16]; B[i + 16] = ab - B[i + 16];
		aa = A[i + 8]; ab = B[i + 8];
		A[i + 8] = aa + A[i + 24];
		B[i + 8] = ab + B[i + 24];
		A[i + 24] = aa - A[i + 24];
		B[i + 24] = ab - B[i + 24];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 4; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 512 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 16, 32, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 16, 32, j, y0v, y1v);
			}
		}
		for (; j < 8; j++) {
			theta = 512 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 16];
			beta1 = A[i + j + 16] * y1 + A[i - j + 32] * y0;
			beta2 = A[i + j + 16] * y0 - A[i - j + 32] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 16] = alpha - beta1;
			A[i - j + 16] = beta + beta2;
			A[i - j + 32] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 16];
			beta1 = B[i + j + 16] * y1 + B[i - j + 32] * y0;
			beta2 = B[i + j + 16] * y0 - B[i - j + 32] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 16] = alpha - beta1;
			B[i - j + 16] = beta + beta2;
			B[i - j + 32] = beta - beta2;
		}
	}

	/* stride-64 batched: theta = 256*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 64) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 32]; B[i] = ab + B[i + 32];
		A[i + 32] = aa - A[i + 32]; B[i + 32] = ab - B[i + 32];
		aa = A[i + 16]; ab = B[i + 16];
		A[i + 16] = aa + A[i + 48];
		B[i + 16] = ab + B[i + 48];
		A[i + 48] = aa - A[i + 48];
		B[i + 48] = ab - B[i + 48];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 12; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 256 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 32, 64, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 32, 64, j, y0v, y1v);
			}
		}
		for (; j < 16; j++) {
			theta = 256 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 32];
			beta1 = A[i + j + 32] * y1 + A[i - j + 64] * y0;
			beta2 = A[i + j + 32] * y0 - A[i - j + 64] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 32] = alpha - beta1;
			A[i - j + 32] = beta + beta2;
			A[i - j + 64] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 32];
			beta1 = B[i + j + 32] * y1 + B[i - j + 64] * y0;
			beta2 = B[i + j + 32] * y0 - B[i - j + 64] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 32] = alpha - beta1;
			B[i - j + 32] = beta + beta2;
			B[i - j + 64] = beta - beta2;
		}
	}

	/* stride-128 batched: theta = 128*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 128) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 64]; B[i] = ab + B[i + 64];
		A[i + 64] = aa - A[i + 64]; B[i + 64] = ab - B[i + 64];
		aa = A[i + 32]; ab = B[i + 32];
		A[i + 32] = aa + A[i + 96];
		B[i + 32] = ab + B[i + 96];
		A[i + 96] = aa - A[i + 96];
		B[i + 96] = ab - B[i + 96];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 28; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 128 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 64, 128, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 64, 128, j, y0v, y1v);
			}
		}
		for (; j < 32; j++) {
			theta = 128 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 64];
			beta1 = A[i + j + 64] * y1 + A[i - j + 128] * y0;
			beta2 = A[i + j + 64] * y0 - A[i - j + 128] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 64] = alpha - beta1;
			A[i - j + 64] = beta + beta2;
			A[i - j + 128] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 64];
			beta1 = B[i + j + 64] * y1 + B[i - j + 128] * y0;
			beta2 = B[i + j + 64] * y0 - B[i - j + 128] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 64] = alpha - beta1;
			B[i - j + 64] = beta + beta2;
			B[i - j + 128] = beta - beta2;
		}
	}

	/* stride-256 batched: theta = 64*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 256) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 128]; B[i] = ab + B[i + 128];
		A[i + 128] = aa - A[i + 128]; B[i + 128] = ab - B[i + 128];
		aa = A[i + 64]; ab = B[i + 64];
		A[i + 64] = aa + A[i + 192];
		B[i + 64] = ab + B[i + 192];
		A[i + 192] = aa - A[i + 192];
		B[i + 192] = ab - B[i + 192];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 60; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 64 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 128, 256, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 128, 256, j, y0v, y1v);
			}
		}
		for (; j < 64; j++) {
			theta = 64 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 128];
			beta1 = A[i + j + 128] * y1 + A[i - j + 256] * y0;
			beta2 = A[i + j + 128] * y0 - A[i - j + 256] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 128] = alpha - beta1;
			A[i - j + 128] = beta + beta2;
			A[i - j + 256] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 128];
			beta1 = B[i + j + 128] * y1 + B[i - j + 256] * y0;
			beta2 = B[i + j + 128] * y0 - B[i - j + 256] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 128] = alpha - beta1;
			B[i - j + 128] = beta + beta2;
			B[i - j + 256] = beta - beta2;
		}
	}

	/* stride-512 batched: theta = 32*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 512) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 256]; B[i] = ab + B[i + 256];
		A[i + 256] = aa - A[i + 256]; B[i + 256] = ab - B[i + 256];
		aa = A[i + 128]; ab = B[i + 128];
		A[i + 128] = aa + A[i + 384];
		B[i + 128] = ab + B[i + 384];
		A[i + 384] = aa - A[i + 384];
		B[i + 384] = ab - B[i + 384];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 124; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 32 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 256, 512, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 256, 512, j, y0v, y1v);
			}
		}
		for (; j < 128; j++) {
			theta = 32 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 256];
			beta1 = A[i + j + 256] * y1 + A[i - j + 512] * y0;
			beta2 = A[i + j + 256] * y0 - A[i - j + 512] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 256] = alpha - beta1;
			A[i - j + 256] = beta + beta2;
			A[i - j + 512] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 256];
			beta1 = B[i + j + 256] * y1 + B[i - j + 512] * y0;
			beta2 = B[i + j + 256] * y0 - B[i - j + 512] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 256] = alpha - beta1;
			B[i - j + 256] = beta + beta2;
			B[i - j + 512] = beta - beta2;
		}
	}

	/* stride-1024 batched: theta = 16*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 1024) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 512]; B[i] = ab + B[i + 512];
		A[i + 512] = aa - A[i + 512]; B[i + 512] = ab - B[i + 512];
		aa = A[i + 256]; ab = B[i + 256];
		A[i + 256] = aa + A[i + 768];
		B[i + 256] = ab + B[i + 768];
		A[i + 768] = aa - A[i + 768];
		B[i + 768] = ab - B[i + 768];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 252; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 16 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 512, 1024, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 512, 1024, j, y0v, y1v);
			}
		}
		for (; j < 256; j++) {
			theta = 16 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 512];
			beta1 = A[i + j + 512] * y1 + A[i - j + 1024] * y0;
			beta2 = A[i + j + 512] * y0 - A[i - j + 1024] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 512] = alpha - beta1;
			A[i - j + 512] = beta + beta2;
			A[i - j + 1024] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 512];
			beta1 = B[i + j + 512] * y1 + B[i - j + 1024] * y0;
			beta2 = B[i + j + 512] * y0 - B[i - j + 1024] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 512] = alpha - beta1;
			B[i - j + 512] = beta + beta2;
			B[i - j + 1024] = beta - beta2;
		}
	}

	/* stride-2048 batched: theta = 8*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 2048) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 1024]; B[i] = ab + B[i + 1024];
		A[i + 1024] = aa - A[i + 1024]; B[i + 1024] = ab - B[i + 1024];
		aa = A[i + 512]; ab = B[i + 512];
		A[i + 512] = aa + A[i + 1536];
		B[i + 512] = ab + B[i + 1536];
		A[i + 1536] = aa - A[i + 1536];
		B[i + 1536] = ab - B[i + 1536];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 508; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 8 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 1024, 2048, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 1024, 2048, j, y0v, y1v);
			}
		}
		for (; j < 512; j++) {
			theta = 8 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 1024];
			beta1 = A[i + j + 1024] * y1 + A[i - j + 2048] * y0;
			beta2 = A[i + j + 1024] * y0 - A[i - j + 2048] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 1024] = alpha - beta1;
			A[i - j + 1024] = beta + beta2;
			A[i - j + 2048] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 1024];
			beta1 = B[i + j + 1024] * y1 + B[i - j + 2048] * y0;
			beta2 = B[i + j + 1024] * y0 - B[i - j + 2048] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 1024] = alpha - beta1;
			B[i - j + 1024] = beta + beta2;
			B[i - j + 2048] = beta - beta2;
		}
	}

	/* stride-4096 batched: theta = 4*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 4096) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 2048]; B[i] = ab + B[i + 2048];
		A[i + 2048] = aa - A[i + 2048]; B[i + 2048] = ab - B[i + 2048];
		aa = A[i + 1024]; ab = B[i + 1024];
		A[i + 1024] = aa + A[i + 3072];
		B[i + 1024] = ab + B[i + 3072];
		A[i + 3072] = aa - A[i + 3072];
		B[i + 3072] = ab - B[i + 3072];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 1020; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 4 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 2048, 4096, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 2048, 4096, j, y0v, y1v);
			}
		}
		for (; j < 1024; j++) {
			theta = 4 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 2048];
			beta1 = A[i + j + 2048] * y1 + A[i - j + 4096] * y0;
			beta2 = A[i + j + 2048] * y0 - A[i - j + 4096] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 2048] = alpha - beta1;
			A[i - j + 2048] = beta + beta2;
			A[i - j + 4096] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 2048];
			beta1 = B[i + j + 2048] * y1 + B[i - j + 4096] * y0;
			beta2 = B[i + j + 2048] * y0 - B[i - j + 4096] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 2048] = alpha - beta1;
			B[i - j + 2048] = beta + beta2;
			B[i - j + 4096] = beta - beta2;
		}
	}

	/* stride-8192 batched: theta = 2*j, scratch-gather twiddle load (see stride-64 in DFT1024_x2). */
	for (i = 0; i < 16384; i += 8192) {
		aa = A[i]; ab = B[i];
		A[i] = aa + A[i + 4096]; B[i] = ab + B[i + 4096];
		A[i + 4096] = aa - A[i + 4096]; B[i + 4096] = ab - B[i + 4096];
		aa = A[i + 2048]; ab = B[i + 2048];
		A[i + 2048] = aa + A[i + 6144];
		B[i + 2048] = ab + B[i + 6144];
		A[i + 6144] = aa - A[i + 6144];
		B[i + 6144] = ab - B[i + 6144];
		{
			svbool_t pg = svptrue_b32();
			for (j = 1; j + 3 <= 2044; j += 4) {
				float y0_tmp[4], y1_tmp[4];
				for (int k = 0; k < 4; k++) {
					int th = 2 * (j + k);
					y0_tmp[k] = sinTab[th];
					y1_tmp[k] = sinTab[th + 4096];
				}
				svfloat32_t y0v = svld1_f32(pg, y0_tmp);
				svfloat32_t y1v = svld1_f32(pg, y1_tmp);
				dft_twiddle_bfly4(A + i, 4096, 8192, j, y0v, y1v);
				dft_twiddle_bfly4(B + i, 4096, 8192, j, y0v, y1v);
			}
		}
		for (; j < 2048; j++) {
			theta = 2 * j;
			y0 = sinTab[theta];
			y1 = sinTab[theta + 4096];
			alpha = A[i + j]; beta = A[i - j + 4096];
			beta1 = A[i + j + 4096] * y1 + A[i - j + 8192] * y0;
			beta2 = A[i + j + 4096] * y0 - A[i - j + 8192] * y1;
			A[i + j] = alpha + beta1;
			A[i + j + 4096] = alpha - beta1;
			A[i - j + 4096] = beta + beta2;
			A[i - j + 8192] = beta - beta2;
			alpha = B[i + j]; beta = B[i - j + 4096];
			beta1 = B[i + j + 4096] * y1 + B[i - j + 8192] * y0;
			beta2 = B[i + j + 4096] * y0 - B[i - j + 8192] * y1;
			B[i + j] = alpha + beta1;
			B[i + j + 4096] = alpha - beta1;
			B[i - j + 4096] = beta + beta2;
			B[i - j + 8192] = beta - beta2;
		}
	}

	/* final stage: theta == j, forward twiddle loads (see DFT1024_x2). */
	aa = A[0]; ab = B[0];
	A[0] = aa + A[8192]; B[0] = ab + B[8192];
	A[8192] = aa - A[8192]; B[8192] = ab - B[8192];
	aa = A[4096]; ab = B[4096];
	A[4096] = aa + A[12288]; B[4096] = ab + B[12288];
	A[12288] = aa - A[12288]; B[12288] = ab - B[12288];
	{
		svbool_t pg = svptrue_b32();
		for (j = 1; j + 3 <= 4092; j += 4) {
			svfloat32_t y0v = svld1_f32(pg, &sinTab[j]);
			svfloat32_t y1v = svld1_f32(pg, &sinTab[j + 4096]);
			dft_twiddle_bfly4(A, 8192, 16384, j, y0v, y1v);
			dft_twiddle_bfly4(B, 8192, 16384, j, y0v, y1v);
		}
	}
	for (; j < 4096; j++) {
		y0 = sinTab[j];
		y1 = sinTab[j + 4096];
		alpha = A[j]; beta = A[-j + 8192];
		beta1 = A[j + 8192] * y1 + A[-j + 16384] * y0;
		beta2 = A[j + 8192] * y0 - A[-j + 16384] * y1;
		A[j] = alpha + beta1;
		A[j + 8192] = alpha - beta1;
		A[-j + 8192] = beta + beta2;
		A[-j + 16384] = beta - beta2;
		alpha = B[j]; beta = B[-j + 8192];
		beta1 = B[j + 8192] * y1 + B[-j + 16384] * y0;
		beta2 = B[j + 8192] * y0 - B[-j + 16384] * y1;
		B[j] = alpha + beta1;
		B[j + 8192] = alpha - beta1;
		B[-j + 8192] = beta + beta2;
		B[-j + 16384] = beta - beta2;
	}
}


#endif // JDSP_SVE2_FFT_X2
