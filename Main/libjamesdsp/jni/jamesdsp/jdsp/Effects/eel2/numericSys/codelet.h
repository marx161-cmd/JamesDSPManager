void DFT1024(float *A, const float *sinTab);
void DFT1048576(float *A, const float *sinTab);
void DFT128(float *A, const float *sinTab);
void DFT131072(float *A, const float *sinTab);
void DFT16(float *A, const float *sinTab);
void DFT16384(float *A, const float *sinTab);
void DFT2(float *A, const float *sinTab);
void DFT2048(float *A, const float *sinTab);
void DFT256(float *A, const float *sinTab);
void DFT262144(float *A, const float *sinTab);
void DFT32(float *A, const float *sinTab);
void DFT32768(float *A, const float *sinTab);
void DFT4(float *A, const float *sinTab);
void DFT4096(float *A, const float *sinTab);
void DFT512(float *A, const float *sinTab);
void DFT524288(float *A, const float *sinTab);
void DFT64(float *A, const float *sinTab);
void DFT65536(float *A, const float *sinTab);
void DFT8(float *A, const float *sinTab);
void DFT8192(float *A, const float *sinTab);
// Batched dual-buffer variants: SVE2, Tensor G5 (Pixel 10 Pro) only. This
// build targets armeabi-v7a/arm64-v8a/x86/x86_64; only arm64-v8a gets the
// -march=...+sve2 compile flag (see CMakeLists.txt), so these must not be
// declared -- let alone called -- on the other three, where the SVE2
// intrinsics wouldn't even compile. FFTConvolver.c's fft_x2 wiring is
// guarded by the same JDSP_SVE2_FFT_X2 check and always has a scalar
// fallback, so this isn't a functionality gap on those targets.
#if defined(JDSP_SVE2_FFT_X2)
void DFT256_x2(float *A, float *B, const float *sinTab);
void DFT512_x2(float *A, float *B, const float *sinTab);
void DFT1024_x2(float *A, float *B, const float *sinTab);
void DFT2048_x2(float *A, float *B, const float *sinTab);
void DFT4096_x2(float *A, float *B, const float *sinTab);
void DFT8192_x2(float *A, float *B, const float *sinTab);
void DFT16384_x2(float *A, float *B, const float *sinTab);
#endif
