#ifdef DEBUG
#define TAG "EffectDSPMain"
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG,__VA_ARGS__)
#include "MemoryUsage.h"
#endif
#include <string.h>
#include <math.h>
#include "essential.h"
// Effect section
#include "jdsp/jdsp_header.h"
#include "lua_bridge.h"
#include "lsp_wrap.h"
#include "lsp_wrap_expander.h"
#include "lsp_wrap_autogain.h"
typedef struct
{
	unsigned long long initializeForFirst;
	int32_t engineImpulseResponseHash;
	int32_t hashSlot[4];
	int mEnable;
	JamesDSPLib jdsp;
    float mSamplingRate;
    int formatFloatModeInt32Mode;
	char *stringEq;
	float *tempImpulseIncoming;
	float drcAtkRel[4];
	float boostingCon;
	int bbMaxGain;
	// Variables
	int numTime2Send, samplesInc, stringIndex;
	int16_t impChannels;
	int32_t impulseLengthActual, convolverNeedRefresh;
} EffectDSPMain;
typedef struct
{
	int32_t status;
	uint32_t psize;
	uint32_t vsize;
	int32_t cmd;
	int32_t data;
} reply1x4_1x4_t;

// Lua command dispatch — exposed to lua_bridge via set_command_handler().
// Handles module enable toggles (1201-1213) and param set commands in a
// simplified int+float format that skips the AIDL wire format.
static EffectDSPMain *g_lua_cmd_dspmain = NULL;

static int lua_cmd_dispatch(int cmd, const float *params, int nparams) {
    if (!g_lua_cmd_dspmain) return -1;
    EffectDSPMain *d = g_lua_cmd_dspmain;

    // Module enable toggles (1201-1213): params[0] = 0/1
    if (cmd >= 1201 && cmd <= 1213 && nparams >= 1) {
        int en = (int)params[0];
        switch (cmd) {
        case 1201: // bassboost
            if (en) BassBoostEnable(&d->jdsp);
            else BassBoostDisable(&d->jdsp);
            return 0;
        case 1202: // equalizer
            if (en) MultimodalEqualizerEnable(&d->jdsp, 1);
            else MultimodalEqualizerDisable(&d->jdsp);
            return 0;
        case 1203: // reverb
            if (en) ReverbEnable(&d->jdsp);
            else ReverbDisable(&d->jdsp);
            return 0;
        case 1204: // stereo
            if (en) StereoEnhancementEnable(&d->jdsp);
            else StereoEnhancementDisable(&d->jdsp);
            return 0;
        case 1205: // convolver
            if (en) Convolver1DEnable(&d->jdsp);
            else Convolver1DDisable(&d->jdsp);
            return 0;
        case 1206: // tube (analogue modelling)
            if (en) VacuumTubeEnable(&d->jdsp);
            else VacuumTubeDisable(&d->jdsp);
            return 0;
        case 1207: // autogain
            d->jdsp.autogainEnabled = en;
            return 0;
        case 1208: // crossfeed
            if (en) CrossfeedEnable(&d->jdsp, 1);
            else CrossfeedDisable(&d->jdsp);
            return 0;
        case 1210: // arbitrary response eq
            if (en) ArbitraryResponseEqualizerEnable(&d->jdsp, 1);
            else ArbitraryResponseEqualizerDisable(&d->jdsp);
            return 0;
        case 1212: // ddc
            if (en) DDCEnable(&d->jdsp, 1);
            else DDCDisable(&d->jdsp);
            return 0;
        case 1213: // liveprog
            if (en) LiveProgEnable(&d->jdsp);
            else LiveProgDisable(&d->jdsp);
            return 0;
        default: return -1;
        }
    }

    // Gate/Compressor (10100/10101). nparams==1 is enable-only: flip the flag
    // without touching parameters — the LSP wrapper's Bypass crossfade does
    // the actual on/off each block, and re-sending a fixed param preset here
    // would stomp the user's configured settings. nparams>=7 additionally
    // reconfigures the full {en, p1..p6} bundle for scripts that want to set
    // parameters explicitly.
    if (cmd == 10100 && nparams >= 1) {
        d->jdsp.gateEnabled = (int)params[0];
        if (nparams >= 7 && d->jdsp.gateEnabled) {
            lsp_gate_configure(d->jdsp.gate_handle, params[1], params[2], params[3], params[4], params[5], params[6]);
        }
        return 0;
    }
    if (cmd == 10101 && nparams >= 1) {
        d->jdsp.mbDynEnabled = (int)params[0];
        if (nparams >= 7 && d->jdsp.mbDynEnabled) {
            lsp_comp_configure(d->jdsp.compressor_handle, params[1], params[2], params[3], params[4], params[5], params[6]);
        }
        return 0;
    }
    // Expander (10103): {en[, mode, threshold_db, ratio, knee(linear), attack, release, hold]}
    if (cmd == 10103 && nparams >= 1) {
        d->jdsp.expanderEnabled = (int)params[0];
        if (nparams >= 8 && d->jdsp.expanderEnabled) {
            lsp_exp_configure(d->jdsp.expander_handle, (int)params[1], params[2], params[3], params[4], params[5], params[6], params[7]);
        }
        return 0;
    }
    // Autogain (10104): {en[, level_lufs, long_period_ms, short_period_ms,
    // drift_db, silence_lufs, max_gain_db, max_gain_on]}. Enable flag is the
    // same autogainEnabled that legacy cmd 1207 sets.
    if (cmd == 10104 && nparams >= 1) {
        d->jdsp.autogainEnabled = (int)params[0];
        if (nparams >= 8 && d->jdsp.autogainEnabled) {
            lsp_autogain_configure(d->jdsp.autogain_handle, params[1], params[2], params[3], params[4], params[5], params[6], (int)params[7]);
        }
        return 0;
    }

    // Tube drive (150): params[0] = drive (0.0 = off)
    if (cmd == 150 && nparams >= 1) {
        VacuumTubeSetGain(&d->jdsp, params[0]);
        return 0;
    }

    // Limiter (1500): params = {threshold, attack, release, knee, postgain[, enabled]}
    if (cmd == 1500 && nparams >= 5) {
        if (nparams >= 6) d->jdsp.limiterEnabled = (params[5] >= 0.5f) ? 1 : 0;
        lsp_limiter_configure(d->jdsp.limiter_handle, params[0], params[1], params[2], params[3]);
        // dB in, like every other postgain path (was stored raw as a linear
        // multiplier here -- review 2026-10-21).
        JamesDSPSetPostGain(&d->jdsp, params[4]);
        return 0;
    }

    return -1; // unknown command
}

// ============================================================================
// Lua property accessors — per-parameter control (write side)
// ============================================================================
// Registered via lua_bridge_register_property() in EffectDSPMainConstructor.
// The property ABI is context-free (double get(void) / int set(double)), so
// these close over the file-static g_lua_cmd_dspmain set on the AIDL path.
// Each setter reads the module's current config, changes exactly one
// parameter, and re-applies the full bundle — never stomps the others.

static void comp_config(float out[6]) {
    if (g_lua_cmd_dspmain)
        lsp_comp_get_config(g_lua_cmd_dspmain->jdsp.compressor_handle,
                            &out[0], &out[1], &out[2], &out[3], &out[4], &out[5]);
    else
        for (int i = 0; i < 6; i++) out[i] = 0.0f;
}
static void comp_apply(const float c[6]) {
    if (g_lua_cmd_dspmain) {
        lsp_comp_configure(g_lua_cmd_dspmain->jdsp.compressor_handle,
                           c[0], c[1], c[2], c[3], c[4], c[5]);
    }
}

static void gate_config(float out[6]) {
    if (g_lua_cmd_dspmain)
        lsp_gate_get_config(g_lua_cmd_dspmain->jdsp.gate_handle,
                            &out[0], &out[1], &out[2], &out[3], &out[4], &out[5]);
    else
        for (int i = 0; i < 6; i++) out[i] = 0.0f;
}
static void gate_apply(const float c[6]) {
    if (g_lua_cmd_dspmain) {
        lsp_gate_configure(g_lua_cmd_dspmain->jdsp.gate_handle,
                           c[0], c[1], c[2], c[3], c[4], c[5]);
    }
}

static void lim_config(float out[4]) {
    if (g_lua_cmd_dspmain)
        lsp_limiter_get_config(g_lua_cmd_dspmain->jdsp.limiter_handle,
                               &out[0], &out[1], &out[2], &out[3]);
    else
        for (int i = 0; i < 4; i++) out[i] = 0.0f;
}
static void lim_apply(const float c[4]) {
    if (g_lua_cmd_dspmain) {
        lsp_limiter_configure(g_lua_cmd_dspmain->jdsp.limiter_handle,
                              c[0], c[1], c[2], c[3]);
    }
}

#define JDSP_DYN_PROP(TAG, LOAD, APPLY, IDX, NC)                 \
    static double jdsp_prop_get_##TAG(void) {                    \
        float c[NC]; LOAD(c); return (double)c[IDX];             \
    }                                                            \
    static int jdsp_prop_set_##TAG(double v) {                   \
        float c[NC]; LOAD(c); c[IDX] = (float)v; APPLY(c);       \
        return 0;                                                \
    }

JDSP_DYN_PROP(comp_threshold, comp_config, comp_apply, 0, 6)
JDSP_DYN_PROP(comp_ratio,     comp_config, comp_apply, 1, 6)
JDSP_DYN_PROP(comp_knee,      comp_config, comp_apply, 2, 6)
JDSP_DYN_PROP(comp_attack,    comp_config, comp_apply, 3, 6)
JDSP_DYN_PROP(comp_release,   comp_config, comp_apply, 4, 6)
JDSP_DYN_PROP(comp_hold,      comp_config, comp_apply, 5, 6)

JDSP_DYN_PROP(gate_open,      gate_config, gate_apply, 0, 6)
JDSP_DYN_PROP(gate_close,     gate_config, gate_apply, 1, 6)
JDSP_DYN_PROP(gate_attack,    gate_config, gate_apply, 2, 6)
JDSP_DYN_PROP(gate_release,   gate_config, gate_apply, 3, 6)
JDSP_DYN_PROP(gate_hold,      gate_config, gate_apply, 4, 6)
JDSP_DYN_PROP(gate_reduction, gate_config, gate_apply, 5, 6)

JDSP_DYN_PROP(lim_threshold,  lim_config, lim_apply, 0, 4)
JDSP_DYN_PROP(lim_attack,     lim_config, lim_apply, 1, 4)
JDSP_DYN_PROP(lim_release,    lim_config, lim_apply, 2, 4)
JDSP_DYN_PROP(lim_knee,       lim_config, lim_apply, 3, 4)

static double jdsp_prop_get_tube_drive(void) {
    if (!g_lua_cmd_dspmain) return 0.0;
    return mag2dB(g_lua_cmd_dspmain->jdsp.tube.pregain);
}
static int jdsp_prop_set_tube_drive(double v) {
    if (!g_lua_cmd_dspmain) return -1;
    VacuumTubeSetGain(&g_lua_cmd_dspmain->jdsp, v);
    return 0;
}

static void register_lua_properties(void) {
    lua_bridge_register_property("compressor.threshold_db", jdsp_prop_get_comp_threshold, jdsp_prop_set_comp_threshold);
    lua_bridge_register_property("compressor.ratio",        jdsp_prop_get_comp_ratio,     jdsp_prop_set_comp_ratio);
    lua_bridge_register_property("compressor.knee_db",      jdsp_prop_get_comp_knee,      jdsp_prop_set_comp_knee);
    lua_bridge_register_property("compressor.attack_ms",    jdsp_prop_get_comp_attack,    jdsp_prop_set_comp_attack);
    lua_bridge_register_property("compressor.release_ms",   jdsp_prop_get_comp_release,   jdsp_prop_set_comp_release);
    lua_bridge_register_property("compressor.hold_ms",      jdsp_prop_get_comp_hold,      jdsp_prop_set_comp_hold);
    lua_bridge_register_property("gate.open_db",            jdsp_prop_get_gate_open,      jdsp_prop_set_gate_open);
    lua_bridge_register_property("gate.close_db",           jdsp_prop_get_gate_close,     jdsp_prop_set_gate_close);
    lua_bridge_register_property("gate.attack_ms",          jdsp_prop_get_gate_attack,    jdsp_prop_set_gate_attack);
    lua_bridge_register_property("gate.release_ms",         jdsp_prop_get_gate_release,   jdsp_prop_set_gate_release);
    lua_bridge_register_property("gate.hold_ms",            jdsp_prop_get_gate_hold,      jdsp_prop_set_gate_hold);
    lua_bridge_register_property("gate.reduction_db",       jdsp_prop_get_gate_reduction, jdsp_prop_set_gate_reduction);
    lua_bridge_register_property("limiter.threshold_db",    jdsp_prop_get_lim_threshold,  jdsp_prop_set_lim_threshold);
    lua_bridge_register_property("limiter.attack_ms",       jdsp_prop_get_lim_attack,     jdsp_prop_set_lim_attack);
    lua_bridge_register_property("limiter.release_ms",      jdsp_prop_get_lim_release,    jdsp_prop_set_lim_release);
    lua_bridge_register_property("limiter.knee_db",         jdsp_prop_get_lim_knee,       jdsp_prop_set_lim_knee);
    lua_bridge_register_property("tube.drive_db",           jdsp_prop_get_tube_drive,     jdsp_prop_set_tube_drive);
}

void EffectDSPMainConstructor(EffectDSPMain *dspmain)
{
	dspmain->initializeForFirst = 0;
	dspmain->tempImpulseIncoming = 0;
	dspmain->stringEq = 0;
	dspmain->boostingCon = 3;
	dspmain->bbMaxGain = 0;
	dspmain->numTime2Send = 0;
	dspmain->samplesInc = 0;
	dspmain->stringIndex = 0;
	JamesDSPInit(&dspmain->jdsp, 128, 48000.0f);
	// Wire the Lua command handler + property target HERE, on the real AIDL
	// path. These two lines previously lived only in EffectCreate() (the
	// legacy audio_effect_library_t entry point), which the shipping AIDL
	// wrapper never calls — so g_lua_cmd_dspmain stayed NULL on-device and
	// jdsp.toggle_module()/command() were unreachable. EffectCreate() still
	// sets them too (harmless); this is the call site that actually matters.
	g_lua_cmd_dspmain = dspmain;
	lua_bridge_set_command_handler(lua_cmd_dispatch);
	register_lua_properties();
}
void EffectDSPMainDestructor(EffectDSPMain *dspmain)
{
	JamesDSPFree(&dspmain->jdsp);
	if (dspmain->tempImpulseIncoming)
		free(dspmain->tempImpulseIncoming);
	if (dspmain->stringEq)
		free(dspmain->stringEq);
#ifdef DEBUG
	LOGI("Buffer freed");
#endif
}
int32_t configure(EffectDSPMain *dspmain, void* pCmdData, effect_buffer_access_e* mAccessMode)
{
    effect_config_t *cfg = (effect_config_t*)pCmdData;
    buffer_config_t in = cfg->inputCfg;
    buffer_config_t out = cfg->outputCfg;
    /* Check that we aren't asked to do resampling. Note that audioflinger
     * always provides full setup info at initial configure time. */
#ifdef DEBUG
    	LOGI("Sample rate of In: %u and out: %u", in.samplingRate, out.samplingRate);
#endif
    if ((in.mask & EFFECT_CONFIG_SMP_RATE) && (out.mask & EFFECT_CONFIG_SMP_RATE))
    {
        if (out.samplingRate != in.samplingRate)
        {
#ifdef DEBUG
	LOGW("In/out sample rate doesn't match");
#endif
    return -EINVAL;
        }
        dspmain->mSamplingRate = (float)in.samplingRate;
    }
    if (in.mask & EFFECT_CONFIG_CHANNELS && out.mask & EFFECT_CONFIG_CHANNELS)
    {
        if (in.channels != AUDIO_CHANNEL_OUT_STEREO)
        {
#ifdef DEBUG
	LOGE("Input is non stereo signal. It's channel count is %u", in.channels);
#endif
            return -EINVAL;
        }
        if (out.channels != AUDIO_CHANNEL_OUT_STEREO)
        {
#ifdef DEBUG
	LOGE("Output is non stereo signal. It's channel count is %u", out.channels);
#endif
            return -EINVAL;
        }
    }
    else
    {
#ifdef DEBUG
	LOGE("In/out channel mask doesn't match");
#endif
    }
    if (in.mask & EFFECT_CONFIG_FORMAT)
    {
        if (in.format == AUDIO_FORMAT_PCM_FLOAT)
        	dspmain->formatFloatModeInt32Mode = 1;
        else if (in.format == AUDIO_FORMAT_PCM_32_BIT)
        	dspmain->formatFloatModeInt32Mode = 2;
        else if (in.format == AUDIO_FORMAT_PCM_24_BIT_PACKED)
        	dspmain->formatFloatModeInt32Mode = 3;
        else if (in.format == AUDIO_FORMAT_PCM_8_24_BIT)
        	dspmain->formatFloatModeInt32Mode = 4;
        else if (in.format == AUDIO_FORMAT_PCM_16_BIT)
        	dspmain->formatFloatModeInt32Mode = 0;
    }
    if (out.mask & EFFECT_CONFIG_FORMAT)
    {
        if (out.format == AUDIO_FORMAT_PCM_FLOAT)
        	dspmain->formatFloatModeInt32Mode = 1;
        else if (out.format == AUDIO_FORMAT_PCM_32_BIT)
        	dspmain->formatFloatModeInt32Mode = 2;
        else if (out.format == AUDIO_FORMAT_PCM_24_BIT_PACKED)
        	dspmain->formatFloatModeInt32Mode = 3;
        else if (out.format == AUDIO_FORMAT_PCM_8_24_BIT)
        	dspmain->formatFloatModeInt32Mode = 4;
        else if (out.format == AUDIO_FORMAT_PCM_16_BIT)
        	dspmain->formatFloatModeInt32Mode = 0;
    }
#ifdef DEBUG
	LOGW("I/O FMT = { %u, %u }", in.format, out.format);
#endif
    if (out.mask & EFFECT_CONFIG_ACC_MODE)
        *mAccessMode = (effect_buffer_access_e) out.accessMode;
    return 0;
}
int32_t EffectDSPMainCommand(EffectDSPMain *dspmain, uint32_t cmdCode, uint32_t cmdSize, void* pCmdData, uint32_t* replySize, void* pReplyData)
{
#ifdef DEBUG
//		LOGI("Memory used: %f Mb", (float)getCurrentRSS() / 1024.0 / 1024.0);
#endif
	if (cmdCode == EFFECT_CMD_SET_CONFIG)
	{
		effect_buffer_access_e mAccessMode;
		int32_t *replyData = (int32_t *)pReplyData;
		int32_t ret = configure(dspmain, pCmdData, &mAccessMode);
		if (ret != 0)
		{
			*replyData = ret;
			return 0;
		}
		// Set sample rate
		JamesDSPSetSampleRate(&dspmain->jdsp, dspmain->mSamplingRate, 0);
		*replyData = 0;
		return 0;
	}
	// Hardening (2026-10-21): a vendor parameter blob is untrusted client
	// input (any IEffect client can send one). An empty/short blob
	// null-derefs below (observed: open() with an empty DefaultExtension ->
	// SIGSEGV in EffectDSPMainCommand+320, i.e. the audio HAL process dies);
	// a blob whose declared psize/vsize exceed cmdSize over-reads the Binder
	// buffer. Reject both before any field is read.
	if (cmdCode == EFFECT_CMD_GET_PARAM || cmdCode == EFFECT_CMD_SET_PARAM)
	{
		if (!pCmdData || cmdSize < sizeof(effect_param_t))
			return -1;
		effect_param_t *hdr = (effect_param_t *)pCmdData;
		uint32_t need = (uint32_t)sizeof(effect_param_t) + ((hdr->psize + 3u) & ~3u) + hdr->vsize;
		if (need < hdr->vsize || cmdSize < need)
			return -1;
	}
	if (cmdCode == EFFECT_CMD_GET_PARAM)
	{
		effect_param_t *cep = (effect_param_t *)pCmdData;
		if (cep->psize == 4 && cep->vsize == 4)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 19998)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 19998;
				replyData->data = (int32_t)dspmain->initializeForFirst++;
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
			if (cmd == 19999)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 19999;
				replyData->data = (int32_t)dspmain->jdsp.blockSize;
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
			else if (cmd == 20000)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 20000;
				replyData->data = (int32_t)dspmain->jdsp.blockSizeMax;
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
			else if (cmd == 20001)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 20001;
				replyData->data = (int32_t)dspmain->jdsp.fs;
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
			else if (cmd == 20002)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 20002;
				replyData->data = (int32_t)getpid();
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
			else if (cmd == 30000)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 30000;
				replyData->data = dspmain->hashSlot[0];
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
			else if (cmd == 30001)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 30001;
				replyData->data = dspmain->hashSlot[1];
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
			else if (cmd == 30002)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 30002;
				replyData->data = dspmain->hashSlot[2];
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
			else if (cmd == 30003)
			{
				reply1x4_1x4_t *replyData = (reply1x4_1x4_t *)pReplyData;
				replyData->status = 0;
				replyData->psize = 4;
				replyData->vsize = 4;
				replyData->cmd = 30003;
				replyData->data = dspmain->hashSlot[3];
				*replySize = sizeof(reply1x4_1x4_t);
				return 0;
			}
		}
	}
	if (cmdCode == EFFECT_CMD_SET_PARAM)
	{
		effect_param_t *cep = (effect_param_t *)pCmdData;
		int32_t *replyData = (int32_t *)pReplyData;
		if (cep->psize == 4 && cep->vsize == 2)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 128)
			{
				int reverbMode = (int)((int16_t *)cep)[8];
				Reverb_SetParam(&dspmain->jdsp, reverbMode);
#ifdef DEBUG
				LOGI("Reverb mode: %d", reverbMode);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 137)
			{
				float agress = ((int16_t *)cep)[8] / 100.0f;
				StereoEnhancementSetParam(&dspmain->jdsp, agress);
#ifdef DEBUG
				LOGE("Stereo widen ag: %1.7f", agress);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 188)
			{
				int16_t nMode = ((int16_t *)cep)[8];
				if (nMode < 0)
					nMode = 0;
				if (nMode > 5)
					nMode = 5;
				CrossfeedChangeMode(&dspmain->jdsp, nMode);
#ifdef DEBUG
				LOGI("Crossfeed mode: %d", nMode);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 150)
			{
				float tubedrive = ((int16_t *)cep)[8] / 1000.0f;
#ifdef DEBUG
				LOGI("Tube drive: %1.7f", tubedrive);
#endif
				VacuumTubeSetGain(&dspmain->jdsp, tubedrive);
				*replyData = 0;
				return 0;
			}
			else if (cmd == 112)
			{
				float maxGain = (float)((int16_t *)cep)[8];
				BassBoostSetParam(&dspmain->jdsp, maxGain);
#ifdef DEBUG
				LOGI("Bass boost max gain: %1.7f", maxGain);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1201)
			{
				int16_t bassBoostEnabled = ((int16_t *)cep)[8];
				if (!bassBoostEnabled)
					BassBoostDisable(&dspmain->jdsp);
				else
					BassBoostEnable(&dspmain->jdsp);
#ifdef DEBUG
				LOGE("Bass boost enabled: %d", bassBoostEnabled);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1202)
			{
				int16_t equalizerEnabled = ((int16_t *)cep)[8];
				if (!equalizerEnabled)
					MultimodalEqualizerDisable(&dspmain->jdsp);
				else
					MultimodalEqualizerEnable(&dspmain->jdsp, 1);
#ifdef DEBUG
				LOGE("FIR equalizer enabled: %d", equalizerEnabled);
#endif
				// Enable EQ
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1203)
			{
				int16_t reverbEnabled = ((int16_t *)cep)[8];
				if (!reverbEnabled)
					ReverbDisable(&dspmain->jdsp);
				else
					ReverbEnable(&dspmain->jdsp);
#ifdef DEBUG
				LOGE("Reverb enabled: %d", reverbEnabled);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1204)
			{
				int16_t stereoWidenEnabled = ((int16_t *)cep)[8];
				if (!stereoWidenEnabled)
					StereoEnhancementDisable(&dspmain->jdsp);
				else
					StereoEnhancementEnable(&dspmain->jdsp);
#ifdef DEBUG
				LOGE("Stereo widen enabled: %d", stereoWidenEnabled);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1205)
			{
				int16_t convolverEnabled = ((int16_t *)cep)[8];
#ifdef DEBUG
				LOGE("Convolver enabled: %d", convolverEnabled);
#endif
				if (!convolverEnabled)
					Convolver1DDisable(&dspmain->jdsp);
				else
					Convolver1DEnable(&dspmain->jdsp);
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1206)
			{
				int16_t analogueModelEnable = ((int16_t *)cep)[8];
				if (!analogueModelEnable)
					VacuumTubeDisable(&dspmain->jdsp);
				else
					VacuumTubeEnable(&dspmain->jdsp);
#ifdef DEBUG
				LOGE("Analogue modelling enabled: %d", analogueModelEnable);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1207)
			{
				int16_t autogainEnabled = ((int16_t *)cep)[8];
				dspmain->jdsp.autogainEnabled = autogainEnabled ? 1 : 0;
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1208)
			{
				int16_t bs2bEnabled = ((int16_t *)cep)[8];
				if (!bs2bEnabled)
					CrossfeedDisable(&dspmain->jdsp);
				else
					CrossfeedEnable(&dspmain->jdsp, 1);
#ifdef DEBUG
				LOGE("Crossfeed enabled: %d", bs2bEnabled);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1210)
			{
				int16_t arbitraryResponseEnabled = ((int16_t *)cep)[8];
				if (!arbitraryResponseEnabled)
					ArbitraryResponseEqualizerDisable(&dspmain->jdsp);
				else
					ArbitraryResponseEqualizerEnable(&dspmain->jdsp, 1);
#ifdef DEBUG
				LOGE("Arbitrary response eq enabled: %d", arbitraryResponseEnabled);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1212)
			{
				int16_t viperddcEnabled = ((int16_t *)cep)[8];
				int errCode = 0;
				if (!viperddcEnabled)
					DDCDisable(&dspmain->jdsp);
				else
					errCode = DDCEnable(&dspmain->jdsp, 1);
#ifdef DEBUG
				LOGE("viperddcEnabled: %d, success?: %d", viperddcEnabled, errCode);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 1213)
			{
				int16_t EELEnabled = ((int16_t *)cep)[8];
				if (!EELEnabled)
					LiveProgDisable(&dspmain->jdsp);
				else
					LiveProgEnable(&dspmain->jdsp);
#ifdef DEBUG
				LOGE("Liveprog enabled: %d", EELEnabled);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 10004)
			{
				// Load impulse
#ifdef DEBUG
				if (dspmain->samplesInc == dspmain->numTime2Send)
					LOGI("Buffer slices complete, chs: %d, impLen: %d", dspmain->impChannels, dspmain->impulseLengthActual);
				else
					LOGI("Ops! Buffer slices got %d missing holes", dspmain->numTime2Send - dspmain->samplesInc);
#endif
				dspmain->samplesInc = 0;
				int success = Convolver1DLoadImpulseResponse(&dspmain->jdsp, dspmain->tempImpulseIncoming, dspmain->impChannels, dspmain->impulseLengthActual, 1);
				union
				{
					int32_t raw;
					float f;
				} fltInt;
				int32_t crc = 0xFFFFFFFF;
				for (unsigned int i = 0; i < dspmain->impulseLengthActual * dspmain->impChannels; i++)
				{
					fltInt.f = dspmain->tempImpulseIncoming[i];
					crc = crc ^ fltInt.raw;
					for (int j = 7; j >= 0; j--)
					{
						int32_t mask = -(crc & 1);
						crc = (crc >> 1) ^ (0xEDB88320 & mask);
					}
				}
				dspmain->engineImpulseResponseHash = ~crc;
				free(dspmain->tempImpulseIncoming);
				dspmain->tempImpulseIncoming = 0;
#ifdef DEBUG
				LOGI("FFT convolver errCode: %d", success);
#endif
				*replyData = 0;
				return 0;
			}
			else if (cmd == 10006)
			{
				dspmain->stringIndex = 0;
				if (dspmain->stringEq)
				{
#ifdef DEBUG
					LOGI("%s", dspmain->stringEq);
#endif
					ArbitraryResponseEqualizerStringParser(&dspmain->jdsp, dspmain->stringEq);
					free(dspmain->stringEq);
					dspmain->stringEq = 0;
#ifdef DEBUG
					LOGI("Arbitrary response initialized");
#endif
				}
				*replyData = 0;
				return 0;
			}
			else if (cmd == 10009)
			{
				dspmain->stringIndex = 0;
				if (dspmain->stringEq)
				{
#ifdef DEBUG
					LOGI("%s", dspmain->stringEq);
#endif
					// Initialize DDC
					DDCStringParser(&dspmain->jdsp, dspmain->stringEq);
					free(dspmain->stringEq);
					dspmain->stringEq = 0;
#ifdef DEBUG
					LOGI("DDC initialized");
#endif
				}
				*replyData = 0;
				return 0;
			}
			else if (cmd == 10010)
			{
				dspmain->stringIndex = 0;
				if (dspmain->stringEq)
				{
#ifdef DEBUG
					LOGI("%s", dspmain->stringEq);
#endif
					// Initialize EEL
					int errorCode = LiveProgStringParser(&dspmain->jdsp, dspmain->stringEq);
					free(dspmain->stringEq);
					dspmain->stringEq = 0;
#ifdef DEBUG
					LOGI("Live prog error message: %s", checkErrorCode(errorCode));
#endif
				}
				*replyData = 0;
				return 0;
			}
			else if (cmd == 10020) // Load Lua script from accumulated string
			{
				dspmain->stringIndex = 0;
				if (dspmain->stringEq)
				{
					lua_bridge_load_script(dspmain->stringEq);
					free(dspmain->stringEq);
					dspmain->stringEq = 0;
				}
				*replyData = 0;
				return 0;
			}
			else if (cmd == 10021) // Unload Lua script
			{
				lua_bridge_destroy();
				lua_bridge_init();
				*replyData = 0;
				return 0;
			}
		}
		// Gate/Compressor/MB-EQ (10100/10101/10102) send an 8-float payload
		// (32 bytes) from setParameterFloatArray — these used to be
		// (wrongly) nested inside the "vsize == 2" block above, which is
		// sized for old 2-byte int16 commands (Reverb mode, Crossfeed
		// mode, etc). Real cep->vsize for these is 32, so that guard was
		// always false and the whole handler was unreachable: the enable
		// flag never got set, so Compressor's Bypass stayed permanently
		// engaged no matter what the UI sent. Confirmed via logcat
		// ("Failed to apply dsp_compander" on every toggle) before fixing.
		if (cep->psize == 4 && cep->vsize == 32)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 10100) // Gate: enabled, open_db, close_db, attack, release, hold, reduct
			{
				float *p = (float *)cep;
				int en = (int)roundf(p[5]);
				dspmain->jdsp.gateEnabled = en;
				if (en) {
					lsp_gate_configure(dspmain->jdsp.gate_handle, p[6], p[7], p[8], p[9], p[10], p[11]);
				}
				*replyData = 0;
				return 0;
			}
			else if (cmd == 10101) // Compressor: enabled, threshold, ratio, knee, attack, release, hold
			{
				float *p = (float *)cep;
				int en = (int)roundf(p[5]);
				dspmain->jdsp.mbDynEnabled = en;
				if (en) {
					lsp_comp_configure(dspmain->jdsp.compressor_handle, p[6], p[7], p[8], p[9], p[10], p[11]);
				}
				*replyData = 0;
				return 0;
			}
			else if (cmd == 10102) // MB EQ: (no-op, single-band)
			{
				*replyData = 0;
				return 0;
			}
		}
		// Expander (10103): 9-float payload (36 bytes) at int32 index 4..12 =
		// [band(unused)=0, enabled, mode(0=down,1=up), threshold_db, ratio,
		//  knee(linear), attack_ms, release_ms, hold_ms]. Same shape as
		// 10100/10101 above; no jdsp_lock (lsp_exp_configure publishes,
		// the audio thread applies -- rt_config.h).
		if (cep->psize == 4 && cep->vsize == 36)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 10103)
			{
				float *p = (float *)cep;
				int en = (int)roundf(p[5]);
				dspmain->jdsp.expanderEnabled = en;
				if (en) {
					lsp_exp_configure(dspmain->jdsp.expander_handle, (int)roundf(p[6]), p[7], p[8], p[9], p[10], p[11], p[12]);
				}
				*replyData = 0;
				return 0;
			}
			// Autogain (10104): same 9-float shape, int32 index 4..12 =
			// [band(unused)=0, enabled, level_lufs, long_period_ms, short_period_ms,
			//  drift_db, silence_lufs, max_gain_db, max_gain_on]. Sets the same
			// autogainEnabled flag as legacy cmd 1207.
			else if (cmd == 10104)
			{
				float *p = (float *)cep;
				int en = (int)roundf(p[5]);
				dspmain->jdsp.autogainEnabled = en;
				if (en) {
					lsp_autogain_configure(dspmain->jdsp.autogain_handle, p[6], p[7], p[8], p[9], p[10], p[11], (int)roundf(p[12]));
				}
				*replyData = 0;
				return 0;
			}
		}
		if (cep->psize == 4 && cep->vsize == 4)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 25000)
			{
				dspmain->hashSlot[0] = ((int32_t*)cep)[4];
#ifdef DEBUG
				LOGI("ArbEq hash: %d", dspmain->hashSlot[0]);
#endif
				*replyData = 0;
				return 0;
			}
			if (cmd == 25001)
			{
				dspmain->hashSlot[0 + 1] = ((int32_t*)cep)[4];
#ifdef DEBUG
				LOGI("DDC hash: %d", dspmain->hashSlot[0 + 1]);
#endif
				*replyData = 0;
				return 0;
			}
			if (cmd == 25002)
			{
				dspmain->hashSlot[0 + 2] = ((int32_t*)cep)[4];
#ifdef DEBUG
				LOGI("LiveProg hash: %d", dspmain->hashSlot[0 + 2]);
#endif
				*replyData = 0;
				return 0;
			}
			if (cmd == 25003)
			{
				dspmain->hashSlot[0 + 3] = ((int32_t*)cep)[4];
#ifdef DEBUG
				LOGI("Convolver app hash: %d, engine hash: %d", dspmain->hashSlot[0 + 3], dspmain->engineImpulseResponseHash);
#endif
				*replyData = 0;
				return 0;
			}
		}
		if (cep->psize == 4 && cep->vsize == 12)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 1500)
			{
				// Legacy 3-float call shape, kept for compatibility with any
				// caller still sending threshold/release/postgain only —
				// attack/knee default to the same values JLimiterSetCoefficients
				// always hardcoded (5ms/no-knee). The real UI now sends the
				// 5-float shape below (vsize == 20) instead.
				double limThreshold = (double)((float*)cep)[4];
				double limRelease = (double)((float*)cep)[5];
				double postgain = (double)((float*)cep)[6];
				if (limThreshold > -0.09)
					limThreshold = -0.09;
				if (limRelease < 0.15)
					limRelease = 0.15;
				if (postgain > 15.0)
					postgain = 15.0;
				if (postgain < -15.0)
					postgain = -15.0;
				JLimiterSetCoefficients(&dspmain->jdsp, limThreshold, limRelease);
				JamesDSPSetPostGain(&dspmain->jdsp, postgain);
#ifdef DEBUG
				LOGE("limThreshold: %f, limRelease: %f, postgain: %f", limThreshold, limRelease, postgain);
#endif
				*replyData = 0;
				return 0;
			}
		}
		if (cep->psize == 4 && (cep->vsize == 20 || cep->vsize == 24))
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 1500) // Limiter: threshold, attack, release, knee, postgain[, enabled]
			{
				float *p = (float *)cep;
				double limThreshold = (double)p[4];
				double limAttack = (double)p[5];
				double limRelease = (double)p[6];
				double limKnee = (double)p[7];
				double postgain = (double)p[8];
				if (limThreshold > -0.09)
					limThreshold = -0.09;
				if (limAttack < 0.25)
					limAttack = 0.25;
				if (limRelease < 0.25)
					limRelease = 0.25;
				if (limKnee < 0.0)
					limKnee = 0.0;
				if (limKnee > 1.0)
					limKnee = 1.0;
				if (postgain > 15.0)
					postgain = 15.0;
				if (postgain < -15.0)
					postgain = -15.0;
				// Calls the real LSP limiter directly, bypassing the
				// JLimiterSetCoefficients shim (which hardcodes attack=5ms,
				// knee=1.0/no-knee) — this is the path that actually exposes
				// attack/knee as real tunables from the UI.
				// No lock: configure() only publishes; the audio thread applies
				// it at its next block (rt_config.h).
				lsp_limiter_configure(dspmain->jdsp.limiter_handle, (float)limThreshold, (float)limAttack, (float)limRelease, (float)limKnee);
				JamesDSPSetPostGain(&dspmain->jdsp, postgain);
				// Optional 6th float (vsize 24): output limiter on/off. Older
				// app builds send 5 floats and leave it on (stock behaviour).
				if (cep->vsize == 24)
					dspmain->jdsp.limiterEnabled = (p[9] >= 0.5f) ? 1 : 0;
				*replyData = 0;
				return 0;
			}
		}
		if (cep->psize == 4 && cep->vsize == 128)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 116)
			{
				int filtertype = roundf(((float*)cep)[4 + 0]);
				int interpolationMode = (((float*)cep)[4 + 1]) < 0.0f ? 0 : 1;
				float *ptrFreqAxis = &((float*)cep)[4 + 2];
				float *ptrGainAxis = ptrFreqAxis + 15;
				double param[30];
				for (int i = 0; i < 15; i++)
				{
					param[i] = (double)ptrFreqAxis[i];
					param[i + 15] = (double)ptrGainAxis[i];
				}
#ifdef DEBUG
				if (filtertype == 0)
					LOGI("filtertype: FIR Minimum phase");
				else if (filtertype == 1)
					LOGI("filtertype: IIR 4 order");
				else if (filtertype == 2)
					LOGI("filtertype: IIR 6 order");
				else if (filtertype == 3)
					LOGI("filtertype: IIR 8 order");
				else if (filtertype == 4)
					LOGI("filtertype: IIR 10 order");
				else
					LOGI("filtertype: IIR 12 order");
				LOGI("Eq axis: ");
				for (int i = 0; i < 15; i++)
					LOGI("%1.7lf %1.7lf; ", param[i], param[i + 15]);
#endif
				MultimodalEqualizerAxisInterpolation(&dspmain->jdsp, interpolationMode, filtertype, param, param + 15);
				*replyData = 0;
				return 0;
			}
		}
		if (cep->psize == 4 && cep->vsize == 8)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 8888) // Implemented
			{
				int32_t times = ((int32_t *)cep)[4];
				int32_t sizePerBuffer = ((int32_t *)cep)[5];
				int stringLength = times * sizePerBuffer;
#ifdef DEBUG
				LOGI("Allocate %d string length", stringLength);
#endif
				dspmain->stringEq = (char*)calloc(stringLength, sizeof(char));
				*replyData = 0;
				return 0;
			}
		}
		if (cep->psize == 4 && cep->vsize == 16)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 9999) // Implemented
			{
				dspmain->impChannels = ((int32_t *)cep)[5];
				dspmain->impulseLengthActual = ((int32_t *)cep)[4] / dspmain->impChannels;
				float convGaindB = (float)((int32_t *)cep)[6] / 262144.0f;
				if (convGaindB > 50.0f)
					convGaindB = 50.0f;
				dspmain->numTime2Send = ((int32_t *)cep)[7];
				dspmain->tempImpulseIncoming = (float*)calloc(4096 * dspmain->impChannels * dspmain->numTime2Send, sizeof(float));
				*replyData = 0;
				return 0;
			}
		}
		if (cep->psize == 4 && cep->vsize == 16384)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 12000) // Implemented
			{
				memcpy(dspmain->tempImpulseIncoming + (dspmain->samplesInc++ * 4096), ((float*)cep) + 4, 4096 * sizeof(float));
				*replyData = 0;
				return 0;
			}
		}
		if (cep->psize == 4 && cep->vsize == 256)
		{
			int32_t cmd = ((int32_t *)cep)[3];
			if (cmd == 12001) // Implemented
			{
				memcpy(dspmain->stringEq + (dspmain->stringIndex++ * 256), ((char*)cep) + 16, 256 * sizeof(char));
				*replyData = 0;
				return 0;
			}
		}
		return -1;
	}
    switch (cmdCode)
    {
    case EFFECT_CMD_ENABLE:
    case EFFECT_CMD_DISABLE:
    {
        dspmain->mEnable = cmdCode == EFFECT_CMD_ENABLE;
        int32_t *replyData = (int32_t *) pReplyData;
        *replyData = 0;
        break;
    }
    case EFFECT_CMD_INIT:
    case EFFECT_CMD_SET_CONFIG:
    case EFFECT_CMD_SET_PARAM:
    case EFFECT_CMD_SET_PARAM_COMMIT:
    {
        int32_t *replyData = (int32_t *) pReplyData;
        *replyData = 0;
        break;
    }
    case EFFECT_CMD_RESET:
    case EFFECT_CMD_SET_PARAM_DEFERRED:
    case EFFECT_CMD_SET_DEVICE:
    case EFFECT_CMD_SET_AUDIO_MODE:
        break;
    case EFFECT_CMD_GET_PARAM:
    {
        effect_param_t *rep = (effect_param_t *) pReplyData;
        rep->status = -EINVAL;
        rep->psize = 0;
        rep->vsize = 0;
        *replySize = 12;
        break;
    }
    }
    return 0;
}

int32_t EffectDSPMainProcess(EffectDSPMain *dspmain, audio_buffer_t *in, audio_buffer_t *out)
{
	size_t actualFrameCount = in->frameCount;
	switch (dspmain->formatFloatModeInt32Mode)
	{
	case 0:
		dspmain->jdsp.processInt16Multiplexd(&dspmain->jdsp, in->s16, out->s16, actualFrameCount);
		break;
	case 1:
		dspmain->jdsp.processFloatMultiplexd(&dspmain->jdsp, in->f32, out->f32, actualFrameCount);
		break;
	case 2:
		dspmain->jdsp.processInt32Multiplexd(&dspmain->jdsp, in->s32, out->s32, actualFrameCount);
		break;
	case 3:
		dspmain->jdsp.processInt24PackedMultiplexd(&dspmain->jdsp, (uint8_t*)in->raw, (uint8_t*)out->raw, actualFrameCount);
		break;
	case 4:
		dspmain->jdsp.processInt8_24Multiplexd(&dspmain->jdsp, in->s32, out->s32, actualFrameCount);
		break;
	}
	return dspmain->mEnable ? 0 : -ENODATA;
}
// Effect section end
static effect_descriptor_t jamesdsp_descriptor =
{
	{ 0xf98765f4, 0xc321, 0x5de6, 0x9a45, { 0x12, 0x34, 0x59, 0x49, 0x5a, 0xb2 } },
	{ 0xf27317f4, 0xc984, 0x4de6, 0x9a90, { 0x54, 0x57, 0x59, 0x49, 0x5b, 0xf2 } }, // own UUID
	EFFECT_CONTROL_API_VERSION,
	EFFECT_FLAG_TYPE_INSERT | EFFECT_FLAG_INSERT_FIRST,
	10,
	1,
	"JamesDSP v4.01",
	"James Fung"
};
__attribute__((constructor)) static void initialize(void)
{
	JamesDSPGlobalMemoryAllocation();
#ifdef DEBUG
	LOGI("Initialization: DLL loaded");
#endif
}
__attribute__((destructor)) static void destruction(void)
{
	JamesDSPGlobalMemoryDeallocation();
#ifdef DEBUG
	LOGI("Initialization: DLL unloaded");
#endif
}
// Library mandatory methods
struct effect_module_s
{
	const struct effect_interface_s *itfe;
	EffectDSPMain effect;
	effect_descriptor_t *descriptor;
};
static int32_t generic_process(effect_handle_t self, audio_buffer_t *in, audio_buffer_t *out)
{
	struct effect_module_s *e = (struct effect_module_s *) self;
	return EffectDSPMainProcess(&e->effect, in, out);
}
static int32_t generic_command(effect_handle_t self, uint32_t cmdCode, uint32_t cmdSize, void *pCmdData, uint32_t *replySize, void *pReplyData)
{
	struct effect_module_s *e = (struct effect_module_s *) self;
	return EffectDSPMainCommand(&e->effect, cmdCode, cmdSize, pCmdData, replySize, pReplyData);
}
static int32_t generic_getDescriptor(effect_handle_t self, effect_descriptor_t *pDescriptor)
{
	struct effect_module_s *e = (struct effect_module_s *) self;
	memcpy(pDescriptor, e->descriptor, sizeof(effect_descriptor_t));
	return 0;
}
static const struct effect_interface_s generic_interface =
{
	generic_process,
	generic_command,
	generic_getDescriptor,
	NULL
};
int32_t EffectCreate(const effect_uuid_t *uuid, int32_t sessionId, int32_t ioId, effect_handle_t *pEffect)
{
	(void)uuid;
	(void)ioId;
	struct effect_module_s *e = (struct effect_module_s *) calloc(1, sizeof(struct effect_module_s));
	e->itfe = &generic_interface;
#ifdef DEBUG
	LOGI("JamesDSP: Creating effect with %d sessionId", sessionId);
#endif
	EffectDSPMainConstructor(&e->effect);
	g_lua_cmd_dspmain = &e->effect;
	lua_bridge_set_command_handler(lua_cmd_dispatch);
	e->descriptor = &jamesdsp_descriptor;
	*pEffect = (effect_handle_t)e;
#ifdef DEBUG
	LOGI("JamesDSP: Effect created");
#endif
	return 0;
}
int32_t EffectRelease(effect_handle_t ei)
{
	struct effect_module_s *e = (struct effect_module_s *) ei;
	EffectDSPMainDestructor(&e->effect);
	free(e);
	return 0;
}
int32_t EffectGetDescriptor(const effect_uuid_t *uuid, effect_descriptor_t *pDescriptor)
{
#ifdef DEBUG
	LOGI("JamesDSP: Copying descriptor info");
#endif
	if (pDescriptor == NULL || uuid == NULL)
	{
#ifdef DEBUG
		LOGI("JamesDSP: EffectGetDescriptor() called with NULL pointer");
#endif
		return -EINVAL;
	}
	*pDescriptor = jamesdsp_descriptor;
	return 0;
}
__attribute__((visibility("default"))) audio_effect_library_t AUDIO_EFFECT_LIBRARY_INFO_SYM =
{
	.tag = AUDIO_EFFECT_LIBRARY_TAG,
	.version = EFFECT_LIBRARY_API_VERSION,
#if __aarch64__ == 1
		.name = "James Audio DSP arm64",
#elif __ARM_ARCH_7A__ == 1
		.name = "James Audio DSP arm32",
#elif __i386__ == 1
		.name = "James Audio DSP x86",
#elif __x86_64__ == 1
		.name = "James Audio DSP x64",
#endif
		.implementor = "James Fung",
		.create_effect = EffectCreate,
		.release_effect = EffectRelease,
		.get_descriptor = EffectGetDescriptor,
};
