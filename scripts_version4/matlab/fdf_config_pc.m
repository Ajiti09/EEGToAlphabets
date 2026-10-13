function cfg = paperstyle_fdf_config_pc()


cfg.DATA_ROOT  = '/home/naveed/Desktop/zips';
cfg.OUTPUT_DIR = '/home/naveed/Desktop/c3_picard_50HzNotch_FDF_epochs';


cfg.EEGLAB_PATH = '/home/naveed/Desktop/Pipelines/eeglab_current/eeglab2026.0.0';

cfg.CLASSES = {'A','K','N','P','Z'};
cfg.DEFAULT_FS = 125.0;


cfg.CHANNEL_NAMES = {'Fp1','Fp2','C3','C4','P7','P8','O1','O2', ...
                     'F7','F8','F3','F4','T7','T8','P3','P4'};


cfg.GLOBAL_DROP_CHANNELS = {'T8'};


cfg.APPLY_50HZ_NOTCH = true;
cfg.NOTCH_FREQ = 50.0;
cfg.NOTCH_Q = 30.0;


cfg.FDF_REJECT_LOW = 4.0;
cfg.FDF_REJECT_HIGH = 15.0;
cfg.FDF_TRANSITION_HZ = 1.0; 


cfg.PREFER_PICARD = true;
cfg.REQUIRE_PICARD = true;
cfg.ICA_MAX_COMPONENTS_REMOVED = 2;
cfg.ICA_CORR_THRESHOLD = 0.35; 


cfg.TASK_TRIM_START_SEC = 0.0;
cfg.TASK_TRIM_END_SEC   = 0.0;

cfg.RAIL_UV = 180000.0;
cfg.MAX_RAIL_RATIO = 0.05;
cfg.MIN_STD_UV = 0.01;
cfg.ROBUST_Z_STD_THRESHOLD = 8.0;
cfg.IMPUTE_STRATEGY = 'mean_good';
cfg.REFERENCE = 'none'; 


cfg.APPLY_BAD_EPOCH_REJECTION = false;
cfg.BAD_EPOCH_ROBUST_Z = 6.0;
cfg.BAD_EPOCH_MAX_ABS_UV = 1000.0;
cfg.MIN_KEEP_PER_CLASS_SESSION = 18;


cfg.NEIGHBORS = containers.Map();
cfg.NEIGHBORS('Fp1') = {'Fp2','F7','F3'};
cfg.NEIGHBORS('Fp2') = {'Fp1','F8','F4'};
cfg.NEIGHBORS('F7')  = {'Fp1','F3','T7'};
cfg.NEIGHBORS('F8')  = {'Fp2','F4','P8'};
cfg.NEIGHBORS('F3')  = {'Fp1','F7','C3','F4'};
cfg.NEIGHBORS('F4')  = {'Fp2','F8','C4','F3'};
cfg.NEIGHBORS('C3')  = {'F3','C4','P3','T7'};
cfg.NEIGHBORS('C4')  = {'F4','C3','P4','P8'};
cfg.NEIGHBORS('T7')  = {'F7','C3','P7'};
cfg.NEIGHBORS('P7')  = {'T7','P3','O1'};
cfg.NEIGHBORS('P8')  = {'F8','C4','P4','O2'};
cfg.NEIGHBORS('P3')  = {'C3','P7','P4','O1'};
cfg.NEIGHBORS('P4')  = {'C4','P8','P3','O2'};
cfg.NEIGHBORS('O1')  = {'P7','P3','O2'};
cfg.NEIGHBORS('O2')  = {'P8','P4','O1'};
cfg.NEIGHBORS('T8')  = {'F8','P8','C4'};

W = struct([]);
W(1).name = 'win8_0';
W(1).description = 'Paper Scheme-B: 8-sample non-overlapping windows for LOSO/GroupKFold-style evaluation';
W(1).is_windowed = true;
W(1).window_samples = 8;
W(1).step_samples = 8;
W(1).window_sec = W(1).window_samples / cfg.DEFAULT_FS;
W(1).step_sec = W(1).step_samples / cfg.DEFAULT_FS;

cfg.FEATURE_MODES = W;

P = struct([]);
P(1).name = 'C3_picard_50HzNotch_FDF';
P(1).description = 'Bangladesh-safe paper-adapted branch: narrow zero-phase 50 Hz notch + Picard ICA + FFT 4-15 Hz FD-F suppression + IFFT + 8-sample non-overlap';
P(1).extra_drop_channels = {};
P(1).active_channels = {};
P(1).apply_ica = true;
P(1).prefer_picard = cfg.PREFER_PICARD;
P(1).require_picard = cfg.REQUIRE_PICARD;
P(1).ica_corr_threshold = cfg.ICA_CORR_THRESHOLD;
P(1).ica_max_components_removed = cfg.ICA_MAX_COMPONENTS_REMOVED;
P(1).impute_strategy = cfg.IMPUTE_STRATEGY;
P(1).reference = cfg.REFERENCE;
P(1).task_trim_start = cfg.TASK_TRIM_START_SEC;
P(1).task_trim_end = cfg.TASK_TRIM_END_SEC;

cfg.PREPROCESSINGS = P;


cfg.SKIP_IF_EXISTS = false;

end
