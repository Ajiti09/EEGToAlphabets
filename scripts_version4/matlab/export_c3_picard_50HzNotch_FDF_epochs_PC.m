clear; clc;
cfg = paperstyle_fdf_config_pc();
setupEEGLABIfAvailable(cfg);

if ~exist(cfg.OUTPUT_DIR, 'dir'); mkdir(cfg.OUTPUT_DIR); end

fprintf('\n=== C3 Picard + 50 Hz Notch + FD-F MATLAB Stage: Tensor Export ===\n');
fprintf('DATA_ROOT : %s\n', cfg.DATA_ROOT);
fprintf('OUTPUT_DIR: %s\n', cfg.OUTPUT_DIR);

searchRoot = unzipArchives(cfg);
sessions = findSessions(searchRoot, cfg);
fprintf('\nFound %d recording sessions.\n', height(sessions));
writetable(sessions, fullfile(cfg.OUTPUT_DIR, 'paperstyle_fdf_session_index.csv'));

manifest = table();

for pi = 1:numel(cfg.PREPROCESSINGS)
    variant = cfg.PREPROCESSINGS(pi);
    for wi = 1:numel(cfg.FEATURE_MODES)
        wmode = cfg.FEATURE_MODES(wi);
        outDir = fullfile(cfg.OUTPUT_DIR, variant.name, wmode.name);
        if ~exist(outDir, 'dir'); mkdir(outDir); end

        matPath = fullfile(outDir, 'epochs.mat');
        metaPath = fullfile(outDir, 'metadata.csv');
        auditPath = fullfile(outDir, 'bad_channel_audit.csv');
        icaPath = fullfile(outDir, 'ica_log.csv');
        chanPath = fullfile(outDir, 'channel_names.csv');

        fprintf('\n============================================================\n');
        fprintf('Exporting %s / %s\n', variant.name, wmode.name);
        fprintf('%s\n', variant.description);
        fprintf('%s\n', wmode.description);
        fprintf('============================================================\n');

        if cfg.SKIP_IF_EXISTS && exist(matPath, 'file') == 2 && exist(metaPath, 'file') == 2
            fprintf('Existing output found. Skipping: %s\n', matPath);
            info = whos('-file', matPath, 'X');
            nSamples = NaN; nChannels = NaN; nTime = NaN;
            if ~isempty(info)
                dims = info.size;
                if numel(dims) >= 3
                    nSamples = dims(1); nChannels = dims(2); nTime = dims(3);
                end
            end
            newRow = table({variant.name}, {wmode.name}, {matPath}, {metaPath}, {auditPath}, {icaPath}, {chanPath}, ...
                nSamples, nChannels, nTime, 'VariableNames', ...
                {'preprocessing','feature_mode','epoch_mat_path','metadata_path','audit_path','ica_log_path','channel_names_path','n_samples','n_channels','n_time'});
            manifest = [manifest; newRow]; 
            continue;
        end

        [X, y, labels, participants, trial_uid, window_uid, meta, auditTbl, icaTbl, activeNames] = ...
            exportEpochDataset(sessions, cfg, variant, wmode);

        fs = cfg.DEFAULT_FS; 
        class_names = cfg.CLASSES; 
        channel_names = activeNames; 
        preprocessing = variant.name; 
        feature_mode = wmode.name; 

        save(matPath, 'X', 'y', 'labels', 'participants', 'trial_uid', 'window_uid', ...
            'fs', 'class_names', 'channel_names', 'preprocessing', 'feature_mode', '-v7.3');

        writetable(meta, metaPath);
        writetable(auditTbl, auditPath);
        writetable(icaTbl, icaPath);
        writecell(activeNames(:), chanPath);

        nSamples = size(X,1); nChannels = size(X,2); nTime = size(X,3);
        fprintf('Saved %s / %s: samples=%d, channels=%d, time=%d\n', variant.name, wmode.name, nSamples, nChannels, nTime);

        newRow = table({variant.name}, {wmode.name}, {matPath}, {metaPath}, {auditPath}, {icaPath}, {chanPath}, ...
            nSamples, nChannels, nTime, 'VariableNames', ...
            {'preprocessing','feature_mode','epoch_mat_path','metadata_path','audit_path','ica_log_path','channel_names_path','n_samples','n_channels','n_time'});
        manifest = [manifest; newRow]; 
    end
end

writetable(manifest, fullfile(cfg.OUTPUT_DIR, 'paperstyle_fdf_epoch_manifest.csv'));
fprintf('\nDONE. Paper-style FD-F/Picard epoch output saved to:\n%s\n', cfg.OUTPUT_DIR);
fprintf('Next step: run the Python fully paper-style deep-learning notebook with fold-wise adaptive z-score normalization.\n');


function setupEEGLABIfAvailable(cfg)
    
    fprintf('Checking EEGLAB/runica/Picard availability...\n');

    if isfield(cfg, 'EEGLAB_PATH') && isfolder(cfg.EEGLAB_PATH)
        addpath(cfg.EEGLAB_PATH);
        fprintf('Added EEGLAB path: %s\n', cfg.EEGLAB_PATH);
    else
        if isfield(cfg, 'EEGLAB_PATH')
            fprintf('Configured EEGLAB path not found: %s\n', cfg.EEGLAB_PATH);
        end
    end

    if exist('runica', 'file') == 2
        fprintf('runica already found: %s\n', which('runica'));
        return;
    end

    if exist('eeglab', 'file') == 2
        try
            fprintf('Initializing EEGLAB in nogui mode...\n');
            eeglab('nogui');
        catch
            try
                fprintf('nogui initialization failed; initializing EEGLAB normally...\n');
                eeglab;
            catch ME
                fprintf('EEGLAB initialization failed: %s\n', ME.message);
            end
        end
    else
        fprintf('eeglab.m not found on path. Paper-style FD-F export will require Picard/runica for ICA.\n');
    end

    if exist('runica', 'file') == 2
        fprintf('runica found after EEGLAB setup: %s\n', which('runica'));
    else
        fprintf('runica still not found. If Picard is also unavailable, ICA cannot be applied.\n');
    end
    if exist('picard', 'file') == 2
        fprintf('Picard found: %s\n', which('picard'));
    else
        fprintf('Picard not found on path. Will use runica fallback unless REQUIRE_PICARD=true.\n');
    end
end



function searchRoot = unzipArchives(cfg)
    zipFiles = dir(fullfile(cfg.DATA_ROOT, '*.zip'));
    extractRoot = fullfile(cfg.OUTPUT_DIR, 'extracted_data');
    if ~exist(extractRoot, 'dir'); mkdir(extractRoot); end

    if isempty(zipFiles)
        fprintf('No ZIP files found. Assuming DATA_ROOT is already extracted.\n');
        searchRoot = cfg.DATA_ROOT;
        return;
    end

    for k = 1:numel(zipFiles)
        zpath = fullfile(zipFiles(k).folder, zipFiles(k).name);
        [~, stem, ~] = fileparts(zpath);
        outDir = fullfile(extractRoot, stem);
        if exist(outDir, 'dir') && ~isempty(dir(fullfile(outDir, '**', '*')))
            continue;
        end
        fprintf('Unzipping %s\n', zipFiles(k).name);
        if ~exist(outDir, 'dir'); mkdir(outDir); end
        unzip(zpath, outDir);
    end
    searchRoot = extractRoot;
end

function sessions = findSessions(searchRoot, cfg)
    rawFiles = dir(fullfile(searchRoot, '**', '*OpenBCI*RAW*.txt'));
    rows = {};
    for k = 1:numel(rawFiles)
        rawPath = fullfile(rawFiles(k).folder, rawFiles(k).name);
        seg1 = dir(fullfile(rawFiles(k).folder, '*training*segment*.csv'));
        seg2 = dir(fullfile(rawFiles(k).folder, '*segments*.csv'));
        segs = [seg1; seg2];
        if isempty(segs); continue; end

        [participant, label] = inferParticipantLabel(rawPath, cfg);
        if isempty(participant) || isempty(label); continue; end

        ev = dir(fullfile(rawFiles(k).folder, '*event*log*.csv'));
        eventPath = "";
        if ~isempty(ev); eventPath = string(fullfile(ev(1).folder, ev(1).name)); end

        rows(end+1, :) = {participant, label, string(rawFiles(k).folder), string(rawPath), string(fullfile(segs(1).folder, segs(1).name)), eventPath}; %#ok<AGROW>
    end

    if isempty(rows)
        sessions = cell2table(cell(0,6), 'VariableNames', {'participant','label','folder','raw_path','segments_path','event_path'});
    else
        sessions = cell2table(rows, 'VariableNames', {'participant','label','folder','raw_path','segments_path','event_path'});
        sessions = unique(sessions, 'rows');
        sessions = sortrows(sessions, {'participant','label','raw_path'});
    end
end

function [participant, label] = inferParticipantLabel(pathStr, cfg)
    txt = char(pathStr);
    participant = '';
    label = '';

    tok = regexp(txt, 'TEST[_\- ]*SUBJECT[_\- ]*(\d+)', 'tokens', 'once', 'ignorecase');
    if isempty(tok)
        tok = regexp(txt, '(?<![A-Za-z0-9])P(\d{1,2})(?![A-Za-z0-9])', 'tokens', 'once', 'ignorecase');
    end
    if ~isempty(tok)
        participant = sprintf('P%02d', str2double(tok{1}));
    end

    parts = regexp(txt, '[\\/]', 'split');
    for p = 1:numel(parts)
        clean = upper(regexprep(parts{p}, '[^A-Za-z0-9]', ' '));
        tokens = regexp(clean, '\s+', 'split');
        for c = 1:numel(cfg.CLASSES)
            if any(strcmp(tokens, cfg.CLASSES{c}))
                label = cfg.CLASSES{c};
                return;
            end
        end
    end

    tok2 = regexp(upper(txt), '(?<![A-Z])([AKNPZ])(?![A-Z])', 'tokens', 'once');
    if ~isempty(tok2); label = tok2{1}; end
end

function [X, y, labels, participants, trial_uid, window_uid, meta, auditTbl, icaTbl, activeNames] = exportEpochDataset(sessions, cfg, variant, wmode)
    [activeNames, activeIdx] = getActiveChannels(cfg, variant);
    fprintf('Active channels: %s\n', strjoin(activeNames, ', '));

    Xcells = {};
    y = [];
    labels = {};
    participants = {};
    trial_uid = {};
    window_uid = {};
    metaRows = {};
    auditRows = {};
    icaRows = {};
    totalRejected = 0;
    expectedLen = getExpectedTimeLength(cfg, variant, wmode);

    for si = 1:height(sessions)
        sess = sessions(si, :);
        try
            [raw, fs] = loadOpenBCIRaw(char(sess.raw_path), cfg);
            [badChannels, audit] = auditBadChannels(raw, cfg, variant);
            [rawP, icaInfo] = preprocessRaw(raw, fs, badChannels, cfg, variant);

            segments = readtable(char(sess.segments_path));
            required = {'trial','label','segment_type','start_time_sec','end_time_sec'};
            for rr = 1:numel(required)
                if ~ismember(required{rr}, segments.Properties.VariableNames)
                    error('Segments file missing column %s: %s', required{rr}, char(sess.segments_path));
                end
            end

            segType = lower(string(segments.segment_type));
            segLabels = string(segments.label);
            taskMask = segType == "task" & ismember(segLabels, string(cfg.CLASSES));
            taskRows = segments(taskMask, :);

            sessionX = {};
            sessionMeta = {};

            for ti = 1:height(taskRows)
                tr = taskRows(ti, :);
                trialId = double(tr.trial);
                baseTaskStart = double(tr.start_time_sec) + variant.task_trim_start;
                baseTaskEnd   = double(tr.end_time_sec)   - variant.task_trim_end;
                windows = makeTaskWindows(baseTaskStart, baseTaskEnd, wmode);
                if isempty(windows); continue; end

                for wi = 1:size(windows,1)
                    taskStart = windows(wi,1);
                    taskEnd = windows(wi,2);
                    epoch = segmentByTime(rawP, fs, taskStart, taskEnd);
                    epoch = epoch(:, activeIdx);
                    epoch = fixEpochLength(epoch, expectedLen);
                    met = epochArtifactMetrics(epoch);

                    trialUid = sprintf('%s_%s_trial%02d', char(sess.participant), char(sess.label), trialId);
                    if wmode.is_windowed
                        winUid = sprintf('%s_%s_w%02d', trialUid, wmode.name, wi);
                    else
                        winUid = trialUid;
                    end
                    classIdx = find(strcmp(cfg.CLASSES, char(string(tr.label))), 1);
                    if isempty(classIdx); continue; end

                    sessionX{end+1,1} = single(epoch'); 
                    sessionMeta(end+1,:) = {char(sess.participant), char(sess.label), char(string(tr.label)), classIdx, trialId, trialUid, wi, winUid, ...
                        fs, taskStart, taskEnd, strjoin(badChannels, ';'), char(sess.raw_path), false, ...
                        met.max_abs, met.ptp_max, met.std_median, 0.0, 0.0}; 
                end
            end

            if ~isempty(sessionMeta) && isfield(cfg, 'APPLY_BAD_EPOCH_REJECTION') && cfg.APPLY_BAD_EPOCH_REJECTION
                [sessionX, sessionMeta, rejected] = applyBadEpochRejection(sessionX, sessionMeta, cfg);
                totalRejected = totalRejected + rejected;
            else
                rejected = 0;
            end

            for k = 1:numel(sessionX)
                Xcells{end+1,1} = sessionX{k}; 
                y(end+1,1) = sessionMeta{k,4}; 
                labels{end+1,1} = sessionMeta{k,3}; 
                participants{end+1,1} = sessionMeta{k,1}; 
                trial_uid{end+1,1} = sessionMeta{k,6}; 
                window_uid{end+1,1} = sessionMeta{k,8}; 
            end
            metaRows = [metaRows; sessionMeta]; 

            auditRows(end+1,:) = {variant.name, char(sess.participant), char(sess.label), char(sess.raw_path), strjoin(badChannels, ';'), ...
                audit.max_rail_ratio, audit.min_std, audit.max_abs_robust_z_std}; 
            icaRows(end+1,:) = {variant.name, char(sess.participant), char(sess.label), strjoin(badChannels, ';'), ...
                icaInfo.ica_applied, icaInfo.ica_algorithm, icaInfo.ica_removed_components, icaInfo.ica_error}; 

            fprintf('Done %s/%s %s-%s: kept epochs=%d, bad=%s\n', variant.name, wmode.name, char(sess.participant), char(sess.label), numel(sessionX), strjoin(badChannels, ','));
        catch ME
            fprintf(2, 'FAILED %s/%s session %d: %s\n', variant.name, wmode.name, si, ME.message);
        end
    end

    if isempty(Xcells)
        X = zeros(0, numel(activeNames), expectedLen, 'single');
    else
        X = zeros(numel(Xcells), numel(activeNames), expectedLen, 'single');
        for i = 1:numel(Xcells)
            X(i,:,:) = Xcells{i};
        end
    end

    metaVars = {'participant','session_label','label','class_index','trial','trial_uid','window_id','window_uid', ...
        'fs','task_start_used','task_end_used','bad_channels','raw_path','sample_rejected_by_preproc', ...
        'sample_max_abs','sample_ptp_max','sample_std_median','sample_reject_z_std','sample_reject_z_ptp'};
    if isempty(metaRows)
        meta = cell2table(cell(0,numel(metaVars)), 'VariableNames', metaVars);
    else
        meta = cell2table(metaRows, 'VariableNames', metaVars);
    end

    auditVars = {'preprocessing','participant','label','raw_path','bad_channels','max_rail_ratio','min_std','max_abs_robust_z_std'};
    if isempty(auditRows)
        auditTbl = cell2table(cell(0,numel(auditVars)), 'VariableNames', auditVars);
    else
        auditTbl = cell2table(auditRows, 'VariableNames', auditVars);
    end

    icaVars = {'preprocessing','participant','label','bad_channels','ica_applied','ica_algorithm','ica_removed_components','ica_error'};
    if isempty(icaRows)
        icaTbl = cell2table(cell(0,numel(icaVars)), 'VariableNames', icaVars);
    else
        icaTbl = cell2table(icaRows, 'VariableNames', icaVars);
    end

    fprintf('Total rejected epochs/windows for %s/%s: %d\n', variant.name, wmode.name, totalRejected);
end

function expectedLen = getExpectedTimeLength(cfg, variant, wmode)
    fs = cfg.DEFAULT_FS;
    if wmode.is_windowed
        if isfield(wmode, 'window_samples')
            expectedLen = wmode.window_samples;
        else
            expectedLen = round(wmode.window_sec * fs);
        end
    else
        expectedLen = round(10.0 * fs);
    end
end

function [activeNames, activeIdx] = getActiveChannels(cfg, variant)
    drop = unique([cfg.GLOBAL_DROP_CHANNELS, variant.extra_drop_channels]);
    if ~isempty(variant.active_channels)
        activeNames = variant.active_channels;
        activeNames = activeNames(~ismember(activeNames, drop));
    else
        activeNames = cfg.CHANNEL_NAMES(~ismember(cfg.CHANNEL_NAMES, drop));
    end
    activeIdx = zeros(1, numel(activeNames));
    for i = 1:numel(activeNames)
        activeIdx(i) = find(strcmp(cfg.CHANNEL_NAMES, activeNames{i}), 1);
    end
end

function [raw, fs] = loadOpenBCIRaw(rawPath, cfg)
    fs = cfg.DEFAULT_FS;
    fid = fopen(rawPath, 'r');
    if fid < 0; error('Cannot open %s', rawPath); end

    numericRows = {};
    lineNo = 0;
    while true
        line = fgetl(fid);
        if ~ischar(line); break; end
        lineNo = lineNo + 1;
        if lineNo <= 30
            tok = regexp(line, 'Sample\s*Rate\s*=\s*([0-9.]+)', 'tokens', 'once', 'ignorecase');
            if ~isempty(tok); fs = str2double(tok{1}); end
        end
        line = strtrim(line);
        if isempty(line) || startsWith(line, '%'); continue; end
        parts = regexp(line, ',', 'split');
        nums = nan(1, numel(parts));
        for j = 1:numel(parts)
            nums(j) = str2double(strtrim(parts{j}));
        end
        if numel(nums) >= 17
            chans = nums(2:17);
            if sum(isfinite(chans)) >= 12
                numericRows{end+1,1} = chans; 
            end
        end
    end
    fclose(fid);

    if isempty(numericRows)
        error('No numeric EEG rows found in %s', rawPath);
    end
    raw = cell2mat(numericRows);
    if size(raw,2) ~= 16
        error('Expected 16 EEG channels, got %d', size(raw,2));
    end
end

function z = robustZ(v)
    v = double(v(:));
    medv = median(v, 'omitnan');
    madv = median(abs(v - medv), 'omitnan') + 1e-12;
    z = 0.6745 .* (v - medv) ./ madv;
end

function [badChannels, audit] = auditBadChannels(raw, cfg, variant)
    drop = unique([cfg.GLOBAL_DROP_CHANNELS, variant.extra_drop_channels]);
    stdVals = std(raw, 0, 1, 'omitnan');
    railRatio = mean(abs(raw) >= cfg.RAIL_UV, 1, 'omitnan');
    zstd = robustZ(stdVals);

    badMask = false(1, numel(cfg.CHANNEL_NAMES));
    badMask = badMask | ismember(cfg.CHANNEL_NAMES, drop);
    badMask = badMask | (railRatio > cfg.MAX_RAIL_RATIO);
    badMask = badMask | (stdVals < cfg.MIN_STD_UV);
    badMask = badMask | (abs(zstd(:)') > cfg.ROBUST_Z_STD_THRESHOLD);

    badChannels = cfg.CHANNEL_NAMES(badMask);
    audit.max_rail_ratio = max(railRatio);
    audit.min_std = min(stdVals);
    audit.max_abs_robust_z_std = max(abs(zstd));
end

function y = notchFilter(x, fs, f0, Q)
    try
        wo = f0 / (fs/2);
        bw = wo / Q;
        [b,a] = iirnotch(wo, bw);
        y = filtfilt(b, a, x);
    catch
        warning('iirnotch failed or unavailable; skipping notch filter.');
        y = x;
    end
end

function y = bandpassFilter(x, fs, lo, hi, order)
    hi = min(hi, fs/2 - 1e-3);
    [b,a] = butter(order, [lo hi] ./ (fs/2), 'bandpass');
    y = filtfilt(b, a, x);
end

function [x, icaInfo] = preprocessRaw(raw, fs, badChannels, cfg, variant)
    x = double(raw);
    x = x - median(x, 1, 'omitnan');

  
    if isfield(cfg, 'APPLY_50HZ_NOTCH') && cfg.APPLY_50HZ_NOTCH
        x = notchFilter(x, fs, cfg.NOTCH_FREQ, cfg.NOTCH_Q);
    end

   
    [x, icaInfo] = applyAutoICA(x, badChannels, cfg, variant);


    x = fdfSuppressBandFFT(x, fs, cfg.FDF_REJECT_LOW, cfg.FDF_REJECT_HIGH, cfg.FDF_TRANSITION_HZ);

 
    badIdx = find(ismember(cfg.CHANNEL_NAMES, badChannels));
    goodIdx = setdiff(1:numel(cfg.CHANNEL_NAMES), badIdx);
    if isempty(goodIdx); error('All channels marked bad; cannot reference/impute.'); end

    switch lower(variant.reference)
        case 'car'
            ref = mean(x(:, goodIdx), 2, 'omitnan');
            x = x - ref;
        case 'median'
            ref = median(x(:, goodIdx), 2, 'omitnan');
            x = x - ref;
        case 'none'
         
        otherwise
            error('Unknown reference: %s', variant.reference);
    end

    x = imputeBadChannels(x, badChannels, cfg, variant.impute_strategy);
end

function y = fdfSuppressBandFFT(x, fs, lowHz, highHz, transHz)
    
    x = double(x);
    n = size(x,1);
    Y = fft(x, [], 1);
    f = (0:n-1)' * fs / n;
    fAbs = min(f, fs - f); 

    mask = ones(n,1);
    lo0 = max(0, lowHz - transHz);
    lo1 = lowHz;
    hi1 = highHz;
    hi2 = highHz + transHz;

    
    mask(fAbs >= lo1 & fAbs <= hi1) = 0;

    
    idx = fAbs >= lo0 & fAbs < lo1;
    if any(idx)
        r = (fAbs(idx) - lo0) ./ max(eps, (lo1 - lo0));
        mask(idx) = 0.5 .* (1 + cos(pi .* r));
    end


    idx = fAbs > hi1 & fAbs <= hi2;
    if any(idx)
        r = (fAbs(idx) - hi1) ./ max(eps, (hi2 - hi1));
        mask(idx) = 0.5 .* (1 - cos(pi .* r));
    end

    Yf = Y .* mask;
    y = real(ifft(Yf, [], 1));
end

function [x, info] = applyAutoICA(x, badChannels, cfg, variant)
    info.ica_applied = false;
    info.ica_removed_components = "";
    info.ica_error = "";
    info.ica_algorithm = "none";

    if ~isfield(variant, 'apply_ica') || ~variant.apply_ica
        return;
    end

    try
        badIdx = find(ismember(cfg.CHANNEL_NAMES, badChannels));
        goodIdx = setdiff(1:numel(cfg.CHANNEL_NAMES), badIdx);
        if numel(goodIdx) < 6
            info.ica_error = "too_few_good_channels";
            return;
        end

        Xg = x(:, goodIdx);
        Xg = fillmissing(Xg, 'constant', 0);
        Xg = Xg - mean(Xg, 1);

        usePicard = isfield(variant, 'prefer_picard') && variant.prefer_picard && exist('picard', 'file') == 2;
        requirePicard = isfield(variant, 'require_picard') && variant.require_picard;

        if usePicard
           
            try
                [weights, sphere] = picard(Xg');
            catch
                try
                    [weights, sphere] = picard(Xg', 'verbose', false);
                catch MEp
                    if requirePicard
                        rethrow(MEp);
                    else
                        usePicard = false;
                    end
                end
            end
        end

        if ~usePicard
            if requirePicard
                error('Picard ICA not found or failed and REQUIRE_PICARD=true. Install Picard plugin or set REQUIRE_PICARD=false.');
            end
            if exist('runica', 'file') ~= 2
                info.ica_error = "picard_and_runica_not_found";
                return;
            end
            [weights, sphere] = runica(Xg', 'extended', 1, 'verbose', 'off');
            info.ica_algorithm = "runica_fallback";
        else
            info.ica_algorithm = "picard";
        end

        W = weights * sphere;
        S = W * Xg';
        nComp = size(S,1);

        fpRefs = [];
        goodNames = cfg.CHANNEL_NAMES(goodIdx);
        for refName = {'Fp1','Fp2'}
            ri = find(strcmp(goodNames, refName{1}), 1);
            if ~isempty(ri)
                fpRefs = [fpRefs; Xg(:,ri)']; 
            end
        end
        if isempty(fpRefs)
            info.ica_error = "no_fp_reference_available";
            return;
        end

        scores = zeros(1, nComp);
        for c = 1:nComp
            vals = [];
            for r = 1:size(fpRefs,1)
                C = corrcoef(S(c,:)', fpRefs(r,:)');
                if numel(C) >= 4 && isfinite(C(1,2))
                    vals(end+1) = abs(C(1,2)); 
                end
            end
            if isempty(vals); scores(c) = 0; else; scores(c) = max(vals); end
        end

        threshold = cfg.ICA_CORR_THRESHOLD;
        if isfield(variant, 'ica_corr_threshold'); threshold = variant.ica_corr_threshold; end
        maxRemove = cfg.ICA_MAX_COMPONENTS_REMOVED;
        if isfield(variant, 'ica_max_components_removed'); maxRemove = variant.ica_max_components_removed; end
        remove = find(scores >= threshold);
        if numel(remove) > maxRemove
            [~, ord] = sort(scores(remove), 'descend');
            remove = remove(ord(1:maxRemove));
        end

        S_clean = S;
        S_clean(remove,:) = 0;
        Xg_clean = pinv(W) * S_clean;
        x(:, goodIdx) = Xg_clean';
        info.ica_applied = true;
        info.ica_removed_components = strjoin(string(remove), ';');
    catch ME
        info.ica_error = string(ME.message);
    end
end

function x = imputeBadChannels(x, badChannels, cfg, strategy)
    badIdx = find(ismember(cfg.CHANNEL_NAMES, badChannels));
    goodIdx = setdiff(1:numel(cfg.CHANNEL_NAMES), badIdx);
    if isempty(badIdx); return; end
    if isempty(goodIdx); error('All channels marked bad; cannot impute.'); end

    switch lower(strategy)
        case 'mean_good'
            mg = mean(x(:, goodIdx), 2, 'omitnan');
            for ii = badIdx
                x(:, ii) = mg;
            end
        case 'neighbor'
            for ii = badIdx
                ch = cfg.CHANNEL_NAMES{ii};
                neighIdx = [];
                if isKey(cfg.NEIGHBORS, ch)
                    neigh = cfg.NEIGHBORS(ch);
                    neigh = neigh(~ismember(neigh, badChannels));
                    for k = 1:numel(neigh)
                        idx = find(strcmp(cfg.CHANNEL_NAMES, neigh{k}), 1);
                        if ~isempty(idx); neighIdx(end+1) = idx; end 
                    end
                end
                if ~isempty(neighIdx)
                    x(:, ii) = mean(x(:, neighIdx), 2, 'omitnan');
                else
                    x(:, ii) = mean(x(:, goodIdx), 2, 'omitnan');
                end
            end
        case 'zero'
            x(:, badIdx) = 0;
        otherwise
            error('Unknown impute strategy: %s', strategy);
    end
end

function windows = makeTaskWindows(taskStart, taskEnd, wmode)
    if ~wmode.is_windowed
        windows = [taskStart, taskEnd];
        return;
    end
    dur = taskEnd - taskStart;
    if dur < wmode.window_sec
        windows = [];
        return;
    end
    step = wmode.step_sec;
    starts = taskStart:step:(taskEnd - wmode.window_sec + 1e-9);
    windows = [starts(:), starts(:) + wmode.window_sec];
end

function seg = segmentByTime(x, fs, startSec, endSec)
    n = size(x,1);
    s = max(1, round(startSec * fs) + 1);
    e = min(n, round(endSec * fs));
    if e <= s
        seg = zeros(0, size(x,2));
    else
        seg = x(s:e, :);
    end
end

function epoch = fixEpochLength(epoch, expectedLen)
    n = size(epoch,1);
    if n == expectedLen
        return;
    elseif n > expectedLen
        epoch = epoch(1:expectedLen, :);
    else
        pad = zeros(expectedLen - n, size(epoch,2));
        epoch = [epoch; pad];
    end
end

function met = epochArtifactMetrics(epoch)
    met.max_abs = max(abs(epoch), [], 'all', 'omitnan');
    ptpVals = max(epoch, [], 1, 'omitnan') - min(epoch, [], 1, 'omitnan');
    met.ptp_max = max(ptpVals, [], 'omitnan');
    met.std_median = median(std(epoch, 0, 1, 'omitnan'), 'omitnan');
end

function [sessionX, sessionMeta, rejected] = applyBadEpochRejection(sessionX, sessionMeta, cfg)
    if isempty(sessionMeta)
        rejected = 0;
        return;
    end
    stdVals = cell2mat(sessionMeta(:,17));
    ptpVals = cell2mat(sessionMeta(:,16));
    maxAbsVals = cell2mat(sessionMeta(:,15));
    zStd = robustZ(stdVals);
    zPtp = robustZ(ptpVals);
    keep = abs(zStd) <= cfg.BAD_EPOCH_ROBUST_Z & abs(zPtp) <= cfg.BAD_EPOCH_ROBUST_Z & maxAbsVals(:) <= cfg.BAD_EPOCH_MAX_ABS_UV;

    if sum(keep) < cfg.MIN_KEEP_PER_CLASS_SESSION
        keep(:) = true;
    end

    rejected = sum(~keep);
    for i = 1:numel(keep)
        sessionMeta{i,14} = ~keep(i);
        sessionMeta{i,18} = zStd(i);
        sessionMeta{i,19} = zPtp(i);
    end
    sessionX = sessionX(keep);
    sessionMeta = sessionMeta(keep, :);
end
