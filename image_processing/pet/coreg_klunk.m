function coreg_klunk(cohort, BASE_DIR, csv_list)
addpath('/home/varunaja/spm12/');
%BASE_DIR = ['/SeaExp_1/MRI_PET/AIBL/processed'];

recentered_folder = [BASE_DIR filesep 'realigned'];
T1_folder_recentered = [recentered_folder filesep cohort '_T1_nii_recenter'];
pet_folder_recentered = [recentered_folder filesep cohort '_Amyloid_nii_recenter'];

% define coreg_coseg output folder
mkdir([BASE_DIR filesep 'coreg_klunk']);
out_folder = [BASE_DIR filesep 'coreg_klunk/' cohort '_T1_Amyloid'];
% create output folder
mkdir([out_folder]);
% create log folder
logFolder = [out_folder filesep 'logs'];
mkdir([logFolder]);

% read file list that matches RIDs to unique amyloid identifier and corresponding T1 identifier
% because we are processing amyloid-t1 pairs and one subject can have multiple pairs 
fileID = fopen(csv_list, 'r');
C = textscan(fileID, '%s', 'Delimiter', '\n');
lines = C{1};
fclose(fileID);

% iterate over each line (skipping the header)
% Copy recentered T1 mri and pet image to processing out folder
rand('state',10);
maxNumCompThreads = 1;
spm('defaults', 'PET');
spm_jobman('initcfg');
parpool(6);
parfor i = 2:length(lines)  % start from the second line to skip the header
    currentLine = lines{i};
    % split line based on the comma delimiter
    values = strsplit(currentLine, ',');
    % extract values and convert them to the appropriate data type
    rid = values{1};
    loni_pet_id = values{2};
    loni_t1_id = values{3};

    % check if loni_pet_id exists in pet_folder_recentered
    pet_files = dir(fullfile(pet_folder_recentered, ['*' rid '*' loni_pet_id '*']));
    if ~isempty(pet_files)
        % check if loni_t1_id exists in T1_folder_recentered
        t1_files = dir(fullfile(T1_folder_recentered, ['*' rid '*' loni_t1_id '*']));
        if ~isempty(t1_files)
            try
                % Copy the found pairs to out_folder
                pet_name = pet_files(1).name;
                t1_name = t1_files(1).name;
                % create unique log file name per parallel worker
                logFileName = fullfile(logFolder, sprintf('log_worker_%d.txt', i));
                logFileID = fopen(logFileName, 'w');
                fprintf(logFileID, '=============================================\n');
                fprintf(logFileID, 'Processing %s with %s\n', pet_name, t1_name);
                system(['rsync -av ' fullfile(pet_folder_recentered, pet_name) ' ' out_folder]);
                fprintf(logFileID, 'rsync -av %s %s\n', fullfile(pet_folder_recentered, pet_name), out_folder);
                system(['rsync -av ' fullfile(T1_folder_recentered, t1_name) ' ' out_folder]);
                fprintf(logFileID, 'rsync -av %s %s\n', fullfile(T1_folder_recentered, t1_name), out_folder);
                % run batch_process_pet_t1
                jobs = batch_process_pet_t1(rid, loni_pet_id, loni_t1_id, pet_name, t1_name, out_folder);
                spm_jobman('run', jobs);
                disp(['Processing Complete Amyloid Scan: ' rid '_' loni_pet_id]);
                fprintf(logFileID, 'Processing Complete Amyloid Scan: %s\n', [rid '_' loni_pet_id]);
                fclose(logFileID);
            catch exception
                disp(['Error Processing Amyloid Scan: ' rid '_' loni_pet_id]);
                disp(exception);
                % Generate a unique error log file name for each parallel worker
                errorLogFileName = fullfile(logFolder, sprintf('error_log_worker_%d.txt', i));
                errorLogFileID = fopen(errorLogFileName, 'w');
                fprintf(errorLogFileID, 'Error Processing Amyloid Scan: %s_%s\n', rid, loni_pet_id);
                fprintf(errorLogFileID, '%s\n', exception.getReport());
                fclose(errorLogFileID);
            end
        end
    end
end
end
%-----------------------------------------------------------------------
function matlabbatch = batch_process_pet_t1(rid, loni_pet_id, loni_t1_id, pet_filename, t1_filename, out_folder)
pet_file = spm_select('FPList', out_folder, ['^.*' pet_filename '.*$']);
t1_file = spm_select('FPList', out_folder, ['^.*' t1_filename '.*$']);

%Co-register the MRI to the MNI brain:
%Reference Image: MNI-152 template; Source Image: MRI
matlabbatch{1}.spm.spatial.coreg.estimate.ref = {'/home/varunaja/spm12/toolbox/cat12/templates_MNI152NLin2009cAsym/Template_T1.nii'};
matlabbatch{1}.spm.spatial.coreg.estimate.source = {t1_file};
matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];

%Co-register the pet to the MRI that is coregistered to MNI template. 
%Reference Image: MRI in MNI ; Source Image: PET
matlabbatch{2}.spm.spatial.coreg.estimate.ref(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
matlabbatch{2}.spm.spatial.coreg.estimate.source = {pet_file};
matlabbatch{2}.spm.spatial.coreg.estimate.other = {''};
matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{2}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];

% Process the MRI coregistered to MNI Space
matlabbatch{3}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
matlabbatch{3}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{3}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{3}.spm.spatial.preproc.channel.write = [0 1];
matlabbatch{3}.spm.spatial.preproc.tissue(1).tpm = {'/home/varunaja/spm12/tpm/TPM.nii,1'};
matlabbatch{3}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{3}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{3}.spm.spatial.preproc.tissue(1).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(2).tpm = {'/home/varunaja/spm12/tpm/TPM.nii,2'};
matlabbatch{3}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{3}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{3}.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(3).tpm = {'/home/varunaja/spm12/tpm/TPM.nii,3'};
matlabbatch{3}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{3}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{3}.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(4).tpm = {'/home/varunaja/spm12/tpm/TPM.nii,4'};
matlabbatch{3}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{3}.spm.spatial.preproc.tissue(4).native = [1 0];
matlabbatch{3}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(5).tpm = {'/home/varunaja/spm12/tpm/TPM.nii,5'};
matlabbatch{3}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{3}.spm.spatial.preproc.tissue(5).native = [1 0];
matlabbatch{3}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(6).tpm = {'/home/varunaja/spm12/tpm/TPM.nii,6'};
matlabbatch{3}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{3}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{3}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{3}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{3}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{3}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{3}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{3}.spm.spatial.preproc.warp.write = [0 1];
matlabbatch{3}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{3}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
% normalize to mni
matlabbatch{4}.spm.spatial.normalise.write.subj(1).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
matlabbatch{4}.spm.spatial.normalise.write.subj(1).resample(1) = cfg_dep('Segment: Bias Corrected (1)', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','channel', '()',{1}, '.','biascorr', '()',{':'}));
matlabbatch{4}.spm.spatial.normalise.write.subj(2).def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
matlabbatch{4}.spm.spatial.normalise.write.subj(2).resample(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
matlabbatch{4}.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72; 90 90 108];
matlabbatch{4}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
matlabbatch{4}.spm.spatial.normalise.write.woptions.interp = 4;
matlabbatch{4}.spm.spatial.normalise.write.woptions.prefix = 'w';
end
