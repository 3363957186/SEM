function gaain_klunk_coreg(BASE_DIR)
addpath('/home/varunaja/spm12/');
%BASE_DIR = ['/SeaExp_1/MRI_PET/GAAIN/FLU/processed'];

recentered_folder = [BASE_DIR filesep 'realigned'];
t1_folder_recentered = [recentered_folder filesep 'GAAIN_T1_nii_recenter'];
pet_folder_recentered = [recentered_folder filesep 'GAAIN_Amyloid_nii_recenter'];

% define coreg_coseg output folder
mkdir([BASE_DIR filesep 'coreg_klunk']);
out_folder = [BASE_DIR filesep 'coreg_klunk/GAAIN_T1_Amyloid'];

% create output folder
mkdir([out_folder]);

% filenames 
fnames = dir(t1_folder_recentered);
fnames = {fnames.name};
excludeNames = {'.', '..'};

% Create a logical index to exclude specified names
excludeIndex = ismember(fnames, excludeNames);

% Filter fnames to exclude specified names
filteredFnames = fnames(~excludeIndex);

% Copy recentered T1 mri and pet image to processing out folder
rand('state',10);
maxNumCompThreads = 1;
spm('defaults', 'PET');
spm_jobman('initcfg');
parpool(5);
parfor i=1:length(filteredFnames) 
    currentLine = filteredFnames{i};
    % split line based on the underscore delimiter
    values = strsplit(currentLine, '_');
    % extract values and convert them to the appropriate data type
    rid = values{1};
    pet_files = dir(fullfile(pet_folder_recentered, [rid '*']));
    t1_files = dir(fullfile(t1_folder_recentered, [rid '*']));
    pet_name = pet_files(1).name;
    t1_name = t1_files(1).name;
    system(['rsync -av ' fullfile(pet_folder_recentered, pet_name) ' ' out_folder]);
    system(['rsync -av ' fullfile(t1_folder_recentered, t1_name) ' ' out_folder]);
    % run batch_process_pet_t1
    jobs = batch_process_pet_t1(rid, pet_name, t1_name, out_folder);
    spm_jobman('run', jobs);
    disp(['Processing completed: ' rid]);
end
end

%-----------------------------------------------------------------------
function matlabbatch = batch_process_pet_t1(rid, pet_filename, t1_filename,  out_folder)
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
