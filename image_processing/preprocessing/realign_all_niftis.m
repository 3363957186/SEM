function rids = realign_all_niftis(cohort, processedfolder, modality)
addpath('/home/varunaja/spm12')

if isempty(modality)
    modality = 'T1';
end

outputfolder=[processedfolder filesep 'realigned']

if strcmpi(modality, 'T1')
    % MRI folder
    folder_init = [processedfolder filesep 'reoriented/T1_nii/'];
    % recentered MRI folder
    folder_new = [outputfolder filesep cohort '_T1_nii_recenter/'];
elseif strcmpi(modality, 'Amyloid')
    % Amyloid folder
    folder_init = [processedfolder filesep 'reoriented/Amyloid_nii/'];
    % recentered Amyloid folder
    folder_new = [outputfolder filesep cohort '_Amyloid_nii_recenter/'];
elseif strcmpi(modality, 'Tau')
    % Tau folder
    folder_init = [processedfolder filesep 'reoriented/Tau_nii/'];
    % recentered Tau folder
    folder_new = [outputfolder filesep cohort '_Tau_nii_recenter/'];
else
    error('Invalid modality specified.');
end

fnames = dir(folder_init);
fnames = {fnames.name};
rids = cell(1, numel(fnames));
for i = 1:numel(fnames)
    rids{i} = strtok(fnames{i}, '_');  % Extract the part before the first underscore
end

% Initialize a logical array to identify cells with numbers
isNumeric = false(size(rids));

% Loop through the 'rids' cell array and identify cells with numbers
for i = 1:numel(rids)
    isNumeric(i) = any(isstrprop(rids{i}, 'digit'));
end

% Filter 'rids' to keep only cells with numbers
filteredRids = rids(isNumeric);
rids = filteredRids

% output folder is realigned folder
if ~exist(outputfolder,'dir')
    mkdir(outputfolder)
end
% folder_new is modality specific folder in the realigne folder
if ~exist(folder_new,'dir')
    mkdir(folder_new);
end

% realign_niftin function takes raw folder, output recentered folder, rids and modality
realign_nifti(outputfolder, folder_init, folder_new, rids, modality);