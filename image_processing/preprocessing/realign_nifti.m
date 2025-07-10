function fnames = realign_nifti(outputfolder, raw_nifti_folder, recentered_folder_name, rid_list, modality)
if ~exist(recentered_folder_name,'dir')
    mkdir(recentered_folder_name);
end

listing = dir(raw_nifti_folder);
fnames = {};
% initialize tmp folder
tmp = [outputfolder filesep 'tmp'];
if ~exist(tmp, 'dir')
    mkdir(tmp)
end

for i=1:length(listing)
    % find nii
    if ~listing(i).isdir && strcmp(listing(i).name(end-3:end), '.nii')
        rid = strtok(listing(i).name, '_')
	% fname: path to raw nifti
        fname = join([listing(i).folder filesep listing(i).name],'');
	% copy raw nii to a tmp folder, which is one back from target recentered folder
        system(['rsync -a ' fname ' ' recentered_folder_name '/../tmp/']);
	    % run SetOriginToCenter.m
        Vo = SetOriginToCenter([recentered_folder_name '/../tmp/' listing(i).name]);
        [data] = spm_read_vols(Vo);
        % append realigned at the end of the filename
        [path, filename, ext] = fileparts(listing(i).name);
        new_filename = [filename '_realigned' ext];
        Vo.fname = [recentered_folder_name filesep new_filename];
        fnames = [fnames, Vo.fname];
        if ~exist(Vo.fname, 'file')
            spm_write_vol(Vo, data);
        end
    else
        disp(['skipping ' listing(i).folder filesep listing(i).name])
    end
end
