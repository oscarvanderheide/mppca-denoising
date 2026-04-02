function run_matlab_reference(output_dir, toolbox_dir)
%RUN_MATLAB_REFERENCE Run Tensor-MP-PCA MATLAB reference on prepared cases.
%
% Usage from a shell:
%   matlab -batch "addpath('/path/to/repo/scripts'); run_matlab_reference('/path/to/output', '/path/to/Tensor-MP-PCA')"

if nargin < 1 || strlength(string(output_dir)) == 0
    output_dir = fullfile(pwd, "debug", "matlab_compare");
end
if nargin < 2 || strlength(string(toolbox_dir)) == 0
    toolbox_dir = "/home/oheide/Documents/MATLAB/Tensor-MP-PCA";
end

output_dir = char(string(output_dir));
toolbox_dir = char(string(toolbox_dir));

manifest_path = fullfile(output_dir, "manifest.json");
if ~isfile(manifest_path)
    error("Manifest not found: %s", manifest_path);
end
if ~isfolder(toolbox_dir)
    error("Tensor-MP-PCA toolbox directory not found: %s", toolbox_dir);
end

addpath(toolbox_dir);
if exist("denoise_recursive_tensor", "file") ~= 2
    error("Could not find denoise_recursive_tensor.m in %s", toolbox_dir);
end

manifest = jsondecode(fileread(manifest_path));
window = double(manifest.window(:)');
num_spatial_dims = double(manifest.num_spatial_dims);
opt_shrink = logical(manifest.opt_shrink);

fprintf("%s\n", repmat('=', 1, 72));
fprintf("Running MATLAB Tensor-MP-PCA reference\n");
fprintf("%s\n", repmat('=', 1, 72));
fprintf("Output directory : %s\n", output_dir);
fprintf("Toolbox          : %s\n", toolbox_dir);
fprintf("Window           : [%s]\n", num2str(window));
fprintf("Opt shrink       : %d\n\n", opt_shrink);

for idx = 1:numel(manifest.cases)
    case_info = manifest.cases(idx);
    case_name = char(string(case_info.name));
    input_path = fullfile(output_dir, char(string(case_info.input_mat)));
    result_path = fullfile(output_dir, char(string(case_info.matlab_output_mat)));
    runtime_path = fullfile(output_dir, char(string(case_info.matlab_runtime_json)));

    fprintf("%s:\n", case_name);
    if ~isfile(input_path)
        error("Input .mat not found for %s: %s", case_name, input_path);
    end

    in = load(input_path, "signal_noisy");
    noisy = in.signal_noisy;
    measurement_shape = double(case_info.measurement_shape(:)');
    indices = build_indices(num_spatial_dims, numel(measurement_shape));

    tic;
    [denoised, Sigma2, P, SNR_gain] = denoise_recursive_tensor( ...
        noisy, ...
        window, ...
        "indices", indices, ...
        "opt_shrink", opt_shrink ...
    );
    runtime_seconds = toc;

    save(result_path, "denoised", "Sigma2", "P", "SNR_gain", "runtime_seconds", "-v7");
    runtime_info = struct( ...
        "case", case_name, ...
        "runtime_seconds", runtime_seconds, ...
        "measurement_shape", measurement_shape, ...
        "opt_shrink", opt_shrink ...
    );
    write_text(runtime_path, jsonencode(runtime_info));

    fprintf("  noisy shape     : %s\n", mat2str(size(noisy)));
    fprintf("  result saved to : %s\n", result_path);
    fprintf("  runtime         : %.4fs\n\n", runtime_seconds);
end
end


function indices = build_indices(num_spatial_dims, num_measurement_dims)
indices = cell(1, num_measurement_dims + 1);
indices{1} = 1:num_spatial_dims;
for k = 1:num_measurement_dims
    indices{k + 1} = num_spatial_dims + k;
end
end


function write_text(path, text)
folder = fileparts(path);
if strlength(string(folder)) > 0 && ~isfolder(folder)
    mkdir(folder);
end
fid = fopen(path, "w");
if fid == -1
    error("Could not open %s for writing", path);
end
cleaner = onCleanup(@() fclose(fid));
fprintf(fid, "%s\n", text);
end
