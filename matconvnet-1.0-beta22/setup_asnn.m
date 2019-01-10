function setup_asnn()
%SETUP_ASNN Add the AdSNN code and MatConvNet to the MATLAB path.
%   Run once per MATLAB session, from any folder:
%     >> run('path/to/cwi-asnn/matconvnet-1.0-beta22/setup_asnn.m')
%   MatConvNet must be compiled first (see vl_compilenn and the README).
root = asnn_root();
addpath(genpath(root));
run(fullfile(root, 'matlab', 'vl_setupnn.m'));
cd(root);
end
