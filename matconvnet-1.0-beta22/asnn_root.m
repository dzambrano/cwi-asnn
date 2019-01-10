function root = asnn_root()
%ASNN_ROOT Return the root folder of this code base (matconvnet-1.0-beta22).
%   Used by the experiment scripts instead of hard-coded absolute paths.
root = fileparts(mfilename('fullpath'));
end
