% read keras weights
function netw=readweights()

cd(fullfile(asnn_root, 'examples', 'cifar'))
fitDfile = dir(fullfile('*.h5'));
info = h5info(fitDfile.name);
cd(asnn_root)
%%
netw = struct();
netw.batchn = struct();
ibn = 0;
netw.conv = struct();
ic2d = 0;
netw.dense = struct();
id = 0;

for i=1:numel(info.Groups(1).Groups)
    if ~isempty(strfind(info.Groups(1).Groups(i).Name,'batchnormalization_'))
        ibn = ibn+1;
        if ibn == 1
            ibi = i;
        end
    end
    if ~isempty(strfind(info.Groups(1).Groups(i).Name,'convolution2d_'))
        ic2d = ic2d+1;
    end
    if ~isempty(strfind(info.Groups(1).Groups(i).Name,'dense_'))
        id = id+1;
    end
end
netw.countbatch = ibn;
netw.countconv = ic2d;
netw.countdense = id;
%%
for ibn=1:netw.countbatch %numel(info.Groups(1).Groups)
    
     ibns = num2str(ibn);
     netw.batchn(ibn).beta =  h5read(fitDfile.name,['/model_weights/batchnormalization_' ibns '/batchnormalization_' ibns '_beta:0']);
     netw.batchn(ibn).gamma =  h5read(fitDfile.name,['/model_weights/batchnormalization_' ibns '/batchnormalization_' ibns '_gamma:0']);
     netw.batchn(ibn).runningmean =  h5read(fitDfile.name,['/model_weights/batchnormalization_' ibns '/batchnormalization_' ibns '_running_mean:0']);
     netw.batchn(ibn).runningstd =  h5read(fitDfile.name,['/model_weights/batchnormalization_' ibns '/batchnormalization_' ibns '_running_std:0']);

end

for ic2d=1:netw.countconv
    
    ic2ds = num2str(ic2d);
    netw.conv(ic2d).weights =  h5read(fitDfile.name,['/model_weights/convolution2d_' ic2ds '/convolution2d_' ic2ds '_W:0']);
    netw.conv(ic2d).bias =  h5read(fitDfile.name,['/model_weights/convolution2d_' ic2ds '/convolution2d_' ic2ds '_b:0']);
    
    netw.countconv = ic2d;
    
end

for id=1:netw.countdense

    ids = num2str(id);
    netw.dense(id).weights =  h5read(fitDfile.name,['/model_weights/dense_' ids '/dense_' ids '_W:0']);
    netw.dense(id).bias =  h5read(fitDfile.name,['/model_weights/dense_' ids '/dense_' ids '_b:0']);
    
    netw.countdense = id;

end
