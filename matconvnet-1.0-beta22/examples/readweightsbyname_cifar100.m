% read keras weights
function netw=readweightsbyname_cifar100(h5filename)


% 
% cd(asnn_root)
% addpath(genpath('./'));


% insert here the full path h5filename = 'single_gpu_model.hdf5'
% h5filename = 'single_gpu_asn_model.hdf5'
% h5filename = 'Cifar100_vgg12_63.31.h5'
% cd ('../../../python/weights')
cd ('/Volumes/Data/python/weights')
% fitDfile = dir(fullfile('*.h5'));

fitDfile = dir(fullfile(h5filename));
info = h5info(fitDfile.name);

%%
netw = struct();
netw.batchn = struct();
ibn = 0;
netw.conv = struct();
ic2d = 0;
netw.dense = struct();
id = 0;
iblock = 0;

for i=1:numel(info.Groups(1).Groups)
    if ~isempty(strfind(info.Groups(1).Groups(i).Name,'batch_normalization_'))
        ibn = ibn+1;
        if ibn == 1
            ibi = i;
        end
    end
    
    if ~isempty(strfind(info.Groups(1).Groups(i).Name,'block'))
        if str2num(info.Groups(1).Groups(i).Name(21)) == numel(iblock)
            if ~isempty(strfind(info.Groups(1).Groups(i).Name,['conv']))
                iblock(end) = iblock(end) + 1;
            end
        else
            if ~isempty(strfind(info.Groups(1).Groups(i).Name,['conv']))
                iblock = [iblock 1];
            end
        end

        
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
     netw.batchn(ibn).beta =  h5read(fitDfile.name,['/model_weights/batch_normalization_' ibns '/batch_normalization_' ibns '/beta:0']);
     netw.batchn(ibn).gamma =  h5read(fitDfile.name,['/model_weights/batch_normalization_' ibns '/batch_normalization_' ibns '/gamma:0']);
     netw.batchn(ibn).runningmean =  h5read(fitDfile.name,['/model_weights/batch_normalization_' ibns '/batch_normalization_' ibns '/moving_mean:0']);
     netw.batchn(ibn).runningstd =  h5read(fitDfile.name,['/model_weights/batch_normalization_' ibns '/batch_normalization_' ibns '/moving_variance:0']);

end

for ic2d=1:netw.countconv
    
    ic2ds = num2str(ic2d);
    netw.conv(ic2d).weights =  h5read(fitDfile.name,['/model_weights/conv2d_' ic2ds '/conv2d_' ic2ds '/kernel:0']);
    netw.conv(ic2d).bias =  h5read(fitDfile.name,['/model_weights/conv2d_' ic2ds '/conv2d_' ic2ds '/bias:0']);
    
    netw.countconv = ic2d;
    
end

% for ic2d=1:sum(iblock)
ic2d = 0;

for ib = 1:numel(iblock)
    
    for jb = 1:iblock(ib)
        ic2d = ic2d+1;
        ibs = num2str(ib);
        jbs = num2str(jb);
        
        netw.conv(ic2d).weights =  h5read(fitDfile.name,...
            ['/model_weights/block' ibs '_conv' jbs '/block' ibs '_conv' jbs '/kernel:0']);
        netw.conv(ic2d).bias =  h5read(fitDfile.name,...
            ['/model_weights/block' ibs '_conv' jbs '/block' ibs '_conv' jbs '/bias:0']);
        
        netw.countconv = ic2d;
    end
end
    
% end



for id=1:netw.countdense

    ids = num2str(id);
    netw.dense(id).weights =  h5read(fitDfile.name,['/model_weights/dense_' ids '/dense_' ids '/kernel:0']);
    netw.dense(id).bias =  h5read(fitDfile.name,['/model_weights/dense_' ids '/dense_' ids '/bias:0']);
    
    netw.countdense = id;

end

%%
% return to the working dir
cd(asnn_root)



%%
% 
% ibns = num2str(ibn);
%      netw.batchn(ibn).beta =  h5read(fitDfile.name,['/model_weights/batchnormalizationtheta_' ibns '/batchnormalizationtheta_' ibns '_beta:0']);
%      netw.batchn(ibn).gamma =  h5read(fitDfile.name,['/model_weights/batchnormalizationtheta_' ibns '/batchnormalizationtheta_' ibns '_gamma:0']);
%      netw.batchn(ibn).runningmean =  h5read(fitDfile.name,['/model_weights/batchnormalizationtheta_' ibns '/batchnormalizationtheta_' ibns '_running_mean:0']);
%      netw.batchn(ibn).runningstd =  h5read(fitDfile.name,['/model_weights/batchnormalizationtheta_' ibns '/batchnormalizationtheta_' ibns '_running_std:0']);