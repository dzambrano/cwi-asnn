

clear all
close all

cd(asnn_root)
addpath(genpath('./'));

run  matlab/vl_setupnn




%% load high frequency res
fitDfile = dir(fullfile(pwd,'/data/imagenet-keras-resnet18/imagenet-keras-valid-resnet18-SUMSPIKE-b*-0.05*'));



im_converg_time = [];
im_firing_rate = [];
im_val_accuracy = [];
im_tmv = [];
im_mean_acc = [];
im_std_acc = [];

spred = [];
slab = [];
testHfr = [];


er = 0.370; % 62.98% - top5 85%
P1er = er + 1*er/100;

for i = 1:numel(fitDfile)
%     savestring = [fitDfile(i).folder '/' fitDfile(i).name];
    savestring = fitDfile(i).name;
    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    
    im_converg_time = [im_converg_time, sum(snet.best,2)];
    im_firing_rate = [im_firing_rate, round(snet.firingrate,1)];
    im_val_accuracy = [im_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(im_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(im_converg_time(:,i));
    end
    im_tmv = [im_tmv, tm];
    im_mean_acc = [im_mean_acc, (1-mean(im_converg_time(tm:end,i)))];
    im_std_acc = [im_std_acc, std(im_converg_time(tm:end,i))];
    
    spred = [spred single(sinfo_rl.val.pred(1,:))];
    slab = [slab sinfo_rl.val.lab(1,:)];
    testHfr = [testHfr;single(ssnet.testPreSML)];
    

    
end


% spred = sinfo_rl.val.pred(1,:);
% slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim = find(ser == 0);
% testHfr = ssnet.testPreSML;

display(sum(ser)/50000)
%%

%% tot numb of neurons and sum spikes
fitDfile = dir(fullfile(pwd,'/data/imagenet-keras-resnet18/imagenet-keras-valid-resnet18-SUMSPIKE-b*-0.05*'));
tot_neurons = 0;
sum_spikes = 0;
sumSpikesTime = 0;
for ifo = 1:numel(fitDfile)
        
    %     savestring = [fitDfile(ifo).folder '/' fitDfile(ifo).name];
    savestring = fitDfile(ifo).name;

    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    tot_neurons = 0;
    
    for i=1:numel(ssnet.layers)
        %     display(ssnet.layers{i}.type)
        if strcmp(ssnet.layers{i}.type,'snn_relu_th')
            sum_spikes = sum_spikes + sum(double(ssnet.layers{i}.tot_spikes(:))) ;
            tot_neurons = tot_neurons + numel(ssnet.layers{i}.tot_spikes) ;
        end
    end
    sumSpikesTime = sumSpikesTime +  sum(ssnet.sumSpikesTime);
    
end

sum_spikes_im = sum_spikes/50000;
avg_firing_rate = sum_spikes_im / tot_neurons;

sum_spikes_im_b = sumSpikesTime/50000;




%% Fan out
% fitDfile = dir(fullfile(pwd,'/data/imagenet-keras-resnet18/imagenet-keras-valid-resnet18-SUMSPIKE-b*-0.05*'));
fitDfile = dir(fullfile(pwd,'/data/imagenet-keras-resnet18/imagenet-keras-valid-resnet18-AROUSAL-SUMSPIKES-b*-0.15*'));

tot_neurons = 0;
sum_spikes = 0;
sumSpikesTime = 0;
net = cnn_imagenet_init_keras;
sum_spikes_v = zeros(1, numel(net.layers));

for ifo = 1:numel(fitDfile)
        
    %     savestring = [fitDfile(ifo).folder '/' fitDfile(ifo).name];
    savestring = fitDfile(ifo).name;
    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    tot_neurons = 0;
    
    for i=1:numel(ssnet.layers)
        %     display(ssnet.layers{i}.type)
        if strcmp(ssnet.layers{i}.type,'snn_relu_th')
            sum_spikes = sum_spikes + sum(double(ssnet.layers{i}.tot_spikes(:))) ;
            sum_spikes_v(i) = sum_spikes_v(i) + sum(double(ssnet.layers{i}.tot_spikes(:))) ;
            tot_neurons = tot_neurons + numel(ssnet.layers{i}.tot_spikes) ;
        end
    end
    sumSpikesTime = sumSpikesTime +  ssnet.sumSpikesTime;
    
end

sum_spikes_im = sum_spikes/50000;
avg_firing_rate = sum_spikes_im / tot_neurons;

sum_spikes_im_b = sumSpikesTime/50000;
%%
% 
% for i=1:numel(ssnet.layers)
%     display([ssnet.layers{i}.type ' ' num2str(sum_spikes_v(i))])
% end

sum_spikes_im_MT =sum(sumSpikesTime(1:460))/50000;
rats = sum_spikes_im_MT/sum_spikes_im;

% spike_fo_MT = spike_fo * rats;



%%

% net = cnn_imagenet_init_keras;

fanout = zeros(1, numel(net.layers));

for i=1:numel(net.layers)
    
    if strcmp(net.layers{i}.type,'asn')        
        if strcmp(net.layers{i+1}.type,'conv')
            
            siz_out = size(net.layers{i+1}.weights{1});
            fanout(i)=prod(siz_out([1,2,4]));
            
        elseif strcmp(net.layers{i+1}.type,'pool')
            
            if strcmp(net.layers{i+2}.type,'shortcut')
                
                siz_out = size(net.layers{i+3}.weights{1});
                fanout(i)=prod(siz_out([1,2,4]));                
            elseif strcmp(net.layers{i+2}.type,'conv')   
            
                siz_out = size(net.layers{i+2}.weights{1});
                fanout(i)=prod(siz_out([1,2,4]));                
            end
            
        end
        
    end

end
%
spike_fo = 0;

for i=1:numel(ssnet.layers)
%     display(ssnet.layers{i}.type)
    if strcmp(ssnet.layers{i}.type,'snn_relu_th')                
        spike_fo = spike_fo + sum_spikes_v(i) * fanout(i);
    end
  
end

spike_fo = spike_fo/50000;
spike_fo_MT = spike_fo * rats;


%%

for ifo = 1:numel(fitDfile)
        
    savestring = [fitDfile(ifo).folder '/' fitDfile(ifo).name];
    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    tot_neurons = 0;
    
    for i=1:numel(ssnet.layers)
        %     display(ssnet.layers{i}.type)
        if strcmp(ssnet.layers{i}.type,'snn_relu_th')
            sum_spikes = sum_spikes + sum(double(ssnet.layers{i}.tot_spikes(:))) ;
            tot_neurons = tot_neurons + numel(ssnet.layers{i}.tot_spikes) ;
        end
    end
    sumSpikesTime = sumSpikesTime +  sum(ssnet.sumSpikesTime);
    
end

sum_spikes_im = sum_spikes/50000;
avg_firing_rate = sum_spikes_im / tot_neurons;

sum_spikes_im_b = sumSpikesTime/50000;



% 
% fanout = zeros(1, numel(net.layers));
% 
% for i=1:numel(net.layers)
%     
%     if strcmp(net.layers{i}.type,'asn')        
%         if strcmp(net.layers{i+1}.type,'conv')
%             siz_out = size(net.layers{i+1}.weights{1});
%             fanout(i)=prod(siz_out([1,2,4]));
%         elseif strcmp(net.layers{i+1}.type,'pool')
%             siz_out = size(net.layers{i+2}.weights{1});
%             fanout(i)=prod(siz_out([1,2,4]));
%         end
%         
%     end
% 
% end
% spike_fo = 0;
% 
% for i=1:numel(ssnet.layers)
% %     display(ssnet.layers{i}.type)
%     if strcmp(ssnet.layers{i}.type,'snn_relu_th')        
%         sum_spikes = sum_spikes + sum(double(ssnet.layers{i}.tot_spikes(:))) ;   
%         tot_neurons = tot_neurons + numel(ssnet.layers{i}.tot_spikes) ;                       
%         spike_fo = spike_fo + sum(double(ssnet.layers{i}.tot_spikes(:))) * fanout(i);
%     end
% end
% 
% spike_fo = spike_fo / 8000;
% 
% 
% sum_spikes_im = sum_spikes/8000;
% avg_firing_rate = (sum_spikes_im / tot_neurons) / 0.5;
% 
% sum_spikes_im_b = sum(ssnet.sumSpikesTime)/8000;


%% load high frequency res
fitDfile = dir(fullfile('~/Documents/work/data/matlab/imagenet/', 'imagenet-keras-resnet18t-b*-0.15*'));

im_converg_time = [];
im_firing_rate = [];
im_val_accuracy = [];
im_tmv = [];
im_mean_acc = [];
im_std_acc = [];

spred = [];
slab = [];
testLfr = [];


er = 0.370; % 62.98% - top5 85%
P1er = er + 1*er/100;

for i = 1:numel(fitDfile)
    savestring = [fitDfile(i).folder '/' fitDfile(i).name];
    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    
    im_converg_time = [im_converg_time, sum(snet.best,2)];
    im_firing_rate = [im_firing_rate, round(snet.firingrate,1)];
    im_val_accuracy = [im_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(im_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(im_converg_time(:,i));
    end
    im_tmv = [im_tmv, tm];
    im_mean_acc = [im_mean_acc, (1-mean(im_converg_time(tm:end,i)))];
    im_std_acc = [im_std_acc, std(im_converg_time(tm:end,i))];
    
    spred = [spred single(sinfo_rl.val.pred(1,:))];
    slab = [slab sinfo_rl.val.lab(1,:)];
    testLfr = [testLfr;single(ssnet.testPreSML)];
    

    
end

%% tot numb of neurons and sum spikes
fitDfile = dir(fullfile(pwd,'/data/imagenet-keras-resnet18/imagenet-keras-valid-resnet18-AROUSAL-SUMSPIKES-b*-0.15*'));
tot_neurons = 0;
sum_spikes = 0;
sumSpikesTime = 0;
for ifo = 1:numel(fitDfile)
        
    savestring = [fitDfile(ifo).folder '/' fitDfile(ifo).name];
    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    tot_neurons = 0;
    
    for i=1:numel(ssnet.layers)
        %     display(ssnet.layers{i}.type)
        if strcmp(ssnet.layers{i}.type,'snn_relu_th')
            sum_spikes = sum_spikes + sum(double(ssnet.layers{i}.tot_spikes(:))) ;
            tot_neurons = tot_neurons + numel(ssnet.layers{i}.tot_spikes) ;
        end
    end
    sumSpikesTime = sumSpikesTime +  sum(ssnet.sumSpikesTime);
    
end

sum_spikes_im = sum_spikes/50000;
avg_firing_rate = sum_spikes_im / tot_neurons;

sum_spikes_im_b = sumSpikesTime/50000;



%%
% spred = sinfo_rl.val.pred(1,:);
% slab = sinfo_rl.val.lab(1,:);
% ser = slab == spred;
% serim = find(ser == 0);
% testHfr = ssnet.testPreSML;






% spred = sinfo_rl.val.pred(1,:);
% slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim_l = find(ser == 0);
display(sum(ser)/50000)

% testLfr = ssnet.testPreSML;% % shiim = find(ssnet.ara < 0.9);
% av = zeros(size(serim));
% for i = 1:numel(serim)
%     
%     a = find(erim==serim(i));    
%     if isempty(a)==0
%         av(i) = a;
%     end
% end
% 
% display(sum(av>0)/numel(serim))
%     
%
falsePos = [];
fasleNeg = [];
corrToWrong = [];

for i = 1:numel(serim_l)
   
    a = find(serim==serim_l(i));
    if isempty(a)
       corrToWrong =  [corrToWrong, serim_l(i)];
    end
    
end

wrongToCorr = [];

for i = 1:numel(serim)
   
    a = find(serim_l==serim(i));
    if isempty(a)
       wrongToCorr =  [wrongToCorr, serim(i)];
    end
    
end
%%
%% Compute threshold-line
ara = zeros(1,50000);
ara2 = zeros(1,50000);
bb = zeros(size(testLfr));
bb(:,1) = testLfr(:,1);
cc = zeros(50000,1000);
dd = zeros(50000,1);

xti = 370:20:500 ;
yti = [];
for i=350:700
   
%       aa = testLfr(:,i);
      bb(:,i) = bb(:,i-1) + testLfr(:,i);
      aa = reshape(bb(:,i),2,50000);
      aa = sort(aa,'descend');
      
      % Method 1 - cumulated divergence 1st vs 2nd
      cc(:,i) = aa(1,:)-aa(2,:);     
      
      
      
      if any(i == xti)          
          eth = max(cc(corrToWrong,i));
          yti = [yti, round(eth,-1)];
%           ara(cc(:,i)<ceil(eth))=1;  
      end     
      
      
end

% figure
plot(xti,yti)
hold on

%%
p = polyfit(xti,yti+20,1);

plot(xti,xti*p(1)+p(2))