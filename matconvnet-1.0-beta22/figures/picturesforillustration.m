%% Experiment with the cnn_mnist_fc_bnorm
clear all
close all

cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn

%% load database
load('data/mnist-keras-bn/imdb.mat')

%% load high frequency res
load(fullfile(asnn_root, 'data', 'mnist', 'mnist-keras-bnth0.12-99.56_fr_56.mat'))
spred = sinfo_rl.val.pred(1,:);
slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim = find(ser == 0);
testHfr = ssnet.testPreSML;

%% load low frequency res
load('data/mnist-keras-bn/mnist-keras-bnth0.6-99.34_fr_12.mat') % 0.6

spred = sinfo_rl.val.pred(1,:);
slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim_l = find(ser == 0);
testLfr = ssnet.testPreSML;

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
close all

nc = 4;
easy = images.data(:,:,:,60000+nc);
easyi = easy>0.3;
easyn = easyi.*easy + easyi.*(0.6*randn(size(easy)));

nctw = 15;
hard = images.data(:,:,:,60000+corrToWrong(nctw)); 
hardi = hard>0.3;
hardn = hardi.*hard + hardi.*(0.6.*randn(size(easy)));

fprintf('Predicted %d \n', spred(corrToWrong(nctw))-1);
fprintf('Correct %d \n', slab(corrToWrong(nctw))-1);

figure
subplot(2,2,1)
imshow(easy)


subplot(2,2,2)
imshow(easyn)
subplot(2,2,3)
imshow(hard)
subplot(2,2,4)
imshow(hardn)


%% plot threshold-line method
% ara = zeros(1,10000);
% ara2 = zeros(1,10000);
% bb = zeros(size(testLfr));
% bb(:,1) = testLfr(:,1);
% cc = zeros(10000,500);
% dd = zeros(10000,1);
% 
% xti = 300 ;
% yti = [];
% sara = [];
% 
% 
% for i=250:500
%    
% %       aa = testLfr(:,i);
%       bb(:,i) = bb(:,i-1) + testLfr(:,i);
%       aa = reshape(bb(:,i),10,10000);
%       aa = sort(aa,'descend');
%       
%       % Method 1 - cumulated divergence 1st vs 2nd
%       cc(:,i) = aa(1,:)-aa(2,:);     
%       
%       
%       
%       if any(i == xti)          
%           eth = max(cc(corrToWrong,i));
%           yti = [yti, round(eth,-1)];
%           
%           sara = [sara, sum(cc(:,i)<round(eth,-1))];
% %           ara(cc(:,i)<ceil(eth))=1;  
%       end     
%       
%       
% end
%%
% close all
% 
% figure
% subplot(2,1,1)
% plot(bb(nc*10-9:nc*10,:)','LineWidth'   , 2)
% ax = gca;
% set(ax, ...
%     'XLim'        , [250 400],...
%   'Box'         , 'off'     , ... 
%   'YGrid'       , 'off'      , ...
%   'XGrid'       , 'off'      , ...
%   'YMinorTick'  , 'on'      , ...
%   'LineWidth'   , 2         );
% set(ax,'FontSize',28,'FontWeight','Normal');
% 
% subplot(2,1,2)
% plot(bb(corrToWrong(nctw)*10-9:corrToWrong(nctw)*10,:)','LineWidth'   , 2)
% ax = gca;
% set(ax, ...
%     'XLim'        , [250 400],...
%   'Box'         , 'off'     , ... 
%   'YGrid'       , 'off'      , ...
%   'XGrid'       , 'off'      , ...
%   'YMinorTick'  , 'on'      , ...
%   'LineWidth'   , 2         );
% set(ax,'FontSize',28,'FontWeight','Normal');
% hXLabel = xlabel('Time (ms)','FontName','Calibri','FontSize',28,'FontWeight','Normal');

%%

close all

figure
subplot(2,2,1)
plot(testHfr(nc*10-9:nc*10,:)','LineWidth'   , 2)
ax = gca;
set(ax, ...
    'XLim'        , [250 400],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
title('High precision','FontWeight','Normal')
set(ax,'FontSize',28,'FontWeight','Normal');

subplot(2,2,2)
plot(testLfr(nc*10-9:nc*10,:)','LineWidth'   , 2)
ax = gca;
set(ax, ...
    'XLim'        , [250 400],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
title('Low precision','FontWeight','Normal')
set(ax,'FontSize',28,'FontWeight','Normal');

subplot(2,2,3)
plot(testHfr(corrToWrong(nctw)*10-9:corrToWrong(nctw)*10,:)','LineWidth'   , 2)
ax = gca;
set(ax, ...
    'XLim'        , [250 400],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );

set(ax,'FontSize',28,'FontWeight','Normal');
hXLabel = xlabel('Time (ms)','FontName','Calibri','FontSize',28,'FontWeight','Normal');

subplot(2,2,4)
plot(testLfr(corrToWrong(nctw)*10-9:corrToWrong(nctw)*10,:)','LineWidth'   , 2)
ax = gca;
set(ax, ...
    'XLim'        , [250 400],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
set(ax,'FontSize',28,'FontWeight','Normal');
hXLabel = xlabel('Time (ms)','FontName','Calibri','FontSize',28,'FontWeight','Normal');
%%
close all
figure
subplot(2,4,1)
imshow(easy)
subplot(2,4,2)
plot(testHfr(nc*10-9:nc*10,:)','LineWidth'   , 2)
ax = gca;
set(ax, ...
    'XLim'        , [0 500],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
title('High precision','FontWeight','Normal')
set(ax,'FontSize',28,'FontWeight','Normal');

subplot(2,4,3)
imshow(easyn)

subplot(2,4,4)
plot(testLfr(nc*10-9:nc*10,:)','LineWidth'   , 2)
ax = gca;
set(ax, ...
    'XLim'        , [0 500],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
title('Low precision','FontWeight','Normal')
set(ax,'FontSize',28,'FontWeight','Normal');


subplot(2,4,5)
imshow(hard)
subplot(2,4,6)
plot(testHfr(corrToWrong(nctw)*10-9:corrToWrong(nctw)*10,:)','LineWidth'   , 2)
ax = gca;
set(ax, ...
    'XLim'        , [0 500],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );

set(ax,'FontSize',28,'FontWeight','Normal');
hXLabel = xlabel('Time (ms)','FontName','Calibri','FontSize',28,'FontWeight','Normal');


subplot(2,4,7)
imshow(hardn)

subplot(2,4,8)

% load arousal data
load('mnist-keras-ARAUSAL-picture-0.6-0.12-50ms-99.56_fr_13.mat')

% plot(testLfr(corrToWrong(nctw)*10-9:corrToWrong(nctw)*10,:)','LineWidth'   , 2)
plot(ssnet.testPreSML(corrToWrong(nctw)*10-9:corrToWrong(nctw)*10,:)','LineWidth'   , 2)
hold on


plot(250*ones(size([-10:10])),[-10:10],'--','LineWidth'   , 2)

ax = gca;
set(ax, ...
    'XLim'        , [0 500],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
set(ax,'FontSize',28,'FontWeight','Normal');
hXLabel = xlabel('Time (ms)','FontName','Calibri','FontSize',28,'FontWeight','Normal');
