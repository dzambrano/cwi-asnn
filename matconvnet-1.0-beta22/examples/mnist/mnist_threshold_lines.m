% the mnist study - threshold lines

%% Experiment with the cnn_mnist_fc_bnorm
clear all
close all

cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn

%% load high frequency res
load('data/mnist-keras-bn/mnist-keras-bnth0.12-99.56_fr_56.mat')
spred = sinfo_rl.val.pred(1,:);
slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim = find(ser == 0);
testHfr = ssnet.testPreSML;
%% load low frequency res

fitDfile = dir(fullfile('data/mnist-keras-bn', 'mnist-keras-bnth*'));
xti = 210:2:300 ;
ytiv = zeros(numel(xti),numel(fitDfile)-1);
ytivnorm = zeros(numel(xti),numel(fitDfile)-1);

thini = [0.5:0.1:1.0 1.2 1.4];

for ithl = 2:numel(fitDfile)
    
    
    savestring = fitDfile(ithl).name; 
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    
    
    spred = sinfo_rl.val.pred(1,:);
    slab = sinfo_rl.val.lab(1,:);
    ser = slab == spred;
    serim_l = find(ser == 0);
    testLfr = ssnet.testPreSML;% % shiim = find(ssnet.ara < 0.9);
    
    
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
    % Compute threshold-line
    ara = zeros(1,10000);
    ara2 = zeros(1,10000);
    bb = zeros(size(testLfr));
    bb(:,1) = testLfr(:,1);
    cc = zeros(10000,500);
    dd = zeros(10000,1);
    
%     xti = 270:20:400 ;
    yti = [];
    sara = [];
    for i=200:500
        
        %       aa = testLfr(:,i);
        bb(:,i) = bb(:,i-1) + testLfr(:,i);
        aa = reshape(bb(:,i),10,10000);
        aa = sort(aa,'descend');
        
        % Method 1 - cumulated divergence 1st vs 2nd
        cc(:,i) = aa(1,:)-aa(2,:);
        
        
        
        if any(i == xti)
            eth = max(cc(corrToWrong,i));
            yti = [yti, round(eth,-1)];
            
            sara = [sara, sum(cc(:,i)<round(eth,-1))];
            %           ara(cc(:,i)<ceil(eth))=1;
        end
        
        
    end
    
    % figure
    plot(xti,yti)
    hold on
    
    
    ytiv(:,ithl-1) = yti;
    
    % Compute spike-size
    h         = single(1);
    theta0    = single(thini(ithl-1));
    m_f       = single(theta0);
    tau_gamma = single(15);
    tau_eta   = single(50);
    
    
    
    c1 = 2*m_f*tau_gamma*tau_gamma;
    c2 = 2*theta0*tau_eta*tau_gamma;
    c3 = tau_gamma*(m_f*tau_gamma + 2*(m_f + 1)*tau_eta);
    c4 = theta0*tau_gamma*tau_eta + theta0*tau_eta*tau_eta;
    
    c0 = h/(exp((c1*0.5*theta0 + c2)/(c3*0.5*theta0 + c4)) - 1) - h/2;
    
    
    y1 = max(0,h./(exp((c1 + c2)./(c3 + c4)) - 1) - c0 );
    spikesize = single(1/y1);
    
    ytivnorm(:,ithl-1) = yti/spikesize;
    
end

%% Plot


figure
subplot(1,2,1), plot(xti,ytiv)
subplot(1,2,2), plot(xti,ytivnorm)
%%
ytivnorm2 = zeros(size(ytivnorm));

pp = zeros(2,size(ytivnorm,2)); 
spikesizev = zeros(1,size(ytiv,2));
for i=1:size(ytiv,2)
    
    
    % Compute spike-size
    h         = single(1);
    theta0    = single(thini(i));
    m_f       = single(theta0);
    tau_gamma = single(15);
    tau_eta   = single(50);
    
    
    c1 = 2*m_f*tau_gamma*tau_gamma;
    c2 = 2*theta0*tau_eta*tau_gamma;
    c3 = tau_gamma*(m_f*tau_gamma + 2*(m_f + 1)*tau_eta);
    c4 = theta0*tau_gamma*tau_eta + theta0*tau_eta*tau_eta;
    
    c0 = h/(exp((c1*0.5*theta0 + c2)/(c3*0.5*theta0 + c4)) - 1) - h/2;
    
    
    y1 = max(0,h./(exp((c1 + c2)./(c3 + c4)) - 1) - c0 );
    spikesize = single(1/y1);
    
    spikesizev(i) = spikesize;
    
    ytivnorm2(:,i) = ytiv(:,i)/(2*spikesize);
    
    pp(:,i) = polyfit(xti,ytivnorm2(:,i)',1);
end
%%
close all
figure
subplot(1,2,1), plot(xti,ytiv)
subplot(1,2,2), plot(xti,ytivnorm2)

[~,a] = max(ytivnorm2(21,:));

% ytivpp = pp(1,:)'*xti + repmat(pp(2,:)',1,20);
% 
% mpp = mean(pp,2);
% stdp = std(pp,[],2);
% ytivppm = (mpp(1))*xti + (mpp(2));

% figure
% subplot(1,2,1), plot(xti,ytiv)
% subplot(1,2,2), plot(xti,ytivpp)
% hold on
% plot(xti,ytivppm,'LineWidth',2)


ythlineFinal = zeros(size(ytivnorm2));

ytinm = mean(ytivnorm2,2);
ytins = std(ytivnorm2,[],2);

ppm = polyfit(xti',ytinm+2*ytins(6),1);

ppmax = polyfit(xti',ytivnorm2(:,a)+10,1);

for i = 1:size(ytiv,2)
   
%     ythlineFinal(:,i) = (2*spikesizev(i))*(ppm(1)*xti + ppm(2));
    ythlineFinal(:,i) = (2*spikesizev(i))*(ppmax(1)*xti + ppmax(2));
    
end



figure
subplot(1,2,1), plot(xti,ytiv)
hold on
ax = gca;
ax.ColorOrderIndex = 1;
plot(xti,ythlineFinal,'LineWidth',2)

subplot(1,2,2), plot(xti,ytivnorm2)
hold on
plot(xti,ppmax(1)*xti + ppmax(2),'LineWidth',2)




