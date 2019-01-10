
function res_t = fitDistrbLayers(res_,net_rl)


res_t = struct;

datapos = res_(1).x>0;
nntrainmin = min(res_(1).x(datapos));

for irt = 1:numel(res_)
%     
    if irt == 1
        for in_m = 1:size(res_(irt).x,3)
            for in_l1 = 1:size(res_(irt).x,1)
%                 in_l1
                for in_l2 = 1:size(res_(irt).x,2)
%                     in_l2
                    a= double(squeeze(res_(irt).x(in_l1,in_l2,in_m,:)));
                    a(a==0)=NaN;
                    if sum(isnan(a))~=numel(a)
                        
                        pdca = fitdist(a,'Weibull');
                        
                        res_t(irt).fitWDistA(in_l1,in_l2,in_m) = pdca.A;
                        if isinf(pdca.B)
                            res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                        else
                            res_t(irt).fitWDistB(in_l1,in_l2,in_m) = pdca.B;
                        end
                 
                        
                    else
                        res_t(irt).fitWDistA(in_l1,in_l2,in_m) = nntrainmin;
                        res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                        
                    end
                    
                end
            end
        end
        
        
    elseif strcmp(net_rl.layers{irt-1}.type,'relu')
        
        for in_m = 1:size(res_(irt).x,3)
            
            fprintf('Processing layer %2.0f, map %3.0f.\n ',[irt, in_m] );
            
            for in_l1 = 1:size(res_(irt).x,1)
                for in_l2 = 1:size(res_(irt).x,2)
                    
                    a= double(squeeze(res_(irt).x(in_l1,in_l2,in_m,:)));
                    a(a==0)=NaN;
                    if sum(isnan(a))~=numel(a)
                        
                        pdca = fitdist(a,'Weibull');
                        res_t(irt).fitWDistA(in_l1,in_l2,in_m) = pdca.A;
                        if isinf(pdca.B)
                            res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                        else
                            res_t(irt).fitWDistB(in_l1,in_l2,in_m) = pdca.B;
                        end
                        
                    else
                        res_t(irt).fitWDistA(in_l1,in_l2,in_m) = nntrainmin;
                        res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                    end
                    
                    
                    
                    
                end
            end
        end
    elseif strcmp(net_rl.layers{irt-1}.type,'pool')
        if strcmp(net_rl.layers{irt-1}.method,'max')
        for in_m = 1:size(res_(irt).x,3)
            
            fprintf('Processing layer %2.0f, map %3.0f.\n ',[irt, in_m] );
            
            for in_l1 = 1:size(res_(irt).x,1)
                for in_l2 = 1:size(res_(irt).x,2)
                    
                    a= double(squeeze(res_(irt).x(in_l1,in_l2,in_m,:)));
                    a(a==0)=NaN;
                    if sum(isnan(a))~=numel(a)
                        
                        pdca = fitdist(a,'Weibull');
                        res_t(irt).fitWDistA(in_l1,in_l2,in_m) = pdca.A;
                        if isinf(pdca.B)
                            res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                        else
                            res_t(irt).fitWDistB(in_l1,in_l2,in_m) = pdca.B;
                        end
                        
                    else
                        res_t(irt).fitWDistA(in_l1,in_l2,in_m) = nntrainmin;
                        res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                    end
                    
                    
                    
                    
                end
            end
        end
        end
    elseif strcmp(net_rl.layers{irt-1}.type,'softmax')
        
        for in_m = 1:size(res_(irt).x,3)
            
            fprintf('Processing layer %2.0f, map %3.0f.\n ',[irt, in_m] );
            
            for in_l1 = 1:size(res_(irt).x,1)
                for in_l2 = 1:size(res_(irt).x,2)
                    
                    a= double(squeeze(res_(irt).x(in_l1,in_l2,in_m,:)));
                    a(a<=0)=NaN;
                    if sum(isnan(a))~=numel(a)
                        
                        pdca = fitdist(a,'Weibull');
                        res_t(irt).fitWDistA(in_l1,in_l2,in_m) = pdca.A;
                        if isinf(pdca.B)
                            res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                        else
                            res_t(irt).fitWDistB(in_l1,in_l2,in_m) = pdca.B;
                        end
                        
                    else
                        res_t(irt).fitWDistA(in_l1,in_l2,in_m) = nntrainmin;
                        res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                    end
                    
                    
                    
                    
                end
            end
        end
    end
end



end