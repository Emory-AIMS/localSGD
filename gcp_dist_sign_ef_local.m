function [K] =  gcp_dist_sign_ef_local(fileID,client,nd,K,isLogit,isCyclic,params)
%% logit loss; sample fibar

	maxepoch = params.maxepoch;
	gamma = params.gamma;
	nsamplsq = params.nsamplsq;
    epciters = params.epciters;
    maxfails = params.maxfails;
    tau = params.tau;
    
    if isLogit
        fh = @(x,m) log(exp(m) + 1) - x .* m;
        gh = @(x,m) exp(m)./(exp(m) + 1) - x;
    else
        fh = @(x,m) (m-x).^2;
        gh = @(x,m) 2.*(m-x);
    end

    time = 0;
    fest = 0;
    for k=1:K
        T = ktensor(client(k).U);
        fest = fest + gcp_fvals_est(T, fh, client(k).fsubs, client(k).fvals, client(k).fwgts, false);
    end
    disp(['epoch = ' num2str(0) ' ,loss = ' num2str(fest)]);
    fprintf(fileID, 'Epoch, f-est, time\n');
    fprintf(fileID, '%2d, %e, %e\n', 0, fest, time/K);
    
    % Set up loop variables
    nfails = 0;                     % Counter # times fails to decrease
    fest_prev = fest;               % Corresponding function value
    for k=1:K
        client(k).Usaved = client(k).U;
    end

    
    if isCyclic
        nd_list = repmat(1:nd,1,ceil(maxepoch*epciters/nd));
        direction_array = nd_list(1:maxepoch*epciters);
    else
        direction_array = randi([1 nd],1,maxepoch*epciters);
    end
        
	%% begin main iter
	for i = 1:maxepoch
        start = cputime;
        for iter = 1:epciters


            rd_n = direction_array((maxepoch-1)*epciters+iter);
            if (mod(iter,tau) ~= 0)
            %% perform local update
                for k = 1:K
                    client(k).subs = sample_mode_n(nsamplsq, client(k).info.size, rd_n);
                    client(k).G_rd_n = gcp_fg_est_G_rd(client(k).U,fh,gh,client(k).subs,client(k).X(client(k).subs),nd,client(k).info.size,rd_n);
                    client(k).U{rd_n} = client(k).U{rd_n} - gamma*client(k).G_rd_n;
                end		
            else
                %% communicate
                if (rd_n == 1)
                    rd_n = randi([2 nd]);
                end
                for k = 1:K
                    client(k).subs = sample_mode_n(nsamplsq, client(k).info.size, rd_n);
                    client(k).G_rd_n = gcp_fg_est_G_rd(client(k).U,fh,gh,client(k).subs,client(k).X(client(k).subs),nd,client(k).info.size,rd_n);
                    client(k).U{rd_n} = client(k).U{rd_n} - gamma*client(k).G_rd_n;
                    client(k).P_rd_n = Ug{rd_n} - client(k).U{rd_n} + client(k).EF{rd_n};
                    client(k).Delta_rd_n = (norm(client(k).P_rd_n(:),1)/numel(client(k).P_rd_n))*sign(client(k).P_rd_n);
                    client(k).EF{rd_n} = client(k).P_rd_n - client(k).Delta_rd_n;
                end
                sumDelta = zeros(size(client(1).Delta_rd_n));
                for k = 1:K
                    sumDelta = sumDelta + client(k).Delta_rd_n;
                end
                Ug{rd_n} = Ug{rd_n} - sumDelta/K;

                for k = 1:K
                    client(k).U{rd_n} = Ug{rd_n};
                end
            end
        end
		time = cputime-start;
        fest = 0;
        for k=1:K
            T = ktensor(client(k).U);
            fest = fest + gcp_fvals_est(T, fh, client(k).fsubs, client(k).fvals, client(k).fwgts, false);
        end 
%         if (nfails > maxfails) || isnan(fest)
%             break;
%         end
        if fest > fest_prev || isnan(fest)
%             nfails = nfails + 1;
%             gamma = 0.1 * gamma;
%             for k=1:K
%                 client(k).U = client(k).Usaved;
%             end
            fest = fest_prev;
            fprintf(fileID, '%2d, %e, %e\n', i, fest, time/K);
            break;
        else
%             for k=1:K
%                 client(k).Usaved = client(k).U;
%             end
            fest_prev = fest;
            fprintf(fileID, '%2d, %e, %e\n', i, fest, time/K);
        end
        
    end %%end of main iter

end

function [subs] = sample_mode_n(nsamplsq, dims, n)
D = length(dims);
tensor_idx = zeros(nsamplsq, D);     % Tuples that index fibers in original tensor

tensor_idx(:,n) = ones(nsamplsq, 1);
for i = [1:n-1,n+1:D]
    % Uniformly sample w.r. in each dimension besides n
    tensor_idx(:,i) = randi(dims(i), nsamplsq, 1);
end
tensor_idx = kron(tensor_idx,ones(dims(n),1)); % portable
tensor_idx(:,n) = repmat((1:dims(n))',nsamplsq,1);
subs = tensor_idx;
end
