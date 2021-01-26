clear all;clc;
warning('off');
addpath('tensor_toolbox-v3.1');
%% gendata

addpath(genpath('./data'));
K = 8;
file = 'cms1k_dense';
fileName = strcat(file, '.csv');
isBinary = 1;
[X, Xs, dim, nonzero_ratio, cutoffs] = genTensor(K, fileName,isBinary);

rank = 10;

%% initialize
nd = ndims(X);
sz = size(X);
dim = size(X);

%% initilize each client

% EF = cell(nd,1);
for k = 1:K
    client(k).X = Xs{k};
    sz = size(Xs{k});
    tsz = prod(sz);
    nmissing = 0;
    nnonzeros = nnz(Xs{k});
    nzeros = tsz - nnonzeros;    
    % Save info
    client(k).info.size = sz;
    client(k).info.tsz = tsz;
    client(k).info.nmissing = nmissing;
    client(k).info.nnonzeros = nnonzeros;
    client(k).info.nzeros = nzeros;
    client(k).EF{1} = zeros(sz(1),rank);
    for n = 2:nd
        client(k).EF{n} = zeros(sz(n),rank);
    end
end

Uinit = cell(1,nd); % Initial Global factor matrix
for n = 1:nd % n=1 initialize at institutions
    Uinit{n} = rand(dim(n),rank);
end
M0 = ktensor(Uinit);
M0 = M0 * (norm(X)/norm(M0)); % normalize
M0 = normalize(M0,0);
% U = M0;
U = cell(1,nd);
for d=1:nd
    U{d} = M0{d};
end
% U = double(U);

for k=1:K
    client(k).dim=size(client(k).X); %data
    client(k).U=cell(1,nd); % initialize 3 factor matrices
    % n = 1
    client(k).U{1}=zeros(dim(1),rank);
    client(k).U{1}(cutoffs{k},:) = U{1}(cutoffs{k},:);
    for d=2:nd
        client(k).U{d}=U{d};
    end
end

oversample = 1.1;
for k=1:K
    xnzidx = tt_sub2ind64(size(client(k).X),client(k).X.subs);
    xnzidx = sort(xnzidx);
    tsz = prod(size(client(k).X));
    nnonzeros = nnz(client(k).X);
    nzeros = tsz - nnonzeros; 
    ftmp = max(ceil(nnz(client(k).X)/100), 10^5);
    fsamp(1) = min(ftmp, nnonzeros);
    fsamp(2) = min([ftmp, nnonzeros, nzeros]);
    [fsubs, fvals, fwgts] = tt_sample_stratified(client(k).X, xnzidx, fsamp(1), fsamp(2), oversample);
    client(k).fsubs = fsubs;
    client(k).fvals = fvals;
    client(k).fwgts = fwgts;
end

%aggregate the sampled entries
subs = [];
vals = [];
wgts = [];
for k=1:K
   subs = vertcat(subs, client(k).fsubs);
   vals = vertcat(vals, client(k).fvals);
   wgts = vertcat(wgts, client(k).fwgts);
end


%%%%%%%%%%%%%%%%%%%% Run Logit %%%%%%%%%%%%%%%%%%%%
%% initialize
rank = 10;
params.maxepoch = 30;
params.epciters = 50;
params.nsamplsq = 5;
params.maxfails = 2;
nd = ndims(X);
gsamp_rate = 10;
params.decay = 0.5;
v=1;
prox = 1e-3;
isLogit = 1;

params.tau = 10;


for lr = [20]
    params.gamma = lr;

    %%%%%%%%% Distributed Sign EF local SGD %%%%%%%%%%%%%
    isCyclic = 0;
    fileName = sprintf('logitloss/%s_K=%d_rank=%d_lr=%0.0e_gsamp=%0.0e_distEF_local_ver=%d', file, K, rank,lr, gsamp_rate, v);
    fileID = fopen(strcat(fileName,'.txt'), 'w');
    P1 = gcp_dist_sign_ef_local(fileID,client,nd,K,isLogit,isCyclic,params);

        %%%%%%%%% Distributed Sign EF SGD %%%%%%%%%%%%%
    isCyclic = 0;
    fileName = sprintf('logitloss/%s_K=%d_rank=%d_lr=%0.0e_gsamp=%0.0e_distEF_every_ver=%d', file, K, rank,lr, gsamp_rate, v);
    fileID = fopen(strcat(fileName,'.txt'), 'w');
    P2 = gcp_dist_sign_ef(fileID,client,nd,K,isLogit,isCyclic,params);
    

        %%%%%%%%% Distributed Sign EF SGD (proximal) %%%%%%%%%%%%%
    isCyclic = 0;
    fileName = sprintf('logitloss/%s_K=%d_rank=%d_lr=%0.0e_gsamp=%0.0e_distEFprox_every_ver=%d', file, K, rank,lr, gsamp_rate, v);
    fileID = fopen(strcat(fileName,'.txt'), 'w');
    P3 = gcp_dist_sign_ef_prox(fileID,client,nd,K,prox,isLogit,isCyclic,params);
end
% exit;