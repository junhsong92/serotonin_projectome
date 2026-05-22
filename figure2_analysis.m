clear all; close all;
rng('default');
addpath(genpath(pwd))
load YOUR_DATA
% YOUR_DATA contains your data in a 2D matrix ("your_data", n samples x m features; 
% in our case, it's 110 brains x L2 normalized axon density in 280 brain regions)

k_max=15; max_iter=1000;

parfor k_ii=1:k_max  % We plan to test NMF using 1 to 15 basis patterns (factors).
    for iter_ii=1:max_iter  % run NMF with max_iter random sampling
       
        training_binary=ones(1,numel(your_data));
        training_binary(1:round(numel(your_data)*0.1))=0;   %%% 10% elements = 0
        rand_binary_idx=randperm(length(training_binary));
        training_binary=training_binary(rand_binary_idx);
        training_binary=reshape(training_binary,size(your_data));
        
     
        training_data_pop=your_data.*training_binary;
        test_data_pop=your_data.*(1-training_binary);
        
        [W,H,D] = nnmf_crossval(training_data_pop,k_ii,training_binary);

        D_arr_training(k_ii,iter_ii)=D; %% reconstruction error - trainint
        
        test_error_mat=(1-training_binary).*(test_data_pop-W*H);
        test_error_mat = norm(test_error_mat,'fro')/sqrt(sum(sum((1-training_binary))));
        D_arr_test(k_ii,iter_ii)=test_error_mat; %% reconstruction error - test
      
    end
end


%%%%% Cross Validation Plot %%%%%
figure; hold on
h=shadedErrorBar(1:size(D_arr_test,1),mean(D_arr_test,2),...
    std(D_arr_test,0,2),...
    'lineprops',{'-o','MarkerSize',3, 'MarkerFaceColor', [1 0 0],'Color',[1 0 0]});
    %%% source: https://www.mathworks.com/matlabcentral/fileexchange/26311-raacampbell-shadederrorbar
h=shadedErrorBar(1:size(D_arr_training,1),mean(D_arr_training,2),...
    std(D_arr_training,0,2),...
  'lineprops',{'-o','MarkerSize',3, 'MarkerFaceColor', [0 0 1],'Color',[0 0 1]});

ylim([0 inf]);
ylabel('RMSE')
xlabel('Number of Patterns k')
set(gca,'XTick',1:k_max)

[~,~,stats] = anova1(D_arr_test',[],'off');
c = multcompare(stats,'CType','bonferroni');


num_nnmf_k=optimalK %% you can change "optimalK" to the optimal k found using ANOVA (in our case it was 5)



%%  Pattern Plotting
clearvars W_arr H_arr

rng('default');
parfor trial_ii=1:5000
    [W_arr(:,:,trial_ii),H_arr(:,:,trial_ii),D(trial_ii)] = nnmf(your_data,num_nnmf_k);
end

D_best=find(D==min(D));
W=W_arr(:,:,D_best(1));
H=H_arr(:,:,D_best(1));

%%% after this, we are left with the basis pattern matrix (H), k x m,
%%% and dimensionality-reduced dataset W, n x k
%%% Users may design their downstream analyses using these matrices


%%% For example, hierarchical clustering
eva = evalclusters(W,'linkage','gap','KList',1:10, 'SearchMethod','firstMaxSE') % for choosing the optimal number of clusters N
%plot(eva)
tree = linkage(W,'average'); %% you can try using different methods instead of 'average', e.g. 'ward', 'weighted', or 'complete', depending on your purpose
figure; [~,~,op_nmf]=dendrogram(tree, size(your_data,1));
nmf_weight=W(op_nmf,:);
figure; heatmap(nmf_weight','colormap',hot(1000),'gridvisible','off','ColorLimits',[0 1])
