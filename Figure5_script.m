rng('default')
addpath(genpath(pwd))
rand_seed_final=11; %umap random seed; can be any value

run_ren_cv=true;
ren_matrix_file='sert.matrix.csv';
ren_idents_file='idents.csv';

%%% load data
endog_Lowcutoff=20;
min_count=3;
bc_cnt_start=2;

T=readtable('starmap_data.csv');
data_mat=table2array(T);

bc_cols=1:5;
tph2_col=11;
sert_col=17;
ap_col=18; ml_col=19; dv_col=20;
region_col=21;
mouse_col=22;

marker_names_ordered={'Syt10','Sox14','Gad1','Syt2','Npas1','Tacr3','Ppp1r17','Irx2','Met','Piezo2'};
marker_cols_ordered=[13 8 7 12 6 14 15 9 10 16];

sero_idx=data_mat(:,tph2_col)>=endog_Lowcutoff;
sero_table=data_mat(sero_idx,:);
X_raw=sero_table(:,marker_cols_ordered);
raw_total=sum(X_raw,2);
keep_cells=raw_total>=min_count;
sero_table_filt=sero_table(keep_cells,:);
X_filt=X_raw(keep_cells,:);
n_cells=size(X_filt,1);

X_log=log1p(X_filt);
X_z=zscore(X_log,0,1);

mouse_ids_per_cell=sero_table_filt(:,mouse_col);
mouse_ids=unique(mouse_ids_per_cell);
n_mice=length(mouse_ids);

combo_id_per_mouse=zeros(n_mice,1);
combo_id_per_cell=zeros(n_cells,1);
for ii=1:n_mice
    mid=mouse_ids(ii);
    combo=mod(ii-1,7)+1;
    combo_id_per_mouse(ii)=combo;
    combo_id_per_cell(mouse_ids_per_cell==mid)=combo;
end

sex_page_per_mouse=ones(n_mice,1);
sex_page_per_mouse(8:14)=2;

%%% harmony pca leiden
conda_env='scrna_env';
pca_var_thresh=90;
leiden_resolution=0.50;
leiden_n_neighbors=15;

writematrix(X_z,'temp_h.csv');
writematrix(mouse_ids_per_cell,'temp_h_meta.csv');
fid=fopen('run_h.py','w');
fprintf(fid,'import os; os.environ["NUMBA_NUM_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"\n');
fprintf(fid,'import numpy as np; np.random.seed(0)\nimport pandas as pd\nimport harmonypy as hm\n');  % harmonypy=0.2.0
fprintf(fid,'data=pd.read_csv("temp_h.csv",header=None)\nmeta=pd.read_csv("temp_h_meta.csv",header=None,names=["mouse_id"],dtype=str)\n');
fprintf(fid,'ho=hm.run_harmony(data.T,meta,"mouse_id",max_iter_harmony=50,random_state=0)\n');
fprintf(fid,'pd.DataFrame(ho.Z_corr).T.to_csv("temp_h_out.csv",index=False,header=False)\n');
fclose(fid);
system(sprintf('conda run -n %s python run_h.py',conda_env));
X_corr=readmatrix('temp_h_out.csv');
if size(X_corr,1)~=n_cells; X_corr=X_corr'; end
delete('temp_h.csv','temp_h_meta.csv','temp_h_out.csv','run_h.py');

[~,score,~,~,explained,~]=pca(X_corr);
num_pcs=find(cumsum(explained)>pca_var_thresh,1);
if isempty(num_pcs); num_pcs=size(score,2); end
X_pca=score(:,1:num_pcs);

writematrix(X_pca,'temp_l.csv');
fid=fopen('run_l.py','w');
fprintf(fid,'import os; os.environ["NUMBA_NUM_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"\n');
fprintf(fid,'import numpy as np; np.random.seed(0)\nimport pandas as pd\nimport scanpy as sc\nimport anndata as ad\n');  % scanpy=1.10.3, anndata=0.10.9, leidenalg, igraph=0.10.4, numba=0.59.1
fprintf(fid,'data=pd.read_csv("temp_l.csv",header=None)\nadata=ad.AnnData(data.values)\n');
fprintf(fid,'sc.pp.neighbors(adata,n_neighbors=%d,metric="correlation",random_state=0)\n',leiden_n_neighbors);
fprintf(fid,'sc.tl.leiden(adata,resolution=%.2f,flavor="igraph",n_iterations=2,directed=False,random_state=0)\n',leiden_resolution);
fprintf(fid,'pd.DataFrame(adata.obs["leiden"].values.astype(int)+1).to_csv("temp_l_out.csv",index=False,header=False)\n');
fclose(fid);
system(sprintf('conda run -n %s python run_l.py',conda_env));
cluster_id=readmatrix('temp_l_out.csv');
delete('temp_l.csv','temp_l_out.csv','run_l.py');
k=max(cluster_id);

%%$ ren 2019 ref mapping
knn_k=5;
consensus_thresh=3;
n_cv_folds=5;

ren_data=readtable(ren_matrix_file);
jing_idents=readtable(ren_idents_file);
gene_label=marker_names_ordered;
gene_list=table2cell(ren_data(:,1));
data_columnNames=ren_data.Properties.VariableNames(2:end);
dictionary=table2cell(jing_idents(:,1));
original_label_arr=zeros(length(data_columnNames),1);
for ii=1:length(data_columnNames)
    isMatch=find(strcmp(dictionary,data_columnNames{ii}));
    if ~isempty(isMatch)
        dummy_input=table2cell(jing_idents(isMatch,2));
        original_label_arr(ii)=str2double(regexp(dummy_input{1},'\d+','match'));
    end
end
exc_idx=find(original_label_arr==0);
ren_data(:,exc_idx)=[];
original_label_arr(exc_idx)=[];
[~,gene_idx]=ismember(gene_label,gene_list);
ren_essential=table2array(ren_data(gene_idx,2:end))';
valid=~any(isnan(ren_essential),2)&sum(ren_essential,2)>0;
ren_essential=ren_essential(valid,:);
original_label_arr=original_label_arr(valid);
keep_ren=original_label_arr~=4;
ren_essential=ren_essential(keep_ren,:);
original_label_arr=original_label_arr(keep_ren);

keep_clusters_fine=[1 2 3 5 6 7 8 9 10 11];
ren_names_fine={'Syt10/Sox14/Gad1','Sox14/Gad1','Gad1','Syt2','Syt2/Npas1','Tacr3/Met','Ppp1r17','Irx2','Irx2/Tacr3','Met/Piezo2'};
n_subtypes=length(keep_clusters_fine);

n_mega=6;
mega_names={'Gad1 group','Syt2/Npas1 group','Tacr3 group','Ppp1r17','Irx2','Met/Piezo2'};
fine_to_mega_arr=zeros(11,1);
fine_to_mega_arr([1 2 3])=1;
fine_to_mega_arr([5 6])=2;
fine_to_mega_arr([7 10])=3;
fine_to_mega_arr(8)=4;
fine_to_mega_arr(9)=5;
fine_to_mega_arr(11)=6;
mega_to_fine={[1,2,3],[5,6],[7,10],[8],[9],[11]};

ren_log=log1p(ren_essential);
ren_z=zscore(ren_log,0,1);
ren_mega_labels=zeros(size(original_label_arr));
for ii=1:length(original_label_arr)
    ren_mega_labels(ii)=fine_to_mega_arr(original_label_arr(ii));
end
ren_centroids_fine=zeros(n_subtypes,length(gene_label));
for ii=1:n_subtypes
    ren_centroids_fine(ii,:)=mean(ren_z(original_label_arr==keep_clusters_fine(ii),:),1);
end
ren_centroids_mega=zeros(n_mega,length(gene_label));
for ii=1:n_mega
    ren_centroids_mega(ii,:)=mean(ren_z(ren_mega_labels==ii,:),1);
end
knn_mdl_fine=fitcknn(ren_z,original_label_arr,'NumNeighbors',knn_k,'Distance','correlation');
knn_mdl_mega=fitcknn(ren_z,ren_mega_labels,'NumNeighbors',knn_k,'Distance','correlation');

if run_ren_cv
    n_genes=length(gene_label);
    cv_partition=cvpartition(original_label_arr,'KFold',n_cv_folds);
    all_true=[]; all_pred_single=[]; all_pred_twostage=[]; all_confident=[];
    for fold=1:n_cv_folds
        test_idx=find(test(cv_partition,fold));
        train_idx=find(training(cv_partition,fold));
        X_train=ren_z(train_idx,:); y_train=original_label_arr(train_idx);
        X_test=ren_z(test_idx,:); y_test=original_label_arr(test_idx);
        n_test=length(test_idx);
        y_train_mega=zeros(size(y_train));
        for ii=1:length(y_train); y_train_mega(ii)=fine_to_mega_arr(y_train(ii)); end

        knn_single=fitcknn(X_train,y_train,'NumNeighbors',knn_k,'Distance','correlation');
        pred_single=predict(knn_single,X_test);

        centroids_mega_cv=zeros(n_mega,n_genes);
        for ii=1:n_mega
            idx_m=y_train_mega==ii;
            if sum(idx_m)>0; centroids_mega_cv(ii,:)=mean(X_train(idx_m,:),1); end
        end
        knn_mega_cv=fitcknn(X_train,y_train_mega,'NumNeighbors',knn_k,'Distance','correlation');
        mega_knn_cv=predict(knn_mega_cv,X_test);
        dist_mega_cv=zeros(n_test,n_mega);
        for ii=1:n_mega
            dist_mega_cv(:,ii)=1-(X_test*centroids_mega_cv(ii,:)')./(sqrt(sum(X_test.^2,2)).*sqrt(sum(centroids_mega_cv(ii,:).^2)));
        end
        [~,mega_centroid_cv]=min(dist_mega_cv,[],2);

        [coeff_cv,score_train_cv,~,~,expl_cv]=pca(X_train);
        npc_cv=find(cumsum(expl_cv)>pca_var_thresh,1);
        if isempty(npc_cv); npc_cv=size(score_train_cv,2); end
        X_train_pca=score_train_cv(:,1:npc_cv);
        X_test_pca=(X_test-mean(X_train))*coeff_cv(:,1:npc_cv);
        writematrix(X_train_pca,'temp_ren_l.csv');
        fid=fopen('run_ren_l.py','w');
        fprintf(fid,'import os; os.environ["NUMBA_NUM_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"\n');
        fprintf(fid,'import numpy as np; np.random.seed(0)\nimport pandas as pd\nimport scanpy as sc\nimport anndata as ad\n');
        fprintf(fid,'data=pd.read_csv("temp_ren_l.csv",header=None)\nadata=ad.AnnData(data.values)\n');
        fprintf(fid,'sc.pp.neighbors(adata,n_neighbors=%d,metric="correlation",random_state=0)\n',leiden_n_neighbors);
        fprintf(fid,'sc.tl.leiden(adata,resolution=%.2f,flavor="igraph",n_iterations=2,directed=False,random_state=0)\n',leiden_resolution);
        fprintf(fid,'pd.DataFrame(adata.obs["leiden"].values.astype(int)+1).to_csv("temp_ren_l_out.csv",index=False,header=False)\n');
        fclose(fid);
        status_l=system(sprintf('conda run -n %s python run_ren_l.py',conda_env));
        if status_l==0
            clust_train=readmatrix('temp_ren_l_out.csv');
            delete('temp_ren_l.csv','temp_ren_l_out.csv','run_ren_l.py');
            k_cv=max(clust_train);
            cluster_to_mega_cv=zeros(k_cv,1);
            for ii=1:k_cv
                c_profile=mean(X_train(clust_train==ii,:),1);
                d=pdist2(c_profile,centroids_mega_cv,'correlation');
                [~,best]=min(d);
                cluster_to_mega_cv(ii)=best;
            end
            mega_leiden_cv=zeros(n_test,1);
            for ii=1:n_test
                d=pdist2(X_test_pca(ii,:),X_train_pca,'euclidean');
                [~,nn_idx]=mink(d,knn_k);
                mega_leiden_cv(ii)=cluster_to_mega_cv(mode(clust_train(nn_idx)));
            end
        else
            mega_leiden_cv=mega_centroid_cv;
            try; delete('temp_ren_l.csv','temp_ren_l_out.csv','run_ren_l.py'); catch; end
        end

        X_test_qn=zeros(size(X_test));
        for ii=1:n_genes
            [~,to]=sort(X_test(:,ii));
            ts=sort(X_train(:,ii));
            tq=interp1(linspace(0,1,length(ts)),ts,linspace(0,1,n_test),'linear');
            X_test_qn(to,ii)=tq';
        end
        mega_qn_cv=predict(knn_mega_cv,X_test_qn);

        mega_preds_cv={mega_knn_cv,mega_centroid_cv,mega_leiden_cv,mega_qn_cv};
        mega_label_cv=zeros(n_test,1); mega_conf_cv=false(n_test,1);
        for ii=1:n_test
            labels=[mega_preds_cv{1}(ii),mega_preds_cv{2}(ii),mega_preds_cv{3}(ii),mega_preds_cv{4}(ii)];
            [dummy_mode,freq_count]=mode(labels);
            mega_label_cv(ii)=dummy_mode;
            mega_conf_cv(ii)=freq_count>=consensus_thresh;
        end

        fine_label_cv=zeros(n_test,1); fine_conf_cv=false(n_test,1);
        for mi=1:n_mega
            fine_types=mega_to_fine{mi};
            mega_test_idx=find(mega_label_cv==mi&mega_conf_cv);
            if isempty(mega_test_idx); continue; end
            if length(fine_types)==1
                fine_label_cv(mega_test_idx)=fine_types(1);
                fine_conf_cv(mega_test_idx)=true;
                continue;
            end
            train_in_mega=ismember(y_train,fine_types);
            if sum(train_in_mega)<5; continue; end
            X_train_sub=X_train(train_in_mega,:); y_train_sub=y_train(train_in_mega);
            X_test_sub=X_test(mega_test_idx,:); n_sub=length(mega_test_idx);
            fine_centroids_cv=zeros(length(fine_types),n_genes);
            for fi=1:length(fine_types)
                idx_fi=y_train_sub==fine_types(fi);
                if sum(idx_fi)>0; fine_centroids_cv(fi,:)=mean(X_train_sub(idx_fi,:),1); end
            end
            k_nn=min(knn_k,sum(train_in_mega)-1);
            if k_nn<1; k_nn=1; end
            knn_fine_cv=fitcknn(X_train_sub,y_train_sub,'NumNeighbors',k_nn,'Distance','correlation');
            fine_knn_cv=predict(knn_fine_cv,X_test_sub);
            dist_fine_cv=zeros(n_sub,length(fine_types));
            for fi=1:length(fine_types)
                dist_fine_cv(:,fi)=1-(X_test_sub*fine_centroids_cv(fi,:)')./(sqrt(sum(X_test_sub.^2,2)).*sqrt(sum(fine_centroids_cv(fi,:).^2)));
            end
            [~,nearest_fine_cv]=min(dist_fine_cv,[],2);
            fine_centroid_cv=fine_types(nearest_fine_cv)';
            if sum(train_in_mega)>20
                [coeff_sub,score_sub]=pca(X_train_sub);
                npc_sub=find(cumsum(var(score_sub)/sum(var(score_sub))*100)>pca_var_thresh,1);
                if isempty(npc_sub); npc_sub=size(score_sub,2); end
                npc_sub=min(npc_sub,size(score_sub,2));
                X_train_sub_pca=score_sub(:,1:npc_sub);
                X_test_sub_pca=(X_test_sub-mean(X_train_sub))*coeff_sub(:,1:npc_sub);
                writematrix(X_train_sub_pca,'temp_sub_cv.csv');
                fid=fopen('run_sub_cv.py','w');
                fprintf(fid,'import os; os.environ["NUMBA_NUM_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"\n');
                fprintf(fid,'import numpy as np; np.random.seed(0)\nimport pandas as pd\nimport scanpy as sc\nimport anndata as ad\n');
                fprintf(fid,'data=pd.read_csv("temp_sub_cv.csv",header=None)\nadata=ad.AnnData(data.values)\n');
                fprintf(fid,'sc.pp.neighbors(adata,n_neighbors=%d,metric="correlation",random_state=0)\n',leiden_n_neighbors);
                fprintf(fid,'sc.tl.leiden(adata,resolution=%.2f,flavor="igraph",n_iterations=2,directed=False,random_state=0)\n',leiden_resolution);
                fprintf(fid,'pd.DataFrame(adata.obs["leiden"].values.astype(int)+1).to_csv("temp_sub_cv_out.csv",index=False,header=False)\n');
                fclose(fid);
                status_sub=system(sprintf('conda run -n %s python run_sub_cv.py',conda_env));
                if status_sub==0
                    sub_clust=readmatrix('temp_sub_cv_out.csv');
                    delete('temp_sub_cv.csv','temp_sub_cv_out.csv','run_sub_cv.py');
                    k_sub_cv=max(sub_clust);
                    sub_to_fine_cv=zeros(k_sub_cv,1);
                    for sc_i=1:k_sub_cv
                        sc_profile=mean(X_train_sub(sub_clust==sc_i,:),1);
                        d=pdist2(sc_profile,fine_centroids_cv,'correlation');
                        [~,best]=min(d);
                        sub_to_fine_cv(sc_i)=fine_types(best);
                    end
                    fine_leiden_cv=zeros(n_sub,1);
                    for ii=1:n_sub
                        d=pdist2(X_test_sub_pca(ii,:),X_train_sub_pca,'euclidean');
                        [~,nn_idx]=mink(d,knn_k);
                        fine_leiden_cv(ii)=sub_to_fine_cv(mode(sub_clust(nn_idx)));
                    end
                else
                    fine_leiden_cv=fine_centroid_cv;
                    try; delete('temp_sub_cv.csv','run_sub_cv.py'); catch; end
                end
            else
                fine_leiden_cv=fine_centroid_cv;
            end
            X_test_sub_qn=zeros(size(X_test_sub));
            for ii=1:n_genes
                [~,so]=sort(X_test_sub(:,ii));
                ts=sort(X_train_sub(:,ii));
                tq=interp1(linspace(0,1,length(ts)),ts,linspace(0,1,n_sub),'linear');
                X_test_sub_qn(so,ii)=tq';
            end
            fine_qn_cv=predict(knn_fine_cv,X_test_sub_qn);
            fine_preds_cv={fine_knn_cv,fine_centroid_cv,fine_leiden_cv,fine_qn_cv};
            for ci=1:n_sub
                labels=[fine_preds_cv{1}(ci),fine_preds_cv{2}(ci),fine_preds_cv{3}(ci),fine_preds_cv{4}(ci)];
                [dummy_mode,freq_count]=mode(labels);
                fine_label_cv(mega_test_idx(ci))=dummy_mode;
                fine_conf_cv(mega_test_idx(ci))=freq_count>=consensus_thresh;
            end
        end
        both_conf_cv=mega_conf_cv&fine_conf_cv;
        all_true=[all_true;y_test];
        all_pred_single=[all_pred_single;pred_single];
        all_pred_twostage=[all_pred_twostage;fine_label_cv];
        all_confident=[all_confident;both_conf_cv];
    end
    acc_single=100*sum(all_pred_single==all_true)/length(all_true);
    valid_cv=all_pred_twostage>0;
    conf_cv=all_confident&valid_cv;
    acc_twostage_conf=100*sum(all_pred_twostage(conf_cv)==all_true(conf_cv))/sum(conf_cv);
else
    acc_single=NaN; acc_twostage_conf=NaN;
end

cv_mega=crossval(knn_mdl_mega,'KFold',n_cv_folds);

mega_knn=predict(knn_mdl_mega,X_z);
dist_mega=zeros(n_cells,n_mega);
for ii=1:n_mega
    dist_mega(:,ii)=1-(X_z*ren_centroids_mega(ii,:)')./(sqrt(sum(X_z.^2,2)).*sqrt(sum(ren_centroids_mega(ii,:).^2)));
end
[~,mega_centroid]=min(dist_mega,[],2);
mean_z_per_cluster=zeros(k,length(gene_label));
for ii=1:k
    mean_z_per_cluster(ii,:)=mean(X_z(cluster_id==ii,:),1);
end
cluster_to_mega=zeros(k,1);
for ii=1:k
    d=pdist2(mean_z_per_cluster(ii,:),ren_centroids_mega,'correlation');
    [~,best]=min(d);
    cluster_to_mega(ii)=best;
end
mega_cluster=cluster_to_mega(cluster_id);
X_qn=zeros(size(X_z));
for ii=1:length(gene_label)
    [~,star_order]=sort(X_z(:,ii));
    ren_sorted=sort(ren_z(:,ii));
    ren_quantiles=interp1(linspace(0,1,length(ren_sorted)),ren_sorted,linspace(0,1,n_cells),'linear');
    X_qn(star_order,ii)=ren_quantiles';
end
mega_qn=predict(knn_mdl_mega,X_qn);
mega_preds={mega_knn,mega_centroid,mega_cluster,mega_qn};
mega_label=zeros(n_cells,1); mega_confident=false(n_cells,1);
for ii=1:n_cells
    labels=[mega_preds{1}(ii),mega_preds{2}(ii),mega_preds{3}(ii),mega_preds{4}(ii)];
    [dummy_mode,freq_count]=mode(labels);
    mega_label(ii)=dummy_mode;
    mega_confident(ii)=freq_count>=consensus_thresh;
end

fine_label=zeros(n_cells,1); fine_confident=false(n_cells,1);
for mi=1:n_mega
    fine_types=mega_to_fine{mi};
    mega_idx=find(mega_label==mi&mega_confident);
    n_mega_cells=length(mega_idx);
    if length(fine_types)==1
        fine_label(mega_idx)=fine_types(1);
        fine_confident(mega_idx)=true;
        continue;
    end
    ren_in_mega=ismember(original_label_arr,fine_types);
    ren_z_sub=ren_z(ren_in_mega,:);
    ren_labels_sub=original_label_arr(ren_in_mega);
    fine_centroids=zeros(length(fine_types),length(gene_label));
    for fi=1:length(fine_types)
        fine_centroids(fi,:)=mean(ren_z(original_label_arr==fine_types(fi),:),1);
    end
    if size(ren_z_sub,1)>20
        knn_sub=fitcknn(ren_z_sub,ren_labels_sub,'NumNeighbors',knn_k,'Distance','correlation');
    else
        knn_sub=fitcknn(ren_z_sub,ren_labels_sub,'NumNeighbors',3,'Distance','correlation');
    end
    X_z_mega=X_z(mega_idx,:);
    fine_knn=predict(knn_sub,X_z_mega);
    dist_fine=zeros(n_mega_cells,length(fine_types));
    for fi=1:length(fine_types)
        dist_fine(:,fi)=1-(X_z_mega*fine_centroids(fi,:)')./(sqrt(sum(X_z_mega.^2,2)).*sqrt(sum(fine_centroids(fi,:).^2)));
    end
    [~,nearest_fine]=min(dist_fine,[],2);
    fine_centroid_pred=fine_types(nearest_fine)';
    X_pca_mega=X_pca(mega_idx,:);
    writematrix(X_pca_mega,'temp_sub.csv');
    fid=fopen('run_sub.py','w');
    fprintf(fid,'import os; os.environ["NUMBA_NUM_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"\n');
    fprintf(fid,'import numpy as np; np.random.seed(0)\nimport pandas as pd\nimport scanpy as sc\nimport anndata as ad\n');
    fprintf(fid,'data=pd.read_csv("temp_sub.csv",header=None)\nadata=ad.AnnData(data.values)\n');
    fprintf(fid,'sc.pp.neighbors(adata,n_neighbors=%d,metric="correlation",random_state=0)\n',leiden_n_neighbors);
    fprintf(fid,'sc.tl.leiden(adata,resolution=%.2f,flavor="igraph",n_iterations=2,directed=False,random_state=0)\n',leiden_resolution);
    fprintf(fid,'pd.DataFrame(adata.obs["leiden"].values.astype(int)+1).to_csv("temp_sub_out.csv",index=False,header=False)\n');
    fclose(fid);
    status=system(sprintf('conda run -n %s python run_sub.py',conda_env));
    if status==0
        sub_cluster=readmatrix('temp_sub_out.csv');
        delete('temp_sub.csv','temp_sub_out.csv','run_sub.py');
        k_sub=max(sub_cluster);
        sub_to_fine=zeros(k_sub,1);
        for sc_i=1:k_sub
            sub_profile=mean(X_z_mega(sub_cluster==sc_i,:),1);
            d=pdist2(sub_profile,fine_centroids,'correlation');
            [~,best]=min(d);
            sub_to_fine(sc_i)=fine_types(best);
        end
        fine_cluster_pred=sub_to_fine(sub_cluster);
    else
        fine_cluster_pred=fine_centroid_pred;
        delete('temp_sub.csv','run_sub.py');
    end
    X_qn_mega=X_qn(mega_idx,:);
    fine_qn=predict(knn_sub,X_qn_mega);
    fine_preds_mega={fine_knn,fine_centroid_pred,fine_cluster_pred,fine_qn};
    for ci=1:n_mega_cells
        labels=[fine_preds_mega{1}(ci),fine_preds_mega{2}(ci),fine_preds_mega{3}(ci),fine_preds_mega{4}(ci)];
        [dummy_mode,freq_count]=mode(labels);
        fine_label(mega_idx(ci))=dummy_mode;
        fine_confident(mega_idx(ci))=freq_count>=consensus_thresh;
    end
end
both_confident=mega_confident&fine_confident;
n_conf=sum(both_confident);

%%% barcode setup
min_nonzero_for_sweep=10;
min_npos_for_chi2=5;
min_valid_bins=3;
thresh_max_cap=50;

snr_excluded_positions=[2 4 1; 2 4 2; 6 3 2];

inj_site_label=cell(5,7,2);
inj_site_label(:,:,1)={...
    'dCA1' 'GPe'  'DMS' 'vCA1' 'MOp'  'DN'     'PVT';...
    'PVT'  'OB'   'VLS' 'dCA3' 'ORB'  'MGB'    'MEPO';...
    'MOp'  'mPFC' 'cCP' 'ENT'  'SSp'  'Vermis' 'ACB-Core';...
    'GPe'  []     'GPe' 'dCA1' 'mPFC' 'LGd'    'OB';...
    'SC'   'LGd'  'CeA' []     'BLA'  'SC'     'VTA'};
inj_site_label(:,:,2)={...
    'dCA1' 'GPe'  'DMS' 'vCA1' 'MOp'  'DN'  'PVT';...
    'PVT'  'OB'   'VLS' 'dCA3' 'ORB'  'MGB' 'MEPO';...
    'MOp'  'mPFC' 'cCP' 'ENT'  'SSp'  []    'ACB-Core';...
    'GPe'  []     'GPe' 'dCA1' 'mPFC' 'LGd' 'OB';...
    'SC'   'LGd'  'CeA' []     'BLA'  'SC'  'VTA'};

combo_to_group=zeros(7,5,2);
combo_to_group(:,:,1)=[1 4 3 2 5;2 4 3 0 5;2 2 2 2 2;1 1 1 1 0;3 3 3 3 3;5 5 5 5 5;4 4 4 4 4];
combo_to_group(:,:,2)=[1 4 3 2 5;2 4 3 0 5;2 2 2 2 2;1 1 1 1 0;3 3 3 3 3;5 5 0 5 5;4 4 4 4 4];
group_names={'G1:HPC','G2:BG','G3:CTX','G4:MI','G5:BSLT'};

conf_labels_bg=fine_label(both_confident);
conf_mouse_bg=mouse_ids_per_cell(both_confident);
n_conf_bg=sum(both_confident);
bg_dist=zeros(n_subtypes,1);
for si=1:n_subtypes
    bg_dist(si)=sum(conf_labels_bg==keep_clusters_fine(si))/n_conf_bg;
end
k_bins=sum(bg_dist>0);

bc_binary=zeros(n_cells,5);
bc_thresh_used=zeros(n_mice,5);
for ii=1:n_mice
    mid=mouse_ids(ii);
    m_idx_all=find(mouse_ids_per_cell==mid);
    m_idx_conf=find(conf_mouse_bg==mid);
    c=combo_id_per_mouse(ii);
    sp=sex_page_per_mouse(ii);
    for ch=1:5
        target=inj_site_label{ch,c,sp};
        if isempty(target); continue; end
        counts_all=double(sero_table_filt(m_idx_all,bc_cols(ch)));
        counts_conf=double(sero_table_filt(both_confident,bc_cols(ch)));
        counts_conf_m=counts_conf(m_idx_conf);
        labels_conf_m=conf_labels_bg(m_idx_conf);
        max_count=max(counts_all);
        nz=counts_all(counts_all>0);
        if length(nz)<min_nonzero_for_sweep
            bc_binary(m_idx_all,ch)=counts_all>=bc_cnt_start;
            bc_thresh_used(ii,ch)=bc_cnt_start;
            continue;
        end
        thresh_range=bc_cnt_start:1:min(max(ceil(max_count/2),bc_cnt_start+1),thresh_max_cap);
        cramers_v=zeros(length(thresh_range),1);
        n_pos_vals=zeros(length(thresh_range),1);
        for ti=1:length(thresh_range)
            thr=thresh_range(ti);
            is_pos=counts_conf_m>=thr;
            n_pos=sum(is_pos);
            n_pos_vals(ti)=n_pos;
            if n_pos<min_npos_for_chi2; continue; end
            obs_dist=zeros(n_subtypes,1);
            for si=1:n_subtypes
                obs_dist(si)=sum(labels_conf_m(is_pos)==keep_clusters_fine(si));
            end
            expected=bg_dist*n_pos;
            valid_bins=expected>0;
            if sum(valid_bins)<min_valid_bins; continue; end
            chi2=sum((obs_dist(valid_bins)-expected(valid_bins)).^2./expected(valid_bins));
            cramers_v(ti)=sqrt(chi2/(n_pos*max(k_bins-1,1)));
        end
        valid_mask=n_pos_vals>=min_npos_for_chi2;
        if any(valid_mask)
            score=cramers_v; score(~valid_mask)=0;
            [~,best_idx]=max(score);
            opt_thresh=thresh_range(best_idx);
        else
            opt_thresh=bc_cnt_start;
        end
        bc_binary(m_idx_all,ch)=counts_all>=opt_thresh;
        bc_thresh_used(ii,ch)=opt_thresh;
    end
end

group_binary=zeros(n_cells,5);
for ii=1:n_mice
    mid=mouse_ids(ii);
    m_idx=find(mouse_ids_per_cell==mid);
    c=combo_id_per_mouse(ii);
    sp=sex_page_per_mouse(ii);
    for ch=1:5
        grp=combo_to_group(c,ch,sp);
        if grp==0; continue; end
        group_binary(m_idx,grp)=group_binary(m_idx,grp)|bc_binary(m_idx,ch);
    end
end

%%% fisher test
q_fdr=0.05;
min_n_per_cell_for_test=5;
rep1_mice_max=7;

conf_idx=find(both_confident);
conf_labels=fine_label(conf_idx);
conf_group=group_binary(conf_idx,:);
conf_mouse=mouse_ids_per_cell(conf_idx);
n_conf_total=length(conf_idx);

testable_mice=cell(5,1);
for gg=1:5
    testable=[];
    for ii=1:n_mice
        c=combo_id_per_mouse(ii);
        sp=sex_page_per_mouse(ii);
        for ch=1:5
            if combo_to_group(c,ch,sp)==gg
                m_idx_tmp=find(mouse_ids_per_cell==mouse_ids(ii));
                if sum(bc_binary(m_idx_tmp,ch))>0
                    testable=[testable;ii]; break;
                end
            end
        end
    end
    testable_mice{gg}=unique(testable);
end

pval_fisher=ones(n_subtypes,5);
or_fisher=ones(n_subtypes,5);
er_fisher=ones(n_subtypes,5);
rate_final=zeros(n_subtypes,5);
n_mat=zeros(n_subtypes,5);
consistent_final=false(n_subtypes,5);
tested=false(n_subtypes,5);

for si=1:n_subtypes
    ren_c=keep_clusters_fine(si);
    for gg=1:5
        in_sub=conf_labels==ren_c;
        is_pos=conf_group(:,gg)==1;
        n_sub=sum(in_sub); n_grp=sum(is_pos); n_both=sum(in_sub&is_pos);
        if n_sub<min_n_per_cell_for_test||n_grp<min_n_per_cell_for_test; continue; end
        a=n_both; b=n_sub-a; cc_val=n_grp-a; d=n_conf_total-n_sub-n_grp+a;
        if min([a b cc_val d])<0; continue; end
        tested(si,gg)=true;
        [~,p]=fishertest(table([a;cc_val],[b;d],'VariableNames',{'Pos','Neg'},'RowNames',{'In','Out'}));
        if b*cc_val>0; or_val=(a*d)/(b*cc_val); else; or_val=Inf; end
        er_val=(a/max(n_grp,1))/(n_sub/max(n_conf_total,1));
        pval_fisher(si,gg)=p;
        or_fisher(si,gg)=or_val;
        er_fisher(si,gg)=er_val;
        rate_final(si,gg)=100*a/max(n_sub,1);
        n_mat(si,gg)=n_sub;

        test_mice=testable_mice{gg};
        rep1=test_mice(test_mice<=rep1_mice_max);
        rep2=test_mice(test_mice>rep1_mice_max);
        or_rep=zeros(2,1);
        for ri=1:2
            if ri==1; rmice=rep1; else; rmice=rep2; end
            if isempty(rmice); or_rep(ri)=NaN; continue; end
            rmask=false(n_conf_total,1);
            for mi=1:length(rmice)
                rmask=rmask|(conf_mouse==mouse_ids(rmice(mi)));
            end
            in_s=conf_labels(rmask)==ren_c;
            is_p=conf_group(rmask,gg)==1;
            ra=sum(in_s&is_p); rb=sum(in_s&~is_p);
            rc=sum(~in_s&is_p); rd=sum(~in_s&~is_p);
            if rb*rc>0; or_rep(ri)=(ra*rd)/(rb*rc); else; or_rep(ri)=Inf; end
        end
        if ~any(isnan(or_rep)); consistent_final(si,gg)=all(or_rep>1); end
    end
end

pval_final=pval_fisher;
or_final=or_fisher;
er_final=er_fisher;

padj_posthoc=ones(n_subtypes,5);
r2_posthoc_total=0;
n_tested_total=0;
for gg=1:5
    test_idx_g=find(tested(:,gg));
    n_g=length(test_idx_g);
    if n_g==0; continue; end
    n_tested_total=n_tested_total+n_g;
    raw_p_g=pval_final(test_idx_g,gg);
    [sorted_p_g,sort_order_g]=sort(raw_p_g);

    q1=q_fdr/(1+q_fdr);
    bh1_thresh_g=(1:n_g)'/n_g*q1;
    r1_g=find(sorted_p_g<=bh1_thresh_g,1,'last');
    if isempty(r1_g); r1_g=0; end
    m0_g=max(n_g-r1_g,1);

    q2_g=q_fdr*n_g/m0_g;
    bh2_thresh_g=(1:n_g)'/n_g*q2_g;
    r2_g=find(sorted_p_g<=bh2_thresh_g,1,'last');
    if isempty(r2_g); r2_g=0; end
    r2_posthoc_total=r2_posthoc_total+r2_g;

    posthoc_padj_g=ones(n_g,1);
    for ii=n_g:-1:1
        raw_adj=sorted_p_g(ii)*n_g/ii*(m0_g/n_g);
        if ii<n_g; posthoc_padj_g(ii)=min(raw_adj,posthoc_padj_g(ii+1));
        else; posthoc_padj_g(ii)=min(raw_adj,1); end
    end
    posthoc_unsorted_g=zeros(n_g,1);
    posthoc_unsorted_g(sort_order_g)=posthoc_padj_g;
    for ii=1:n_g
        padj_posthoc(test_idx_g(ii),gg)=posthoc_unsorted_g(ii);
    end
end
n_tested=n_tested_total;
r2_posthoc=r2_posthoc_total;

%%% umap
umap_n_neighbors=200;
umap_min_dist=0.25;
umap_spread=4;

colors_fine=[...
     31 120 180; 227  26  28;  51 160  44; 255 127   0; 106  61 154;...
    177  89  40;   0 190 190; 240 200   0; 220  60 150;  80  80  80]/255;
colors_group=[0.20 0.70 0.20; 0.90 0.20 0.20; 0.10 0.40 0.90; 0.70 0.00 0.70; 0.95 0.65 0.00];
colors_14=[...
    0.90 0.10 0.10; 0.20 0.60 0.20; 0.10 0.10 0.90; 0.95 0.60 0.00; 0.60 0.00 0.60;...
    0.00 0.80 0.80; 0.80 0.40 0.60; 0.50 0.50 0.00; 0.00 0.40 0.00; 0.40 0.20 0.00;...
    0.00 0.00 0.50; 0.90 0.40 0.40; 0.40 0.70 0.40; 0.50 0.50 0.80];

cell_colors_all=zeros(n_cells,3);
for ii=1:n_cells
    ri=find(keep_clusters_fine==fine_label(ii),1);
    if ~isempty(ri); cell_colors_all(ii,:)=colors_fine(ri,:); else; cell_colors_all(ii,:)=[0.7 0.7 0.7]; end
end

if run_ren_cv
    conf_mask_cv=all_confident&all_pred_twostage>0;
    true_conf_cv=all_true(conf_mask_cv);
    pred_conf_cv=all_pred_twostage(conf_mask_cv);
    true_names_cv=cell(size(true_conf_cv));
    pred_names_cv=cell(size(pred_conf_cv));
    for ii=1:n_subtypes
        true_names_cv(true_conf_cv==keep_clusters_fine(ii))=ren_names_fine(ii);
        pred_names_cv(pred_conf_cv==keep_clusters_fine(ii))=ren_names_fine(ii);
    end
    valid_cv_cm=~cellfun(@isempty,true_names_cv)&~cellfun(@isempty,pred_names_cv);
    true_cat_cv=categorical(true_names_cv(valid_cv_cm),ren_names_fine);
    pred_cat_cv=categorical(pred_names_cv(valid_cv_cm),ren_names_fine);
    acc_cv_cm=100*sum(true_cat_cv==pred_cat_cv)/length(true_cat_cv);
    fig_cm=figure('Position',[100 100 800 700],'Color','w');
    cm_obj=confusionchart(true_cat_cv,pred_cat_cv);
    cm_obj.Title=sprintf('Ren CV: Two-stage confident (n=%d, acc=%.1f%%)',sum(valid_cv_cm),acc_cv_cm);
    cm_obj.XLabel='Predicted'; cm_obj.YLabel='True'; cm_obj.FontSize=8;
    cm_obj.RowSummary='row-normalized';
    cm_obj.ColumnSummary='column-normalized';
    sortClasses(cm_obj,ren_names_fine);
end

X_conf_pca=X_pca(both_confident,:);
colors_conf=cell_colors_all(both_confident,:);
X_filt_conf=X_filt(both_confident,:);
mouse_conf=mouse_ids_per_cell(both_confident);
writematrix(X_conf_pca,'temp_umap_in.csv');
fid=fopen('run_umap.py','w');
fprintf(fid,'import os; os.environ["NUMBA_NUM_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"\n');
fprintf(fid,'import numpy as np; np.random.seed(0)\nimport umap\nimport warnings\nwarnings.filterwarnings("ignore")\n');  % umap-learn=0.5.9
fprintf(fid,'X=np.loadtxt("temp_umap_in.csv",delimiter=",")\n');
fprintf(fid,'reducer=umap.UMAP(n_neighbors=%d,min_dist=%.2f,metric="correlation",random_state=%d,densmap=True,spread=%d)\n', ...
    umap_n_neighbors,umap_min_dist,rand_seed_final,umap_spread);
fprintf(fid,'Y=reducer.fit_transform(X)\nnp.savetxt("temp_umap_out.csv",Y,delimiter=",")\n');
fclose(fid);
system(sprintf('conda run -n %s python run_umap.py 2>/dev/null',conda_env));
umap_coords=readmatrix('temp_umap_out.csv');
delete('temp_umap_in.csv','temp_umap_out.csv','run_umap.py');

fig_u1=figure('Position',[100 100 800 700],'Color','w');
scatter(umap_coords(:,1),umap_coords(:,2),5,colors_conf,'filled','MarkerFaceAlpha',0.8);
title(sprintf('UMAP: Ren subtypes (confident, %.1f%%, seed=%d)',100*n_conf/n_cells,rand_seed_final),'FontSize',14);
xlabel('UMAP-1'); ylabel('UMAP-2');
hold on;
h=gobjects(length(keep_clusters_fine),1);
for ii=1:length(keep_clusters_fine); h(ii)=scatter(NaN,NaN,40,colors_fine(ii,:),'filled'); end
legend(h,ren_names_fine,'Location','bestoutside','FontSize',7);
hold off;

tph2_conf=sero_table_filt(both_confident,tph2_col);
sert_conf=sero_table_filt(both_confident,sert_col);
all_gene_names=[marker_names_ordered,{'Tph2','Slc6a4'}];
all_gene_data=[X_filt_conf,double(tph2_conf),double(sert_conf)];
n_all_genes=length(all_gene_names);
fig_u2=figure('Position',[50 50 1800 900],'Color','w');
for gg=1:n_all_genes
    subplot(3,4,gg);
    scatter(umap_coords(:,1),umap_coords(:,2),3,log2(all_gene_data(:,gg)+1),'filled','MarkerFaceAlpha',0.4);
    colormap(gca,hot(256));
    max_val=prctile(log2(all_gene_data(:,gg)+1),99);
    if max_val<=0; max_val=1; end
    caxis([0 max_val]);
    cb=colorbar; cb.FontSize=6;
    title(all_gene_names{gg},'FontSize',10,'FontWeight','bold');
    set(gca,'XTick',[],'YTick',[]); axis tight;
end
sgtitle('UMAP: gene expression (confident cells)','FontSize',14);

%%% enrichment anal
fig_f=figure('Position',[100 100 1400 500],'Color','w');

subplot(1,2,1);
log2er=log2(er_final);
log2er(isinf(log2er)&log2er>0)=5;
log2er(isinf(log2er)&log2er<0)=-5;
imagesc(log2er); try; colormap(gca,redblue(1000)); catch; colormap(gca,parula(256)); end
caxis([-3 3]); colorbar;
hold on;
for si=1:n_subtypes
    for gg=1:5
        if padj_posthoc(si,gg)<0.001
            text(gg,si,'***','HorizontalAlignment','center','Color','k','FontSize',12,'FontWeight','bold');
        elseif padj_posthoc(si,gg)<0.01
            text(gg,si,'**','HorizontalAlignment','center','Color','k','FontSize',11,'FontWeight','bold');
        elseif padj_posthoc(si,gg)<0.05
            text(gg,si,'*','HorizontalAlignment','center','Color','k','FontSize',12);
        end
    end
end
hold off;
set(gca,'XTick',1:5,'XTickLabel',group_names,'YTick',1:n_subtypes,'YTickLabel',ren_names_fine);
title('Log2 Enrichment Ratio (* q<0.05, ** q<0.01, *** q<0.001)');
xtickangle(30);

subplot(1,2,2);
imagesc(rate_final); colorbar;
set(gca,'XTick',1:5,'XTickLabel',group_names,'YTick',1:n_subtypes,'YTickLabel',ren_names_fine);
title('% positive in subtype');
xtickangle(30);
for si=1:n_subtypes
    for gg=1:5
        if rate_final(si,gg)>0
            text(gg,si,sprintf('%.0f%%',rate_final(si,gg)),'HorizontalAlignment','center','FontSize',6);
        end
    end
end
sgtitle('Ren subtype x Projectomic group (Fisher + posthoc)');

%%% coprojection anal
cross_group_combos=[1 2];
within_combos_per_group=[4 3 5 7 6];
min_npos_for_site_pair=3;
min_jaccard_for_site_pair=0.05;

min_npos_per_site=3;
all_sites={}; site_bc_conf={}; site_group_id=[];
for ii=1:n_mice
    c=combo_id_per_mouse(ii); sp=sex_page_per_mouse(ii); mid=mouse_ids(ii);
    for ch=1:5
        target=inj_site_label{ch,c,sp};
        if isempty(target); continue; end
        m_mask=mouse_conf==mid;
        ch_pos=bc_binary(both_confident,ch);
        is_pos=m_mask&ch_pos==1;
        if sum(is_pos)<min_npos_per_site; continue; end
        site_idx=find(strcmp(all_sites,target));
        if isempty(site_idx)
            all_sites{end+1}=target;
            site_bc_conf{end+1}=is_pos;
            site_group_id(end+1)=combo_to_group(c,ch,sp);
        else
            site_bc_conf{site_idx}=site_bc_conf{site_idx}|is_pos;
        end
    end
end

all_site_names_tbl={}; all_site_groups_tbl=[]; mouse_site_counts_tbl=[];
for ii=1:n_mice
    mid=mouse_ids(ii); m_idx=find(mouse_ids_per_cell==mid);
    c=combo_id_per_mouse(ii); sp=sex_page_per_mouse(ii);
    for ch=1:5
        target=inj_site_label{ch,c,sp};
        if isempty(target); continue; end
        si=find(strcmp(all_site_names_tbl,target));
        if isempty(si)
            all_site_names_tbl{end+1}=target;
            all_site_groups_tbl(end+1)=combo_to_group(c,ch,sp);
            si=length(all_site_names_tbl);
            mouse_site_counts_tbl(:,si)=0;
        end
        mouse_site_counts_tbl(ii,si)=sum(bc_binary(m_idx,ch)==1);
    end
end

conf_group_bin=group_binary(both_confident,:);

n_sites_cp=length(all_sites);
site_pairs=[];
for s1=1:n_sites_cp
    for s2=s1+1:n_sites_cp
        is_s1=site_bc_conf{s1}; is_s2=site_bc_conf{s2};
        n_inter=sum(is_s1&is_s2);
        n_union=sum(is_s1|is_s2);
        if n_inter<min_npos_for_site_pair&&(n_union==0||n_inter/n_union<min_jaccard_for_site_pair); continue; end
        n_s1=sum(is_s1); n_s2=sum(is_s2);
        jacc=n_inter/max(n_union,1);
        site_pairs=[site_pairs; ...
            s1 s2 n_inter n_s1 n_s2 jacc 100*n_inter/max(n_s1,1) 100*n_inter/max(n_s2,1)];
    end
end

n_groups_per_cell=sum(conf_group_bin,2);

cross_total_bc=0; cross_total_multi=0;
for ii=1:n_mice
    mid=mouse_ids(ii); c=combo_id_per_mouse(ii);
    if ~ismember(c,cross_group_combos); continue; end
    m_idx=find(mouse_ids_per_cell==mid);
    sp=sex_page_per_mouse(ii);
    bc_m=bc_binary(m_idx,:);
    active_chs=[];
    for ch=1:5
        if ~isempty(inj_site_label{ch,c,sp}); active_chs=[active_chs;ch]; end
    end
    n_any=sum(any(bc_m(:,active_chs),2));
    n_multi=sum(sum(bc_m(:,active_chs),2)>=2);
    cross_total_bc=cross_total_bc+n_any;
    cross_total_multi=cross_total_multi+n_multi;
end

within_total_bc=0; within_total_multi=0;
for gi=1:5
    tc=within_combos_per_group(gi);
    grp_bc=0; grp_multi=0;
    for ii=1:n_mice
        c=combo_id_per_mouse(ii);
        if c~=tc; continue; end
        m_idx=find(mouse_ids_per_cell==mouse_ids(ii));
        sp=sex_page_per_mouse(ii);
        bc_m=bc_binary(m_idx,:);
        active_chs=[];
        for ch=1:5
            if ~isempty(inj_site_label{ch,c,sp}); active_chs=[active_chs;ch]; end
        end
        n_any=sum(any(bc_m(:,active_chs),2));
        n_multi=sum(sum(bc_m(:,active_chs),2)>=2);
        grp_bc=grp_bc+n_any;
        grp_multi=grp_multi+n_multi;
    end
    within_total_bc=within_total_bc+grp_bc;
    within_total_multi=within_total_multi+grp_multi;
end

cross_single=cross_total_bc-cross_total_multi;
within_single=within_total_bc-within_total_multi;
[~,p_cw]=fishertest(table( ...
    [cross_total_multi;within_total_multi], ...
    [cross_single;within_single], ...
    'VariableNames',{'Multi','Single'},'RowNames',{'Cross','Within'}));

p_per_group=zeros(5,1);
for gi=1:5
    tc=within_combos_per_group(gi);
    grp_bc=0; grp_multi=0;
    for ii=1:n_mice
        c=combo_id_per_mouse(ii);
        if c~=tc; continue; end
        m_idx=find(mouse_ids_per_cell==mouse_ids(ii));
        sp=sex_page_per_mouse(ii);
        bc_m=bc_binary(m_idx,:);
        active_chs=[];
        for ch=1:5
            if ~isempty(inj_site_label{ch,c,sp}); active_chs=[active_chs;ch]; end
        end
        grp_bc=grp_bc+sum(any(bc_m(:,active_chs),2));
        grp_multi=grp_multi+sum(sum(bc_m(:,active_chs),2)>=2);
    end
    grp_single=grp_bc-grp_multi;
    [~,p_per_group(gi)]=fishertest(table( ...
        [grp_multi;cross_total_multi], ...
        [grp_single;cross_single], ...
        'VariableNames',{'Multi','Single'},'RowNames',{'Within','Cross'}));
end

%%% coprojection bar charts
n_perm=50000;
q_fdr_co=0.05;
disp_labels={'HPC','BG','CTX','MI','BSLT'};

within_rates=zeros(5,1);
within_n=zeros(5,1);
within_multi=zeros(5,1);
within_se=zeros(5,1);
within_mouse_data=cell(5,1);

for gi=1:5
    tc=within_combos_per_group(gi);
    mouse_data={};
    total_any=0; total_multi=0;
    for mm=1:n_mice
        c=combo_id_per_mouse(mm);
        if c~=tc; continue; end
        m_idx=find(mouse_ids_per_cell==mouse_ids(mm));
        sp=sex_page_per_mouse(mm);
        bc_m=bc_binary(m_idx,:);
        active_chs=[];
        for ch=1:5
            if ~isempty(inj_site_label{ch,c,sp})
                active_chs=[active_chs;ch];
            end
        end
        mouse_data{end+1}=bc_m(:,active_chs);
        total_any=total_any+sum(any(bc_m(:,active_chs),2));
        total_multi=total_multi+sum(sum(bc_m(:,active_chs),2)>=2);
    end
    within_mouse_data{gi}=mouse_data;
    within_n(gi)=total_any;
    within_multi(gi)=total_multi;
    if total_any>0
        p_hat=total_multi/total_any;
        within_rates(gi)=100*p_hat;
        within_se(gi)=100*sqrt(p_hat*(1-p_hat)/total_any);
    end
end

cross_mouse_data={};
cross_total_any=0; cross_total_multi_co=0;
for mm=1:n_mice
    c=combo_id_per_mouse(mm);
    if ~ismember(c,cross_group_combos); continue; end
    m_idx=find(mouse_ids_per_cell==mouse_ids(mm));
    sp=sex_page_per_mouse(mm);
    bc_m=bc_binary(m_idx,:);
    active_chs=[];
    for ch=1:5
        if ~isempty(inj_site_label{ch,c,sp})
            active_chs=[active_chs;ch];
        end
    end
    cross_mouse_data{end+1}=bc_m(:,active_chs);
    cross_total_any=cross_total_any+sum(any(bc_m(:,active_chs),2));
    cross_total_multi_co=cross_total_multi_co+sum(sum(bc_m(:,active_chs),2)>=2);
end
cross_bc_co=cross_total_any;
cross_p_co=cross_total_multi_co/max(cross_bc_co,1);
cross_rate=100*cross_p_co;
cross_se=100*sqrt(cross_p_co*(1-cross_p_co)/max(cross_bc_co,1));

null_diff=zeros(n_perm,5);
for pp=1:n_perm
    cn=perm_multi_rate(cross_mouse_data);
    for gi=1:5
        null_diff(pp,gi)=perm_multi_rate(within_mouse_data{gi})-cn;
    end
end

obs_diff=within_rates-cross_rate;
p_diff=zeros(5,1);
for gi=1:5
    p_diff(gi)=(sum(null_diff(:,gi)>=obs_diff(gi))+1)/(n_perm+1);
end

n_tests=5;
[sorted_p,sort_order]=sort(p_diff);

q1_co=q_fdr_co/(1+q_fdr_co);
bh1_thresh_co=(1:n_tests)'/n_tests*q1_co;
r1_co=find(sorted_p<=bh1_thresh_co,1,'last');
if isempty(r1_co); r1_co=0; end

m0_co=max(n_tests-r1_co,1);

q2_co=q_fdr_co*n_tests/m0_co;
bh2_thresh_co=(1:n_tests)'/n_tests*q2_co;
r2_co=find(sorted_p<=bh2_thresh_co,1,'last');
if isempty(r2_co); r2_co=0; end

bky_padj_co=ones(n_tests,1);
for ii=n_tests:-1:1
    raw_adj=sorted_p(ii)*n_tests/ii*(m0_co/n_tests);
    if ii<n_tests
        bky_padj_co(ii)=min(raw_adj,bky_padj_co(ii+1));
    else
        bky_padj_co(ii)=min(raw_adj,1);
    end
end
bky_q=zeros(n_tests,1);
bky_q(sort_order)=bky_padj_co;

disp_order=1:5;
disp_rates=within_rates(disp_order);
disp_se_val=within_se(disp_order);
disp_q=bky_q(disp_order);
disp_colors=colors_group(disp_order,:);

fig_co=figure('Position',[100 100 480 360],'Color','w');
hold on;

fill([0.4 5.6 5.6 0.4], ...
     [cross_rate-cross_se cross_rate-cross_se cross_rate+cross_se cross_rate+cross_se], ...
     [0.85 0.85 0.85],'EdgeColor','none','FaceAlpha',0.5);
plot([0.4 5.6],[cross_rate cross_rate],'--','Color',[0.4 0.4 0.4],'LineWidth',1.5);

b=bar(1:5,disp_rates,0.55,'FaceColor','flat','EdgeColor','none','FaceAlpha',0.85);
for gi=1:5; b.CData(gi,:)=disp_colors(gi,:); end

errorbar(1:5,disp_rates,disp_se_val,'k.','LineWidth',1.2,'CapSize',6);

for gi=1:5
    y_pos=disp_rates(gi)+disp_se_val(gi)+1.5;
    if disp_q(gi)<0.001;     sig_str='***';
    elseif disp_q(gi)<0.01;  sig_str='**';
    elseif disp_q(gi)<0.05;  sig_str='*';
    else;                    sig_str='ns';
    end
    text(gi,y_pos,sig_str,'HorizontalAlignment','center','FontSize',11,'FontWeight','bold');
end

hold off;
set(gca,'XTick',1:5,'XTickLabel',disp_labels);
ylabel('% multi-barcode among BC+ cells');
ylim([0 max(disp_rates+disp_se_val)*1.35]);
set(gca,'FontSize',9,'Box','off');

function rate=perm_multi_rate(mouse_data)
    perm_any=0; perm_multi=0;
    for mi=1:length(mouse_data)
        bc_mat=mouse_data{mi};
        n_cells_m=size(bc_mat,1);
        n_chs_m=size(bc_mat,2);
        perm_mat=zeros(n_cells_m,n_chs_m);
        for ci=1:n_chs_m
            perm_mat(:,ci)=bc_mat(randperm(n_cells_m),ci);
        end
        perm_any=perm_any+sum(any(perm_mat,2));
        perm_multi=perm_multi+sum(sum(perm_mat,2)>=2);
    end
    if perm_any>0
        rate=100*perm_multi/perm_any;
    else
        rate=0;
    end
end