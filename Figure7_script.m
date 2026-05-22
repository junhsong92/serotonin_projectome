rng('default');
clear all;
close all;
addpath(genpath(pwd));

target_cohorts=[1];
windsor_cap=5; % 5 STD
p_threshold=0.05;
lambda_reg=1; % L2 Reg strength
n_boot_iter=50000; % for bootstrap anova
max_injection_iters=10000;
n_perms_actual=10000; % # trials for permutation test
max_total_features_allowed=10;
sex_interest='M';

poolobj=gcp('nocreate');
if isempty(poolobj), parpool; end

csvFilePath='Supplementary_table_0behav_summary.csv';
rawT=readtable(csvFilePath);
temp_sex=upper(strtrim(string(rawT.Sex)));
if strcmp(sex_interest,'M')
    keep_mask=startsWith(temp_sex,'M');
    rawT=rawT(keep_mask,:);
elseif strcmp(sex_interest,'F')
    keep_mask=startsWith(temp_sex,'F');
    rawT=rawT(keep_mask,:);
end

raw_groups_str=string(rawT.Group);
keep_idx=false(height(rawT),1);
for cc=target_cohorts
    suffix=sprintf('_%d',cc);
    keep_idx=keep_idx|endsWith(raw_groups_str,suffix);
end
rawT=rawT(keep_idx,:);
rawT=rawT(~contains(string(rawT.Group),'xxx','IgnoreCase',true),:);

exclude_cols={'ID','mouse_ID','Batch','Animal','Sex','Group','CTRL_grp','Time'};
is_num=false(1,width(rawT));
for ii=1:width(rawT)
    if isnumeric(rawT{:,ii})&&~ismember(rawT.Properties.VariableNames{ii},exclude_cols)
        is_num(ii)=true;
    end
end
X_raw=table2array(rawT(:,is_num));
all_feat_names=rawT.Properties.VariableNames(is_num);

Raw_Group=regexprep(cellstr(rawT.Group),'_[0-9]+$','');
Raw_Sex=cellstr(rawT.Sex); Raw_Sex(cellfun(@isempty,Raw_Sex))={'Unknown'};
Anova_Grps=grp2idx(categorical(Raw_Group));
n_classes=max(Anova_Grps);
Cats=categories(categorical(Raw_Group));
Raw_IDs=string(table2cell(rawT(:,1)));
if size(X_raw,1)==0, error('!!! Error: No samples left after Cohort Filtering. !!!'); end

X_robust=zeros(size(X_raw));
unique_sexes=unique(Raw_Sex);
for ss=1:length(unique_sexes)
    curr_sex=unique_sexes{ss};
    sex_mask=strcmp(Raw_Sex,curr_sex);
    if sum(sex_mask)==0, continue; end

    X_sex=X_raw(sex_mask,:);

    M_sex=mean(X_sex,1,'omitnan');
    D_sex=std(X_sex,0,1,'omitnan');
    D_sex(D_sex==0)=1;

    X_sex_norm=(X_sex-M_sex)./D_sex;

    X_sex_norm(X_sex_norm>windsor_cap)=windsor_cap;
    X_sex_norm(X_sex_norm<-windsor_cap)=-windsor_cap;
    X_robust(sex_mask,:)=X_sex_norm;
end
if any(isnan(X_robust(:)))
    try, X_proc=knnimpute(X_robust',3)'; catch, X_proc=fillmissing(X_robust,'nearest'); end
else
    X_proc=X_robust;
end

R_mat=corr(X_proc);
R_mat=triu(R_mat,1);

n_total=size(X_proc,2);
is_sig=false(n_total,1);
p_values_boot=ones(n_total,1);
X_for_boot=X_proc;
Y_for_boot=Anova_Grps;
n_samples=length(Y_for_boot);
n_groups=length(unique(Y_for_boot));
parfor ii=1:n_total
    rng('default');
    feat_vec=X_for_boot(:,ii);
    if var(feat_vec)==0, continue; end
    [F_obs,~]=fast_anova_f(feat_vec,Y_for_boot,n_groups,n_samples);

    null_better_count=0;
    for bb=1:n_boot_iter
        Y_shuffled=Y_for_boot(randperm(n_samples));
        [F_null,~]=fast_anova_f(feat_vec,Y_shuffled,n_groups,n_samples);
        if F_null>=F_obs, null_better_count=null_better_count+1; end
    end
    p_values_boot(ii)=(null_better_count+1)/(n_boot_iter+1);
    if p_values_boot(ii)<p_threshold, is_sig(ii)=true; end
end
valid_idx=find(is_sig);
X_valid=X_proc(:,valid_idx);
feat_valid=all_feat_names(valid_idx);
n_valid=length(valid_idx);

mean_total=mean(X_valid);
Sb=zeros(n_valid); Sw=zeros(n_valid);
for kk=1:n_classes
    X_k=X_valid(Anova_Grps==kk,:);
    n_k=size(X_k,1);
    if n_k>0
        m_k=mean(X_k);
        diff_b=m_k-mean_total;
        Sb=Sb+n_k*(diff_b'*diff_b);
        if n_k>1, diff_w=X_k-m_k; Sw=Sw+(diff_w'*diff_w); end
    end
end
[V,D_eig]=eig(Sb,Sw+lambda_reg*eye(n_valid));
[eig_vals,sort_eig_idx]=sort(diag(D_eig),'descend');
W=real(V(:,sort_eig_idx));
max_calc_dims=n_classes-1;
scores=zeros(n_valid,1);
for ii=1:n_valid
    for dd=1:min(max_calc_dims,size(W,2))
        scores(ii)=scores(ii)+abs(W(ii,dd))*eig_vals(dd);
    end
end
[~,rank_idx]=sort(scores,'descend');
X_ranked=X_valid(:,rank_idx);
feat_ranked=feat_valid(rank_idx);
full_valid_indices=valid_idx(rank_idx);

Y_lda=Anova_Grps;
n_classes_lda=max(Y_lda);
max_lda_dims=n_classes_lda-1;
feat_counts=1:size(X_ranked,2);
feat_counts=feat_counts(feat_counts<=max_total_features_allowed);
n_feat_steps=length(feat_counts);
grid_avg_acc_map=zeros(n_feat_steps,max_lda_dims);
class_totals=zeros(n_classes_lda,1);
for cc=1:n_classes_lda, class_totals(cc)=sum(Y_lda==cc); end
tic;
reg_val=lambda_reg;
parfor f_i=1:n_feat_steps
    n_f=feat_counts(f_i);
    X_subset=X_ranked(:,1:n_f);
    class_hits=zeros(n_classes_lda,max_lda_dims);

    for ii=1:n_samples
        test_idx=ii; train_idx=setdiff(1:n_samples,test_idx);
        X_tr=X_subset(train_idx,:); Y_tr=Y_lda(train_idx);
        X_te=X_subset(test_idx,:); Y_te_true=Y_lda(test_idx);

        classes_tr=unique(Y_tr);
        m_tot=mean(X_tr); Sb_sub=zeros(n_f); Sw_sub=zeros(n_f);
        for c_idx=1:length(classes_tr)
            cc=classes_tr(c_idx);
            xk=X_tr(Y_tr==cc,:); nk=size(xk,1);
            if nk>0
                mk=mean(xk);
                Sb_sub=Sb_sub+nk*((mk-m_tot)'*(mk-m_tot));
                if nk>1, Sw_sub=Sw_sub+(xk-mk)'*(xk-mk); end
            end
        end

        [V_sub,D_sub]=eig(Sb_sub,Sw_sub+eye(n_f)*reg_val);
        [~,si]=sort(diag(D_sub),'descend');
        W_full=real(V_sub(:,si)); W_full=W_full(:,1:min(size(W_full,2),max_lda_dims));

        X_tr_proj_full=X_tr*W_full; X_te_proj_full=X_te*W_full;

        for dd=1:max_lda_dims
            if dd>size(W_full,2), continue; end
            X_tr_proj=X_tr_proj_full(:,1:dd); X_te_proj=X_te_proj_full(:,1:dd);

            Centroids=zeros(length(classes_tr),dd);
            for kk=1:length(classes_tr)
                Centroids(kk,:)=mean(X_tr_proj(Y_tr==classes_tr(kk),:),1);
            end

            [~,min_k]=min(sum((Centroids-X_te_proj).^2,2));
            pred_label=classes_tr(min_k);
            if pred_label==Y_te_true, class_hits(Y_te_true,dd)=class_hits(Y_te_true,dd)+1; end
        end
    end

    avg_accs_for_dim=zeros(1,max_lda_dims);
    for dd=1:max_lda_dims
        total_correct=sum(class_hits(:,dd));
        avg_accs_for_dim(dd)=(total_correct/n_samples)*100;
    end
    grid_avg_acc_map(f_i,:)=avg_accs_for_dim;
end
toc;
[max_avg_acc,lin_idx]=max(grid_avg_acc_map(:));
[best_f_idx,best_d]=ind2sub(size(grid_avg_acc_map),lin_idx);
best_n_feats=feat_counts(best_f_idx);

current_feat_indices=full_valid_indices(1:best_n_feats);
opt_n_dims=best_d;
calc_min_acc=@(labels_true,labels_pred,classes) ...
    min(arrayfun(@(cc) sum(labels_pred(labels_true==cc)==cc)/sum(labels_true==cc),classes))*100;
X_curr=X_proc(:,current_feat_indices);
n_curr_f=length(current_feat_indices);
pred_labels_init=zeros(n_samples,1);
for ii=1:n_samples
    test_idx=ii; train_idx=setdiff(1:n_samples,test_idx);
    X_tr=X_curr(train_idx,:); Y_tr=Y_lda(train_idx);
    X_te=X_curr(test_idx,:);
    classes_tr=unique(Y_tr);
    m_tot=mean(X_tr); Sb_sub=zeros(n_curr_f); Sw_sub=zeros(n_curr_f);
    for c_idx=1:length(classes_tr)
        cc=classes_tr(c_idx);
        xk=X_tr(Y_tr==cc,:); nk=size(xk,1);
        if nk>0
            mk=mean(xk);
            Sb_sub=Sb_sub+nk*((mk-m_tot)'*(mk-m_tot));
            if nk>1, Sw_sub=Sw_sub+(xk-mk)'*(xk-mk); end
        end
    end
    [V_sub,D_sub]=eig(Sb_sub,Sw_sub+eye(n_curr_f)*lambda_reg);
    [~,si]=sort(diag(D_sub),'descend');
    W_curr=real(V_sub(:,si)); W_curr=W_curr(:,1:min(opt_n_dims,size(W_curr,2)));
    X_tr_proj=X_tr*W_curr; X_te_proj=X_te*W_curr;
    Centroids=zeros(length(classes_tr),size(X_tr_proj,2));
    for kk=1:length(classes_tr), Centroids(kk,:)=mean(X_tr_proj(Y_tr==classes_tr(kk),:),1); end
    [~,min_k]=min(sum((Centroids-X_te_proj).^2,2));
    pred_labels_init(ii)=classes_tr(min_k);
end
current_min_acc=calc_min_acc(Y_lda,pred_labels_init,unique(Y_lda));
for iter=1:max_injection_iters
    if length(current_feat_indices)>=max_total_features_allowed
        break;
    end

    if current_min_acc>=100, break; end

    CM=confusionmat(Y_lda,pred_labels_init);
    CM_no_diag=CM; CM_no_diag(1:n_classes+1:end)=0;
    [row_idx,col_idx,vals]=find(CM_no_diag);
    if isempty(vals), break; end

    [~,sort_err_idx]=sort(vals,'descend');
    sorted_rows=row_idx(sort_err_idx); sorted_cols=col_idx(sort_err_idx);

    iter_improvement_found=false;
    for pair_k=1:length(sorted_rows)
        grp1=sorted_rows(pair_k); grp2=sorted_cols(pair_k);
        mask_pair=ismember(Y_lda,[grp1,grp2]);
        Y_pair=Y_lda(mask_pair);

        available_indices=setdiff(1:n_total,current_feat_indices);
        if isempty(available_indices), break; end

        X_pool=X_proc(mask_pair,available_indices);

        d_vals_pool=zeros(length(available_indices),1);
        for ff=1:length(available_indices)
            col_vec=X_pool(:,ff);
            vec1=col_vec(Y_pair==grp1);
            vec2=col_vec(Y_pair==grp2);

            n1=length(vec1); n2=length(vec2);
            if n1>1&&n2>1&&(var(vec1)>0||var(vec2)>0)
                m1=mean(vec1); m2=mean(vec2);
                s1=std(vec1); s2=std(vec2);
                s_pooled=sqrt(((n1-1)*s1^2+(n2-1)*s2^2)/(n1+n2-2));
                if s_pooled>0
                    d_vals_pool(ff)=abs((m1-m2)/s_pooled);
                end
            end
        end

        [~,sorted_d_idx]=sort(d_vals_pool,'descend');

        for rank_i=1:min(20,length(sorted_d_idx))
            candidate_idx=available_indices(sorted_d_idx(rank_i));
            candidate_name=all_feat_names{candidate_idx};

            test_feat_indices=[current_feat_indices;candidate_idx];
            n_test_f=length(test_feat_indices);

            X_test=X_proc(:,test_feat_indices);
            pred_labels_test=zeros(n_samples,1);

            for ii=1:n_samples
                test_idx=ii; train_idx=setdiff(1:n_samples,test_idx);
                X_tr=X_test(train_idx,:); Y_tr=Y_lda(train_idx);
                X_te=X_test(test_idx,:);

                classes_tr=unique(Y_tr);
                m_tot=mean(X_tr); Sb_sub=zeros(n_test_f); Sw_sub=zeros(n_test_f);
                for c_idx=1:length(classes_tr)
                    cc=classes_tr(c_idx);
                    xk=X_tr(Y_tr==cc,:); nk=size(xk,1);
                    if nk>0
                        mk=mean(xk);
                        Sb_sub=Sb_sub+nk*((mk-m_tot)'*(mk-m_tot));
                        if nk>1, Sw_sub=Sw_sub+(xk-mk)'*(xk-mk); end
                    end
                end

                [V_sub,D_sub]=eig(Sb_sub,Sw_sub+eye(n_test_f)*lambda_reg);
                [~,si]=sort(diag(D_sub),'descend');
                W_curr=real(V_sub(:,si)); W_curr=W_curr(:,1:min(opt_n_dims,size(W_curr,2)));

                X_tr_proj=X_tr*W_curr; X_te_proj=X_te*W_curr;
                Centroids=zeros(length(classes_tr),size(X_tr_proj,2));
                for kk=1:length(classes_tr), Centroids(kk,:)=mean(X_tr_proj(Y_tr==classes_tr(kk),:),1); end

                [~,min_k]=min(sum((Centroids-X_te_proj).^2,2));
                pred_labels_test(ii)=classes_tr(min_k);
            end

            test_min_acc=calc_min_acc(Y_lda,pred_labels_test,unique(Y_lda));
            if test_min_acc>current_min_acc
                current_min_acc=test_min_acc;
                current_feat_indices=test_feat_indices;
                pred_labels_init=pred_labels_test;
                iter_improvement_found=true;
                break;
            end
        end
        if iter_improvement_found, break; end
    end
    if ~iter_improvement_found, break; end
end

current_feat_indices=sort(current_feat_indices,'ascend');
final_feature_names=all_feat_names(current_feat_indices);
X_final_C1=X_proc(:,current_feat_indices);
opt_n_feats=length(current_feat_indices);
m_tot=mean(X_final_C1);
Sb=zeros(opt_n_feats); Sw=zeros(opt_n_feats);
for kk=1:n_classes
    xk=X_final_C1(Anova_Grps==kk,:); nk=size(xk,1);
    if nk>0
        mk=mean(xk);
        Sb=Sb+nk*((mk-m_tot)'*(mk-m_tot));
        if nk>1, Sw=Sw+(xk-mk)'*(xk-mk); end
    end
end
[V,D]=eig(Sb,Sw+lambda_reg*eye(opt_n_feats));
[eig_vals,si]=sort(diag(D),'descend');
discriminant_power=(eig_vals/sum(eig_vals))*100;
W_opt=real(V(:,si));
W_opt=W_opt(:,1:opt_n_dims);
X_proj_C1=X_final_C1*W_opt;
total_variance=sum(var(X_final_C1));
variance_explained_pca=zeros(opt_n_dims,1);
for dd=1:opt_n_dims
    proj_var=var(X_proj_C1(:,dd));
    variance_explained_pca(dd)=(proj_var/total_variance)*100;
end
Pred_Label_Final=zeros(n_samples,1);
for ii=1:n_samples
    test_idx=ii; train_idx=setdiff(1:n_samples,test_idx);
    X_tr=X_final_C1(train_idx,:); Y_tr=Y_lda(train_idx);
    X_te=X_final_C1(test_idx,:);
    classes_tr=unique(Y_tr);
    m_tot=mean(X_tr); Sb_sub=zeros(opt_n_feats); Sw_sub=zeros(opt_n_feats);
    for c_idx=1:length(classes_tr)
        cc=classes_tr(c_idx);
        xk=X_tr(Y_tr==cc,:); nk=size(xk,1);
        if nk>0
            mk=mean(xk);
            Sb_sub=Sb_sub+nk*((mk-m_tot)'*(mk-m_tot));
            if nk>1, Sw_sub=Sw_sub+(xk-mk)'*(xk-mk); end
        end
    end
    [V_sub,D_sub]=eig(Sb_sub,Sw_sub+eye(opt_n_feats)*lambda_reg);
    [~,si]=sort(diag(D_sub),'descend');
    W_sub=real(V_sub(:,si)); W_curr=W_sub(:,1:min(opt_n_dims,size(W_sub,2)));
    X_tr_proj=X_tr*W_curr; X_te_proj=X_te*W_curr;
    Centroids=zeros(length(classes_tr),size(X_tr_proj,2));
    for kk=1:length(classes_tr), Centroids(kk,:)=mean(X_tr_proj(Y_tr==classes_tr(kk),:),1); end
    [~,min_k]=min(sum((Centroids-X_te_proj).^2,2));
    Pred_Label_Final(ii)=classes_tr(min_k);
end
final_overall_acc=sum(Pred_Label_Final==Y_lda)/n_samples*100;
final_min_acc=calc_min_acc(Y_lda,Pred_Label_Final,unique(Y_lda));
figure('Name','C1 Confusion Matrix (Final)','Color','w');
custom_order={'HPC','BG','CTX','MI','BSLT','CTRL'};
detected_cats=categories(categorical(Cats));
missing_in_custom=setdiff(detected_cats,custom_order);
full_chart_order=[custom_order,missing_in_custom'];
Y_true_ord=categorical(Cats(Y_lda),full_chart_order);
Y_pred_ord=categorical(Cats(Pred_Label_Final),full_chart_order);
cm1=confusionchart(Y_true_ord,Y_pred_ord);
cm1.Title=sprintf('Cohort 1 LOOCV (Final, MinAcc: %.1f%%, Overall: %.1f%%)',final_min_acc,final_overall_acc);
cm1.RowSummary='row-normalized'; cm1.ColumnSummary='column-normalized';

perm_min_accs=zeros(n_perms_actual,1);
perm_X=X_proc(:,full_valid_indices); perm_Y=Y_lda;
perm_n_samples=n_samples; perm_n_classes=n_classes;
perm_feat_counts=feat_counts; perm_max_inj=max_injection_iters;
perm_lambda=lambda_reg; perm_opt_d=opt_n_dims;
parfor pp=1:n_perms_actual
    rng(pp);
    Y_shuffled=perm_Y(randperm(perm_n_samples));

    best_p_start_acc=-1; best_p_start_n=perm_feat_counts(1);
    best_p_start_preds=zeros(perm_n_samples,1);

    for n_f=perm_feat_counts
        X_sub=perm_X(:,1:n_f);
        totals=zeros(perm_n_classes,1);
        for kk=1:perm_n_classes, totals(kk)=sum(Y_shuffled==kk); end
        class_hits=zeros(perm_n_classes,1);
        temp_preds=zeros(perm_n_samples,1);
        for ii=1:perm_n_samples
            tr=[1:ii-1,ii+1:perm_n_samples];
            Xtr=X_sub(tr,:); Ytr=Y_shuffled(tr); Xte=X_sub(ii,:);
            mt=mean(Xtr); Sb=0; Sw=0; classes_tr=unique(Ytr);
            for c_idx=1:length(classes_tr)
                curr_c=classes_tr(c_idx); xk=Xtr(Ytr==curr_c,:); nk=size(xk,1);
                if nk>0, mk=mean(xk); Sb=Sb+nk*((mk-mt)'*(mk-mt));
                    if nk>1, Sw=Sw+(xk-mk)'*(xk-mk); end
                end
            end
            [V_s,D_s]=eig(Sb,Sw+eye(n_f)*perm_lambda);
            [~,si]=sort(diag(D_s),'descend');
            W_s=real(V_s(:,si)); W_s=W_s(:,1:min(perm_opt_d,size(W_s,2)));
            tr_p=Xtr*W_s; te_p=Xte*W_s;
            cents=zeros(length(classes_tr),size(W_s,2));
            for kk=1:length(classes_tr), cents(kk,:)=mean(tr_p(Ytr==classes_tr(kk),:),1); end
            [~,min_k]=min(sum((cents-te_p).^2,2));
            pred=classes_tr(min_k);
            temp_preds(ii)=pred;
            if pred==Y_shuffled(ii), class_hits(Y_shuffled(ii))=class_hits(Y_shuffled(ii))+1; end
        end
        curr_acc=sum(class_hits)/perm_n_samples*100;
        if curr_acc>best_p_start_acc
            best_p_start_acc=curr_acc;
            best_p_start_n=n_f;
            best_p_start_preds=temp_preds;
        end
    end

    curr_min_acc=best_p_start_acc;
    curr_preds=best_p_start_preds;
    curr_feats=1:best_p_start_n;
    avail_feats=setdiff(1:size(perm_X,2),curr_feats);

    for iter=1:perm_max_inj
        if curr_min_acc>=100, break; end
        CM=confusionmat(Y_shuffled,curr_preds); CM(1:perm_n_classes+1:end)=0;
        [r,c,v]=find(CM);
        if isempty(v), break; end
        [~,si]=sort(v,'descend');
        grp1=r(si(1)); grp2=c(si(1));

        mask=ismember(Y_shuffled,[grp1,grp2]); Y_pair=Y_shuffled(mask);
        X_pool=perm_X(mask,avail_feats);

        d_vals=zeros(length(avail_feats),1);
        for ff=1:length(avail_feats)
            cv=X_pool(:,ff);
            vec1=cv(Y_pair==grp1); vec2=cv(Y_pair==grp2);
            n1=length(vec1); n2=length(vec2);
            if n1>1&&n2>1&&(var(vec1)>0||var(vec2)>0)
                s_pool=sqrt(((n1-1)*var(vec1)+(n2-1)*var(vec2))/(n1+n2-2));
                if s_pool>0, d_vals(ff)=abs((mean(vec1)-mean(vec2))/s_pool); end
            end
        end
        [~,sort_p]=sort(d_vals,'descend');

        imp=false;
        for kk=1:min(3,length(sort_p))
            cand=avail_feats(sort_p(kk));
            test_fs=[curr_feats,cand];
            X_t=perm_X(:,test_fs);
            hits_t=zeros(perm_n_classes,1);
            totals_t=zeros(perm_n_classes,1); for z=1:perm_n_classes, totals_t(z)=sum(Y_shuffled==z); end
            temp_preds_inj=zeros(perm_n_samples,1);
            for ii=1:perm_n_samples
                tr=[1:ii-1,ii+1:perm_n_samples];
                Xtr=X_t(tr,:); Ytr=Y_shuffled(tr); Xte=X_t(ii,:);
                mt=mean(Xtr); Sb=0; Sw=0; classes_tr=unique(Ytr);
                for c_idx=1:length(classes_tr)
                    curr_c=classes_tr(c_idx); xk=Xtr(Ytr==curr_c,:); nk=size(xk,1);
                    if nk>0, mk=mean(xk); Sb=Sb+nk*((mk-mt)'*(mk-mt));
                        if nk>1, Sw=Sw+(xk-mk)'*(xk-mk); end
                    end
                end
                [V_s,D_s]=eig(Sb,Sw+eye(length(test_fs))*perm_lambda);
                [~,si]=sort(diag(D_s),'descend');
                W_s=real(V_s(:,si)); W_s=W_s(:,1:min(perm_opt_d,size(W_s,2)));
                tr_p=Xtr*W_s; te_p=Xte*W_s;
                cents=zeros(length(classes_tr),size(W_s,2));
                for z=1:length(classes_tr), cents(z,:)=mean(tr_p(Ytr==classes_tr(z),:),1); end
                [~,min_k]=min(sum((cents-te_p).^2,2));
                pred=classes_tr(min_k); temp_preds_inj(ii)=pred;
                if pred==Y_shuffled(ii), hits_t(Y_shuffled(ii))=hits_t(Y_shuffled(ii))+1; end
            end
            new_min=min((hits_t./totals_t)*100);
            if new_min>curr_min_acc
                curr_min_acc=new_min;
                curr_preds=temp_preds_inj;
                curr_feats=test_fs;
                avail_feats(sort_p(kk))=[];
                imp=true;
                break;
            end
        end
        if ~imp, break; end
    end

    perm_min_accs(pp)=sum(curr_preds==Y_shuffled)/perm_n_samples*100;
end
perm_p_val=(sum(perm_min_accs>=final_overall_acc)+1)/(n_perms_actual+1);
figure('Name','Global Permutation Test','Color','w');
histogram(perm_min_accs,30,'FaceColor',[0.7 0.7 0.7],'EdgeColor','none');
hold on;
xline(final_overall_acc,'r','LineWidth',2);
title(sprintf('Global Permutation Test (N=%d, p=%.4f, Mean=%.1f%%, Std=%.2f)',n_perms_actual,perm_p_val,mean(perm_min_accs),std(perm_min_accs)));
xlabel('Average Accuracy (%)'); ylabel('Frequency');
legend({'Null Distribution (Shuffled)','Observed Model'},'Location','best');
hold off;

rawT_c2=readtable(csvFilePath);
temp_sex_c2=upper(strtrim(string(rawT_c2.Sex)));
if strcmp(sex_interest,'M'), rawT_c2=rawT_c2(startsWith(temp_sex_c2,'M'),:);
elseif strcmp(sex_interest,'F'), rawT_c2=rawT_c2(startsWith(temp_sex_c2,'F'),:); end
c2_mask=endsWith(string(rawT_c2.Group),'_2');
X_proj_C2=[];
if sum(c2_mask)~=0
    rawT_c2=rawT_c2(c2_mask,:);
    rawT_c2=rawT_c2(~contains(string(rawT_c2.Group),'xxx','IgnoreCase',true),:);

    [found,c2_col_idx]=ismember(all_feat_names,rawT_c2.Properties.VariableNames);
    if ~all(found)
        error('Some features used in C1 are missing in C2 data!');
    end

    X_c2_raw=table2array(rawT_c2(:,c2_col_idx));

    Raw_Sex_C2=cellstr(rawT_c2.Sex); Raw_Sex_C2(cellfun(@isempty,Raw_Sex_C2))={'Unknown'};
    X_c2_robust=zeros(size(X_c2_raw));

    for ss=1:length(unique_sexes)
        curr_sex=unique_sexes{ss};
        sex_mask=strcmp(Raw_Sex_C2,curr_sex);
        if sum(sex_mask)==0, continue; end

        X_sub=X_c2_raw(sex_mask,:);
        M_sub=mean(X_sub,1,'omitnan');
        D_sub=std(X_sub,0,1,'omitnan'); D_sub(D_sub==0)=1;

        X_norm=(X_sub-M_sub)./D_sub;
        X_norm(X_norm>windsor_cap)=windsor_cap;
        X_norm(X_norm<-windsor_cap)=-windsor_cap;
        X_c2_robust(sex_mask,:)=X_norm;
    end
    if any(isnan(X_c2_robust(:))), X_c2_robust=fillmissing(X_c2_robust,'nearest'); end

    Cov_Source=cov(X_proc)+eye(size(X_proc,2));
    Cov_Target=cov(X_c2_robust)+eye(size(X_c2_robust,2));
    A_source=sqrtm(Cov_Source);
    A_target=sqrtm(Cov_Target);

    X_c2_coral=X_c2_robust*inv(A_target)*A_source;

    X_c2_input=X_c2_coral(:,current_feat_indices);
    X_proj_C2=X_c2_input*W_opt;
end

if ~isempty(X_proj_C2), Combined_Proj=[X_proj_C1;X_proj_C2]; else, Combined_Proj=X_proj_C1; end
Label_C1=regexprep(cellstr(rawT.Group),'_[0-9]+$','');
if ~isempty(X_proj_C2)
    Label_C2=regexprep(cellstr(rawT_c2.Group),'_[0-9]+$','');
    Combined_Grps=[Label_C1;Label_C2];
    Combined_Cohort=[ones(height(rawT),1);2*ones(height(rawT_c2),1)];

    IDs_C2=string(table2cell(rawT_c2(:,1)));
    Combined_IDs=[Raw_IDs;IDs_C2];
else
    Combined_Grps=Label_C1; Combined_Cohort=ones(height(rawT),1);
    Combined_IDs=Raw_IDs;
end
custom_order={'HPC','BG','CTX','MI','BSLT','CTRL'};
detected_groups=unique(Combined_Grps);
missing_in_custom=setdiff(detected_groups,custom_order);
full_order=[custom_order,missing_in_custom'];
Combined_Grps_Cat=categorical(Combined_Grps,full_order,'Ordinal',true);
u_grps=intersect(full_order,detected_groups,'stable');

figure('Name','Final Model Loadings','Color','w','Position',[100 100 1000 500]);
for ld_idx=1:opt_n_dims
    subplot(1,opt_n_dims,ld_idx);
    weights=W_opt(:,ld_idx);
    barh(weights,'FaceColor',[0.2 0.4 0.8]);
    yticks(1:length(weights)); yticklabels(final_feature_names);
    set(gca,'YDir','reverse');
    title(sprintf('LD %d\nDiscrim: %.1f%%\nTotal Var: %.1f%%', ...
        ld_idx,discriminant_power(ld_idx),variance_explained_pca(ld_idx)));
    grid on; xline(0,'k-');
end
sgtitle('Representative Feature Importance (Fixed Order)');

Meta_Table=table(Combined_Grps_Cat,Combined_Cohort,Combined_Proj,Combined_IDs, ...
                 'VariableNames',{'Group','Cohort','Proj','ID'});

Meta_Sorted=table();
groups_in_order=categories(Combined_Grps_Cat);
groups_in_order=groups_in_order(ismember(groups_in_order,unique(cellstr(Meta_Table.Group))));
for g_i=1:length(groups_in_order)
    curr_grp_name=groups_in_order{g_i};

    mask_c1=strcmp(cellstr(Meta_Table.Group),curr_grp_name)&(Meta_Table.Cohort==1);
    T_c1=Meta_Table(mask_c1,:);

    mask_c2=strcmp(cellstr(Meta_Table.Group),curr_grp_name)&(Meta_Table.Cohort==2);
    T_c2=Meta_Table(mask_c2,:);

    mean_c1=[]; mean_c2=[];
    if height(T_c1)>0, mean_c1=mean(T_c1.Proj,1); end
    if height(T_c2)>0, mean_c2=mean(T_c2.Proj,1); end

    if height(T_c1)>0
        if ~isempty(mean_c2)
            d1=pdist2(T_c1.Proj,mean_c2,'cosine');
            [~,idx1]=sort(d1,'descend');
            T_c1=T_c1(idx1,:);
        end
        Meta_Sorted=[Meta_Sorted;T_c1];
    end

    if height(T_c2)>0
        if ~isempty(mean_c1)
            d2=pdist2(T_c2.Proj,mean_c1,'cosine');
            [~,idx2]=sort(d2,'ascend');
            T_c2=T_c2(idx2,:);
        end
        Meta_Sorted=[Meta_Sorted;T_c2];
    end
end
Meta_Sorted.Group=cellstr(Meta_Sorted.Group);
Heatmap_Data=Meta_Sorted.Proj';
n_samples_plot=size(Heatmap_Data,2);
figure('Name','LD Scores Heatmap (Transposed)','Color','w');
imagesc(Heatmap_Data); colormap(redblue); colorbar;
title('Linear Discriminant (LD) Scores per Sample');
ylabel('LD Components'); xlabel('Samples (Sorted)');
set(gca,'YTick',1:opt_n_dims,'YTickLabel',strcat('LD',string(1:opt_n_dims)));
set(gca,'XTick',1:n_samples_plot,'XTickLabel',Meta_Sorted.ID);
set(gca,'XTickLabelRotation',90,'FontSize',8);
hold on;
grp_changes=[0;find(~strcmp(Meta_Sorted.Group(1:end-1),Meta_Sorted.Group(2:end)));height(Meta_Sorted)];
is_same_group=strcmp(Meta_Sorted.Group(1:end-1),Meta_Sorted.Group(2:end));
is_cohort_change=Meta_Sorted.Cohort(1:end-1)~=Meta_Sorted.Cohort(2:end);
c1_c2_boundaries=find(is_same_group&is_cohort_change);
for bb=2:length(grp_changes)-1, xline(grp_changes(bb)+0.5,'k-','LineWidth',2); end
for cc=1:length(c1_c2_boundaries), xline(c1_c2_boundaries(cc)+0.5,'k:','LineWidth',2.0); end
for ii=1:length(grp_changes)-1
    start_pos=grp_changes(ii)+1; end_pos=grp_changes(ii+1);
    center_pos=(start_pos+end_pos)/2;
    text(center_pos,opt_n_dims+0.7,Meta_Sorted.Group{start_pos},'Rotation',45, ...
        'HorizontalAlignment','right','FontSize',10,'Interpreter','none','FontWeight','bold');
end
hold off;

figure('Name','Pairwise Cosine Similarity','Color','w');
D_cos_sorted=pdist(Meta_Sorted.Proj,'cosine');
Sim_Mat=1-squareform(D_cos_sorted);
imagesc(Sim_Mat); colormap(redblue); colorbar; caxis([-1 1]);
title('Pairwise Cosine Similarity (Sorted by Group -> Cohort)');
xlabel('Samples'); ylabel('Samples');
hold on;
for bb=2:length(grp_changes)-1
    loc=grp_changes(bb)+0.5; xline(loc,'k-','LineWidth',1.5); yline(loc,'k-','LineWidth',1.5);
end
for cc=1:length(c1_c2_boundaries)
    loc=c1_c2_boundaries(cc)+0.5; xline(loc,'k:','LineWidth',2.0); yline(loc,'k:','LineWidth',2.0);
end
hold off;

function [F,p]=fast_anova_f(x,group,n_groups,n_samples)
    grand_mean=sum(x)/n_samples;
    SSB=0; SST=sum((x-grand_mean).^2);
    for gg=1:n_groups
        mask=(group==gg); n_k=sum(mask);
        if n_k>0, SSB=SSB+n_k*(sum(x(mask))/n_k-grand_mean)^2; end
    end
    SSW=SST-SSB;
    MSB=SSB/(n_groups-1); MSE=SSW/(n_samples-n_groups);
    if MSE==0, F=0; else, F=MSB/MSE; end
    p=0;
end
function c=redblue(m)
    if nargin<1, m=size(get(gcf,'colormap'),1); end
    if (mod(m,2)==0), m1=m*0.5; r=(0:m1-1)'/max(m1-1,1); g=r; r=[r;ones(m1,1)]; g=[g;flipud(g)]; b=flipud(r);
    else, m1=floor(m*0.5); r=(0:m1-1)'/max(m1,1); g=r; r=[r;ones(m1+1,1)]; g=[g;1;flipud(g)]; b=flipud(r); end
    c=[r g b];
end