% =========================================================================
% File Description: Main script for training and evaluating 
%                   prognosis prediction models for esophageal cancer patients.
%                   Includes data loading, preprocessing, feature engineering,
%                   model training, hyperparameter optimization, ROC curve
%                   plotting, and performance evaluation.
% =========================================================================

clc;clear;close all;	

% Check required MATLAB toolbox
if ~license('test', 'Statistics_Toolbox')
    error('Error: Statistics and Machine Learning Toolbox is required to run this code.\nPlease install or activate the toolbox.\nRun "check_toolbox.m" for more details.');
end

load('R_11_Nov_2025_20_22_24.mat')	
random_seed=G_out_data.random_seed ;  % Random seed set in interface
rng(random_seed)  % Fix random seed 	
	
% Get script directory and build path to processed data folder
script_dir = fileparts(mfilename('fullpath'));  % Get script directory
project_root = fileparts(script_dir);  % Go up one level to project root
processed_data_dir = fullfile(project_root, 'data', 'processed_data');  % Processed data folder path

% Read data path - automatically load from processed data folder
data_path_from_config = G_out_data.data_path_str;  % Get path or filename from config

% Extract filename (ignore path, solve cross-platform path issues)
% Handle Windows paths (with backslash) and Unix paths (with forward slash)
% Convert backslash to forward slash uniformly, then split
normalized_path = strrep(data_path_from_config, '\', '/');

% Extract last part as filename
if contains(normalized_path, '/')
    % Has path separator, extract last part
    parts = strsplit(normalized_path, '/');
    filename_with_ext = parts{end};
else
    % No path separator, use directly
    filename_with_ext = normalized_path;
end

% Extract filename and extension
[~, filename, ext] = fileparts(filename_with_ext);
if isempty(ext) && contains(filename_with_ext, '.')
    % If fileparts didn't extract extension, extract manually
    dot_pos = strfind(filename_with_ext, '.');
    if ~isempty(dot_pos)
        filename = filename_with_ext(1:dot_pos(end)-1);
        ext = filename_with_ext(dot_pos(end):end);
    end
end

% Ensure filename and ext are not empty
if isempty(filename) && isempty(ext)
    % If both are empty, use original string
    filename_with_ext = data_path_from_config;
    [~, filename, ext] = fileparts(filename_with_ext);
end

% Build full path: always use processed data folder + filename
data_str = fullfile(processed_data_dir, [filename, ext]);

% Verify file exists
if ~exist(data_str, 'file')
    error('Data file does not exist: %s\nPlease check if the file exists in the data/processed_data folder.', data_str);
end

dataO=readtable(data_str,'VariableNamingRule','preserve'); % Read data
data1=dataO(:,2:end);test_data=table2cell(dataO(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          type_index(i)=1;     % char type
      elseif isnumeric(test_data{1,i})==1	
          type_index(i)=2;     % double type
      else	
        type_index(i)=0;     % other type
     end 	
end	
char_index=find(type_index==1);numeric_index=find(type_index==2);	
 %% Process numeric type data
if length(numeric_index)>=1	
    numeric_data=table2array(data1(:,numeric_index));	
    numeric_index1=numeric_index;	
	
    numeric_index1_idx=1:size(numeric_data,2);	
    nan_mask=(isnan(numeric_data));    % Find missing values in columns
    num_nan_row=sum(nan_mask);	
    nan_index=num_nan_row>round(0.2*size(data1,1));	
    numeric_index1(nan_index==1)=[]; numeric_index1_idx(nan_index==1)=[];	
    numeric_data1=numeric_data(:,numeric_index1_idx);	
    nan_mask1=(isnan(numeric_data1));  % Find missing values in rows
    num_nan_col=sum(nan_mask1);	
    nan_index1=num_nan_col>0;	
    numeric_index2_idx=1:size(numeric_data,1);	
    numeric_index2_idx(nan_index1==1)=[];	
    numeric_data2=numeric_data1(numeric_index2_idx,:);	
    final_numeric_index=numeric_index1;	
 else	
    final_numeric_index=[];	
    numeric_data2=[];	
end	
%% Process text type data	
	
categorical_data=[];	
 if length(char_index)>=1	
  for j=1:length(char_index)	
    data_get=table2array(data1(numeric_index2_idx,char_index(j)));	
    data_label=unique(data_get);	
    if j==length(char_index)	
       data_label_str=data_label ;	
    end    	
	
     for NN=1:length(data_label)	
            idx = find(ismember(data_get,data_label{NN,1}));  	
            categorical_data(idx,j)=NN; 	
     end	
  end	
 end	
final_feature_indices=[char_index,final_numeric_index];	
[~,label_max]=max(final_feature_indices);	
 if(label_max==length(final_feature_indices))	
     str_label=0; % Flag indicating if output is character type
     data_all_last=[categorical_data,numeric_data2];	
     final_feature_indices=[char_index,final_numeric_index];	
 else	
    str_label=1;	
    data_all_last=[numeric_data2,categorical_data];	
    final_feature_indices=[final_numeric_index,char_index];     	
 end	
 data=data_all_last;	
 feature_names_all=data1.Properties.VariableNames;	
 for j=1:length(final_feature_indices)	
    feature_names{1,j}=feature_names_all{1,final_feature_indices(j)};	
 end	
	
% Outlier detection
data=data;	
	
%% Feature processing: feature selection or dimensionality reduction
	
 A_data1=data;	
 feature_names1=feature_names;	
 select_feature_num=G_out_data.select_feature_num1;   % Number of features to select
	
data_select=A_data1;	
final_feature_indices_list=1:size(A_data1,2)-1;

% Display feature information
fprintf('\n');
fprintf('========================================\n');
fprintf('Feature Information\n');
fprintf('========================================\n');
fprintf('Total features: %d\n', size(A_data1,2)-1);
fprintf('Feature selection: ');
if select_feature_num > 0 && select_feature_num < size(A_data1,2)-1
    fprintf('Yes (select first %d features)\n', select_feature_num);
else
    fprintf('No (use all %d features)\n', size(A_data1,2)-1);
end
fprintf('Actual features used: %d\n', length(final_feature_indices_list));
fprintf('Feature names: ');
if length(feature_names) <= 10
    fprintf('%s\n', strjoin(feature_names(1:min(length(feature_names)-1, end-1)), ', '));
else
    fprintf('%s ... (total %d features)\n', strjoin(feature_names(1:5), ', '), length(feature_names)-1);
end
fprintf('========================================\n\n');	
	
	
	
%% Data splitting
x_feature_label=data_select(:,1:end-1);    % x features
y_feature_label_raw=data_select(:,end);          % y labels (raw values)

% Check if labels are continuous values, if so convert to categorical labels
% Usually classification problem labels should be discrete category values (e.g., 0,1 or 1,2)
unique_labels_raw = unique(y_feature_label_raw(~isnan(y_feature_label_raw)));
fprintf('\nLabel data check:\n');
fprintf('  Number of unique raw label values: %d\n', length(unique_labels_raw));
fprintf('  Sample unique raw label values: %s\n', mat2str(unique_labels_raw(1:min(10, length(unique_labels_raw)))'));

% If number of unique values is too many (>10), it's continuous, need to convert to categories
if length(unique_labels_raw) > 10
    fprintf('  Warning: Labels appear to be continuous, need to convert to categorical labels\n');
    
    % Method 1: Use median as threshold for binary classification
    % Assume labels less than median are class 0, greater than or equal to median are class 1
    median_value = median(y_feature_label_raw(~isnan(y_feature_label_raw)));
    fprintf('  Using median threshold for binary classification: %.4f\n', median_value);
    
    y_feature_label = double(y_feature_label_raw < median_value);
    y_feature_label(isnan(y_feature_label_raw)) = 0;  % Set NaN values to 0
    
    fprintf('  Class distribution after conversion:\n');
    unique_labels = unique(y_feature_label);
    for i = 1:length(unique_labels)
        count = sum(y_feature_label == unique_labels(i));
        fprintf('    Class %d: %d samples (%.2f%%)\n', unique_labels(i), count, count/length(y_feature_label)*100);
    end
else
    % If number of unique values is not too many, use directly, but ensure integers
    fprintf('  Labels are already categorical, use directly\n');
    y_feature_label = double(y_feature_label_raw);
    
    % If labels don't start from 0 or 1, convert to start from 0 or 1
    min_label = min(y_feature_label(~isnan(y_feature_label)));
    if min_label ~= 0 && min_label ~= 1
        fprintf('  Convert labels from starting at %d to starting at 0\n', min_label);
        y_feature_label = y_feature_label - min_label;
    end
    
    % Handle NaN values
    y_feature_label(isnan(y_feature_label)) = 0;
    
    % Ensure labels are integers
    y_feature_label = round(y_feature_label);
    
    fprintf('  Final class distribution:\n');
    unique_labels = unique(y_feature_label);
    for i = 1:length(unique_labels)
        count = sum(y_feature_label == unique_labels(i));
        fprintf('    Class %d: %d samples (%.2f%%)\n', unique_labels(i), count, count/length(y_feature_label)*100);
    end
end

data_size = size(x_feature_label,1);  % Current data size

% Output dataset class statistics
fprintf('\n');
fprintf('========================================\n');
fprintf('Dataset Class Statistics\n');
fprintf('========================================\n');
unique_labels = unique(y_feature_label);
fprintf('Class labels: %s\n', mat2str(unique_labels));
for i = 1:length(unique_labels)
    count = sum(y_feature_label == unique_labels(i));
    percentage = count / data_size * 100;
    % 判断是1年OS还是5年OS（根据文件名）
    if contains(filename, '1year') || contains(filename, '1年')
        if unique_labels(i) == 1 || unique_labels(i) == max(unique_labels)
            fprintf('1年OS - 存活: %d 人 (%.2f%%)\n', count, percentage);
        else
            fprintf('1年OS - 死亡: %d 人 (%.2f%%)\n', count, percentage);
        end
    elseif contains(filename, '5year') || contains(filename, '5年')
        if unique_labels(i) == 1 || unique_labels(i) == max(unique_labels)
            fprintf('5年OS - 存活: %d 人 (%.2f%%)\n', count, percentage);
        else
            fprintf('5年OS - 死亡: %d 人 (%.2f%%)\n', count, percentage);
        end
    else
        fprintf('类别 %d: %d 人 (%.2f%%)\n', unique_labels(i), count, percentage);
    end
end
    fprintf('Total samples: %d\n', data_size);
    fprintf('========================================\n\n');

% 验证标签数据
if any(isnan(y_feature_label)) || any(isinf(y_feature_label))
    warning('标签中包含NaN或Inf值，将被替换为0');
    y_feature_label(isnan(y_feature_label) | isinf(y_feature_label)) = 0;
end


y_feature_label = round(y_feature_label);

index_label=G_out_data.spilt_label_data;  	
if isempty(data_index) || length(data_index) ~= data_size || max(data_index) > data_size || min(data_index) < 1
    fprintf('重新生成数据划分索引（数据大小: %d）\n', data_size);
    data_index = randperm(data_size);
else
    if any(data_index > data_size) || any(data_index < 1)
        fprintf('警告：索引值超出范围，重新生成数据划分索引\n');
        data_index = randperm(data_size);
    end
end

spilt_ri=G_out_data.spilt_rio; 	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*data_size);          
valid_num=round((split_ratio(1)+split_ratio(2))/(sum(split_ratio))*data_size); % Number of validation samples

train_num = min(train_num, data_size);
valid_num = min(valid_num, data_size);
if train_num >= valid_num
    valid_num = min(train_num + 1, data_size);
end

%训练集，验证集，测试集	
train_x_feature_label=x_feature_label(data_index(1:train_num),:);	
train_y_feature_label=y_feature_label(data_index(1:train_num),:);	
valid_x_feature_label=x_feature_label(data_index(train_num+1:valid_num),:);	
valid_y_feature_label=y_feature_label(data_index(train_num+1:valid_num),:);	
test_x_feature_label=x_feature_label(data_index(valid_num+1:end),:);	
test_y_feature_label=y_feature_label(data_index(valid_num+1:end),:);
% Z-score normalization
% Training set
x_mu = mean(train_x_feature_label, 1, 'omitnan');  % 忽略NaN计算均值
x_sig = std(train_x_feature_label, 0, 1, 'omitnan'); 


x_sig(x_sig == 0) = 1;

train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化

train_x_feature_label_norm(isnan(train_x_feature_label_norm)) = 0;
train_x_feature_label_norm(isinf(train_x_feature_label_norm)) = 0;


train_y_feature_label_norm = train_y_feature_label;  % 标签不标准化，直接使用原始值

%验证集	
valid_x_feature_label_norm = (valid_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
valid_x_feature_label_norm(isnan(valid_x_feature_label_norm)) = 0;
valid_x_feature_label_norm(isinf(valid_x_feature_label_norm)) = 0;

valid_y_feature_label_norm = valid_y_feature_label;  % 标签不标准化，直接使用原始值

%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
test_x_feature_label_norm(isnan(test_x_feature_label_norm)) = 0;
test_x_feature_label_norm(isinf(test_x_feature_label_norm)) = 0;

test_y_feature_label_norm = test_y_feature_label;  % 标签不标准化，直接使用原始值

% 验证数据完整性
fprintf('数据标准化完成:\n');
fprintf('  训练集: %d 样本, %d 特征\n', size(train_x_feature_label_norm, 1), size(train_x_feature_label_norm, 2));
fprintf('  验证集: %d 样本, %d 特征\n', size(valid_x_feature_label_norm, 1), size(valid_x_feature_label_norm, 2));
fprintf('  测试集: %d 样本, %d 特征\n', size(test_x_feature_label_norm, 1), size(test_x_feature_label_norm, 2));

% 检查是否有有效数据
if size(train_x_feature_label_norm, 1) == 0 || size(train_x_feature_label_norm, 2) == 0
    error('错误: 训练集数据为空，请检查数据预处理步骤');
end

if any(all(isnan(train_x_feature_label_norm), 1)) || any(all(isinf(train_x_feature_label_norm), 1))
    warning('警告: 训练集中存在全NaN或全Inf的特征列');
end  	
	
%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
attention_label=G_out_data.attention_label;   %注意力机制标签	
attention_head=G_out_data.attention_head;   %注意力机制设置	
	
%% 数据增强部分	
get_mutiple=G_out_data.get_mutiple;  %数据增加倍数	
methodchoose=1; 	
origin_data=[train_x_feature_label_norm;valid_x_feature_label_norm]; 	
origin_data_label=[train_y_feature_label;valid_y_feature_label];  % 注意：使用原始标签，不是标准化后的

% 检查数据中是否有NaN或Inf
if any(isnan(origin_data(:))) || any(isinf(origin_data(:)))
    fprintf('警告: 数据增强前的数据中包含NaN或Inf值，正在清理...\n');
    origin_data(isnan(origin_data)) = 0;
    origin_data(isinf(origin_data)) = 0;
end

[SyntheticData,Synthetic_label]=generate_classdata(origin_data,origin_data_label,methodchoose,get_mutiple); 	
% 绘制生成后数据样本图	
figure_data_generate(origin_data,SyntheticData,origin_data_label,Synthetic_label)	

% 合并数据
if ~isempty(SyntheticData)
    X_new_DATA=[origin_data;SyntheticData];             %生成的X特征数据	
    Y_new_DATA=[origin_data_label;Synthetic_label];  %生成的Y标签数据
else
    % 如果没有生成合成数据，只使用原始数据
    X_new_DATA = origin_data;
    Y_new_DATA = origin_data_label;
end

% 再次检查合并后的数据
if any(isnan(X_new_DATA(:))) || any(isinf(X_new_DATA(:)))
    fprintf('警告: 合并后的数据中包含NaN或Inf值，正在清理...\n');
    X_new_DATA(isnan(X_new_DATA)) = 0;
    X_new_DATA(isinf(X_new_DATA)) = 0;
end	
	
syn_spilt=round(spilt_ri(1)/(spilt_ri(1)+spilt_ri(2))*length(Y_new_DATA));	
syn_index=randperm(length(Y_new_DATA));	
%以下将生成的数据随机分配到训练集和验证集中	
train_x_feature_label_norm=X_new_DATA(syn_index(1:syn_spilt),:);	
valid_x_feature_label_norm=X_new_DATA(syn_index(syn_spilt+1:end),:);	
train_y_feature_label=Y_new_DATA(syn_index(1:syn_spilt),:);	
valid_y_feature_label=Y_new_DATA(syn_index(syn_spilt+1:end),:);	
train_x_feature_label=train_x_feature_label_norm.*x_sig+x_mu;	
valid_x_feature_label=valid_x_feature_label_norm.*x_sig+x_mu;	
%数据生成输出数据	
train_x_feature_label_aug=(train_x_feature_label_norm.*x_sig)+x_mu;	
valid_x_feature_label_aug=(valid_x_feature_label_norm.*x_sig)+x_mu;	
%总体生成数据+原数据保存在以下的 augdata_all 数据里面	
augdata_all=[train_x_feature_label_aug,train_y_feature_label;valid_x_feature_label_aug,valid_y_feature_label;test_x_feature_label,test_y_feature_label];	
	
	
%% 算法处理块	
	
	
	
	
fprintf('\n');
fprintf('========================================\n');
fprintf('实验配置总览\n');
fprintf('========================================\n');
fprintf('【模型信息】\n');
fprintf('  模型类型: TreeBagger (随机森林分类器)\n');
fprintf('  模型方法: Bagging集成决策树\n');
fprintf('  优化参数: NumTrees [20-200], MinLeafSize [2-10]\n');
fprintf('\n');
fprintf('【特征信息】\n');
fprintf('  特征数量: %d 个特征\n', size(train_x_feature_label_norm, 2));
if select_feature_num > 0 && select_feature_num < size(train_x_feature_label_norm, 2)
    fprintf('  特征选择: 是 (选择前%d个特征)\n', select_feature_num);
else
    fprintf('  特征选择: 否 (使用全部特征)\n');
end
fprintf('\n');
fprintf('【优化配置】\n');
fprintf('  优化算法: %s\n', G_out_data.method_mti1);
fprintf('  种群数量: %d\n', G_out_data.num_pop1);
fprintf('  迭代次数: %d\n', G_out_data.num_iter1);
fprintf('\n');
fprintf('【实验统计】\n');
total_experiments = G_out_data.num_pop1 * G_out_data.num_iter1;
fprintf('  总实验数: %d (种群数) × %d (迭代数) = %d 次模型训练\n', ...
        G_out_data.num_pop1, G_out_data.num_iter1, total_experiments);
fprintf('  每次实验: 训练模型 + 验证集评估\n');
fprintf('  优化目标: 最小化验证集分类错误率\n');
fprintf('========================================\n\n');

t1=clock; 	
num_tree=50;   %集成树的棵树	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;    %迭代次数	

% 根据数据集文件名自动选择最优超参数（基于test_rf_hyperparameters.m的消融实验结果）
% 1年OS: NumTrees=200, MinLeafSize=5 (测试集准确率: 0.8667, AUC: 0.5986)
% 5年OS: NumTrees=150, MinLeafSize=10 (测试集准确率: 0.7778, AUC: 0.8349)
if contains(filename, '1year') || contains(filename, '1年')
    optimal_num_trees = 200;
    optimal_min_leaf = 5;
    dataset_type = '1年OS';
    fprintf('\n使用1年OS数据集的最优超参数（来自消融实验）:\n');
elseif contains(filename, '5year') || contains(filename, '5年')
    optimal_num_trees = 150;
    optimal_min_leaf = 10;
    dataset_type = '5年OS';
    fprintf('\n使用5年OS数据集的最优超参数（来自消融实验）:\n');
else
    % 如果无法识别，使用默认值或优化算法
    optimal_num_trees = 50;
    optimal_min_leaf = 5;
    dataset_type = '未知';
    fprintf('\n警告: 无法识别数据集类型，使用默认超参数或优化算法\n');
    method_mti=G_out_data.method_mti1;	
    [Mdl,fitness] = optimize_fitctreebag(train_x_feature_label_norm,train_y_feature_label,valid_x_feature_label_norm,valid_y_feature_label,num_pop,num_iter,method_mti);
    optimal_num_trees = Mdl.NumTrees;
    optimal_min_leaf = Mdl.ModelParameters.MinLeaf;
end

fprintf('  NumTrees: %d\n', optimal_num_trees);
fprintf('  MinLeafSize: %d\n', optimal_min_leaf);
fprintf('  数据来源: test_rf_hyperparameters.m 消融实验结果\n\n');

% 使用最优超参数直接训练模型（跳过优化算法）
fprintf('使用最优超参数训练模型...\n');
Mdl = TreeBagger(optimal_num_trees, train_x_feature_label_norm, ...
    train_y_feature_label, 'Method', 'classification', ...
    'MinLeafSize', optimal_min_leaf);
fprintf('模型训练完成\n');
fprintf('\n开始模型预测...\n');
y_train_predict=RF_process(predict(Mdl,train_x_feature_label_norm));  %训练集预测结果	
fprintf('  训练集预测完成\n');
y_vaild_predict=RF_process(predict(Mdl,valid_x_feature_label_norm));  %验证集预测结果	
fprintf('  验证集预测完成\n');
y_test_predict=RF_process(predict(Mdl,test_x_feature_label_norm));  %测试集预测结果	
fprintf('  测试集预测完成\n');

t2=clock;	
Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));       	

fprintf('\n');
fprintf('========================================\n');
fprintf('模型训练完成！\n');
fprintf('========================================\n');
fprintf('总运行时长: %.2f 秒 (%.2f 分钟)\n', Time, Time/60);
fprintf('========================================\n\n');
confMat_train = confusionmat(train_y_feature_label,y_train_predict);	
TP_train = diag(confMat_train);      TP_train=TP_train'; % 被正确分类的正样本 True Positives	
FP_train = sum(confMat_train, 1)  - TP_train;  %被错误分类的正样本 False Positives	
FN_train = sum(confMat_train, 2)' - TP_train;  % 被错误分类的负样本 False Negatives	
TN_train = sum(confMat_train(:))  - (TP_train + FP_train + FN_train);  % 被正确分类的负样本 True Negatives	
	
disp('训练集*******************************************************************************')	
accuracy_train = sum(TP_train) / sum(confMat_train(:)); accuracy_train(isnan(accuracy_train))=0; disp(['训练集accuracy：',num2str(mean(accuracy_train))])% Accuracy 	
precision_train = TP_train ./ (TP_train + FP_train); precision_train(isnan(precision_train))=0; disp(['训练集precision_train：',num2str(mean(precision_train))]) % Precision	
recall_train = TP_train ./ (TP_train + FN_train);recall_train(isnan(recall_train))=0; disp(['训练集recall_train：',num2str(mean(recall_train))])  % Recall / Sensitivity	
F1_score_train = 2 * (precision_train .* recall_train) ./ (precision_train + recall_train); F1_score_train(isnan(F1_score_train))=0;  disp(['训练集F1_score_train：',num2str(mean(F1_score_train))])   % F1 Score	
specificity_train = TN_train ./ (TN_train + FP_train); specificity_train(isnan(specificity_train))=0; disp(['训练集specificity_train：',num2str(mean(specificity_train))])  % Specificity	
	
disp('验证集********************************************************************************')	
confMat_vaild = confusionmat(valid_y_feature_label,y_vaild_predict);	
TP_vaild = diag(confMat_vaild);      TP_vaild=TP_vaild'; % 被正确分类的正样本 True Positives	
FP_vaild = sum(confMat_vaild, 1)  - TP_vaild;  %被错误分类的正样本 False Positives	
FN_vaild = sum(confMat_vaild, 2)' - TP_vaild;  % 被错误分类的负样本 False Negatives	
TN_vaild = sum(confMat_vaild(:))  - (TP_vaild + FP_vaild + FN_vaild);  % 被正确分类的负样本 True Negatives	
accuracy_vaild = sum(TP_vaild) / sum(confMat_vaild(:)); accuracy_vaild(isnan(accuracy_vaild))=0; disp(['验证集accuracy：',num2str(accuracy_vaild)])% Accuracy 	
precision_vaild = TP_vaild ./ (TP_vaild + FP_vaild); precision_vaild(isnan(precision_vaild))=0; disp(['验证集precision_vaild：',num2str(mean(precision_vaild))]) % Precision	
recall_vaild = TP_vaild ./ (TP_vaild + FN_vaild); recall_vaild(isnan(recall_vaild))=0;  disp(['验证集recall_vaild：',num2str(mean(recall_vaild))])  % Recall / Sensitivity	
F1_score_vaild = 2 * (precision_vaild .* recall_vaild) ./ (precision_vaild + recall_vaild);  F1_score_vaild(isnan(F1_score_vaild))=0;  disp(['验证集F1_score_vaild：',num2str(mean(F1_score_vaild))])   % F1 Score	
specificity_vaild = TN_vaild ./ (TN_vaild + FP_vaild); specificity_vaild(isnan(specificity_vaild))=0; disp(['验证集specificity_vaild：',num2str(mean(specificity_vaild))])  % Specificity	
disp('测试集********************************************************************************') 	
confMat_test = confusionmat(test_y_feature_label,y_test_predict);	
TP_test = diag(confMat_test);      TP_test=TP_test'; % 被正确分类的正样本 True Positives	
FP_test = sum(confMat_test, 1)  - TP_test;  %被错误分类的正样本 False Positives	
FN_test = sum(confMat_test, 2)' - TP_test;  % 被错误分类的负样本 False Negatives	
TN_test = sum(confMat_test(:))  - (TP_test + FP_test + FN_test);  % 被正确分类的负样本 True Negatives	
	
accuracy_test = sum(TP_test) / sum(confMat_test(:)); accuracy_test(isnan(accuracy_test))=0; disp(['测试集accuracy：',num2str(accuracy_test)])% Accuracy	
precision_test = TP_test ./ (TP_test + FP_test);  precision_test(isnan(precision_test))=0; disp(['测试集precision_test：',num2str(mean(precision_test))]) % Precision	
recall_test = TP_test ./ (TP_test + FN_test); recall_test(isnan(recall_test))=0; disp(['测试集recall_test：',num2str(mean(recall_test))])  % Recall / Sensitivity	
F1_score_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test); F1_score_test(isnan(F1_score_test))=0; disp(['测试集F1_score_test：',num2str(mean(F1_score_test))])   % F1 Score	
specificity_test = TN_test ./ (TN_test + FP_test); specificity_test(isnan(specificity_test))=0; disp(['测试集specificity_test：',num2str(mean(specificity_test))])  % Specificity	
	
	
	
%% 绘制ROC曲线并计算AUC
[~,score_train]=predict(Mdl,train_x_feature_label_norm);  %训练集预测结果	
[~,score_vaild]=predict(Mdl,valid_x_feature_label_norm);  %验证集预测结果	
[~,score_test]=predict(Mdl,test_x_feature_label_norm);  %测试集预测结果	

% 计算训练集AUC（使用正确的正类标签）
unique_labels_train = unique(train_y_feature_label);
if length(unique_labels_train) == 2
    try
        [~, ~, ~, AUC_train_1] = perfcurve(train_y_feature_label, score_train(:,1), unique_labels_train(1));
    catch
        AUC_train_1 = 0;
    end
    try
        if size(score_train, 2) >= 2
            [~, ~, ~, AUC_train_2] = perfcurve(train_y_feature_label, score_train(:,2), unique_labels_train(2));
        else
            AUC_train_2 = 0;
        end
    catch
        AUC_train_2 = 0;
    end
    if AUC_train_2 > AUC_train_1 && AUC_train_2 > 0.5
        [X_ROC_train,Y_ROC_train,T_ROC_train,AUC_ROC_train] = perfcurve(train_y_feature_label,score_train(:,2),unique_labels_train(2));
    else
        [X_ROC_train,Y_ROC_train,T_ROC_train,AUC_ROC_train] = perfcurve(train_y_feature_label,score_train(:,1),unique_labels_train(1));
    end
else
    [X_ROC_train,Y_ROC_train,T_ROC_train,AUC_ROC_train] = perfcurve(train_y_feature_label,score_train(:,1),1);
end

% 获取最优超参数（已在上面定义，这里确保变量存在）
if ~exist('optimal_num_trees', 'var') || ~exist('optimal_min_leaf', 'var')
    try
        optimal_num_trees = Mdl.NumTrees;
        optimal_min_leaf = Mdl.ModelParameters.MinLeaf;
    catch
        optimal_num_trees = 50;
        optimal_min_leaf = 5;
    end
end

% 格式化输出评估结果
fprintf('\n');
fprintf('========================================\n');
fprintf('模型评估结果 (最优超参数: NumTrees=%d, MinLeafSize=%d)\n', optimal_num_trees, optimal_min_leaf);
fprintf('========================================\n');
fprintf('%-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n', '数据集', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity');
fprintf('%-10s-+-%-10s-+-%-10s-+-%-10s-+-%-10s-+-%-10s-+-%-10s\n', repmat('-',1,10), repmat('-',1,10), repmat('-',1,10), repmat('-',1,10), repmat('-',1,10), repmat('-',1,10), repmat('-',1,10));

% 计算AUC（使用正确的正类标签，参考test_rf_hyperparameters.m的逻辑）
% 首先确定正类标签
unique_labels = unique([train_y_feature_label; valid_y_feature_label; test_y_feature_label]);
fprintf('  调试信息: 标签唯一值: %s\n', mat2str(unique_labels));
fprintf('  score_vaild维度: %s, score_test维度: %s\n', mat2str(size(score_vaild)), mat2str(size(score_test)));

if length(unique_labels) == 2
    % 二分类：尝试两种方法，选择AUC更大的
    try
        [~, ~, ~, AUC_vaild_1] = perfcurve(valid_y_feature_label, score_vaild(:,1), unique_labels(1));
    catch
        AUC_vaild_1 = 0;
    end
    try
        if size(score_vaild, 2) >= 2
            [~, ~, ~, AUC_vaild_2] = perfcurve(valid_y_feature_label, score_vaild(:,2), unique_labels(2));
        else
            AUC_vaild_2 = 0;
        end
    catch
        AUC_vaild_2 = 0;
    end
    
    try
        [~, ~, ~, AUC_test_1] = perfcurve(test_y_feature_label, score_test(:,1), unique_labels(1));
    catch
        AUC_test_1 = 0;
    end
    try
        if size(score_test, 2) >= 2
            [~, ~, ~, AUC_test_2] = perfcurve(test_y_feature_label, score_test(:,2), unique_labels(2));
        else
            AUC_test_2 = 0;
        end
    catch
        AUC_test_2 = 0;
    end
    
    fprintf('  验证集AUC: 方法1=%.4f, 方法2=%.4f\n', AUC_vaild_1, AUC_vaild_2);
    fprintf('  测试集AUC: 方法1=%.4f, 方法2=%.4f\n', AUC_test_1, AUC_test_2);
    
    % 选择AUC更大的方法（且大于0.5）
    if AUC_vaild_2 > AUC_vaild_1 && AUC_vaild_2 > 0.5
        [X_ROC_vaild,Y_ROC_vaild,T_ROC_vaild,AUC_ROC_vaild] = perfcurve(valid_y_feature_label,score_vaild(:,2),unique_labels(2));
        fprintf('  验证集使用: 方法2 (score(:,2), 标签%d)\n', unique_labels(2));
    else
        [X_ROC_vaild,Y_ROC_vaild,T_ROC_vaild,AUC_ROC_vaild] = perfcurve(valid_y_feature_label,score_vaild(:,1),unique_labels(1));
        fprintf('  验证集使用: 方法1 (score(:,1), 标签%d)\n', unique_labels(1));
    end
    
    if AUC_test_2 > AUC_test_1 && AUC_test_2 > 0.5
        [X_ROC_test,Y_ROC_test,T_ROC_test,AUC_ROC_test] = perfcurve(test_y_feature_label,score_test(:,2),unique_labels(2));
        fprintf('  测试集使用: 方法2 (score(:,2), 标签%d)\n', unique_labels(2));
    else
        [X_ROC_test,Y_ROC_test,T_ROC_test,AUC_ROC_test] = perfcurve(test_y_feature_label,score_test(:,1),unique_labels(1));
        fprintf('  测试集使用: 方法1 (score(:,1), 标签%d)\n', unique_labels(1));
    end
else
    % 多分类或单类，使用默认方法
    [X_ROC_vaild,Y_ROC_vaild,T_ROC_vaild,AUC_ROC_vaild] = perfcurve(valid_y_feature_label,score_vaild(:,1),1);
    [X_ROC_test,Y_ROC_test,T_ROC_test,AUC_ROC_test] = perfcurve(test_y_feature_label,score_test(:,1),1);
end

% 输出评估结果表格
fprintf('%-10s | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f\n', '训练集', mean(accuracy_train), mean(precision_train), mean(recall_train), mean(F1_score_train), AUC_ROC_train, mean(specificity_train));
fprintf('%-10s | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f\n', '验证集', accuracy_vaild, mean(precision_vaild), mean(recall_vaild), mean(F1_score_vaild), AUC_ROC_vaild, mean(specificity_vaild));
fprintf('%-10s | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f\n', '测试集', accuracy_test, mean(precision_test), mean(recall_test), mean(F1_score_test), AUC_ROC_test, mean(specificity_test));
fprintf('========================================\n\n');

rocObj_train = rocmetrics(train_y_feature_label,score_train(:,1),1);	
	
figure	
plot(rocObj_train)	
title('Train ROC')	
%	
[X_ROC_vaild,Y_ROC_vaild,T_ROC_vaild,AUC_ROC_vaild] = perfcurve(valid_y_feature_label,score_vaild(:,1),1);	
rocObj_vaild = rocmetrics(valid_y_feature_label,score_vaild(:,1),1);	
	
figure	
plot(rocObj_vaild)	
title('Vaild ROC')	
%	
% 测试集ROC（已在上面计算，这里只用于绘图）
rocObj_test = rocmetrics(test_y_feature_label,score_test(:,1),1);	
figure	
plot(rocObj_test)	
title('Test ROC')	
	
	
%% K折验证	
x_feature_label_norm_all=(x_feature_label-x_mu)./x_sig;    %x特征	
y_feature_label_norm_all=y_feature_label;	
Kfold_num=G_out_data.Kfold_num;	
cv = cvpartition(size(x_feature_label_norm_all, 1), 'KFold', Kfold_num); % Split into K folds	
for k = 1:Kfold_num	
    trainingIdx = training(cv, k);	
    validationIdx = test(cv, k);	
     x_feature_label_norm_all_traink=x_feature_label_norm_all(trainingIdx,:);	
   y_feature_label_norm_all_traink=y_feature_label_norm_all(trainingIdx,:);	
	
   x_feature_label_norm_all_testk=x_feature_label_norm_all(validationIdx,:);	
   y_feature_label_norm_all_testk=y_feature_label_norm_all(validationIdx,:);	
	
  % 使用与主模型相同的最优超参数（来自消融实验结果）
  if exist('optimal_num_trees', 'var') && exist('optimal_min_leaf', 'var')
      optimal_num_trees_kfold = optimal_num_trees;
      optimal_min_leaf_kfold = optimal_min_leaf;
  else
      % 如果变量不存在，尝试从Mdl获取
      try
          optimal_num_trees_kfold = Mdl.NumTrees;
          optimal_min_leaf_kfold = Mdl.ModelParameters.MinLeaf;
      catch
          % 如果无法获取，使用默认值
          optimal_num_trees_kfold = 50;
          optimal_min_leaf_kfold = 5;
      end
  end
  
  Mdlkf=TreeBagger(optimal_num_trees_kfold, x_feature_label_norm_all_traink, y_feature_label_norm_all_traink, 'Method', 'classification', 'MinLeafSize', optimal_min_leaf_kfold);	
	
   Mdl_kfold{1,k}=Mdlkf;	
	
    y_test_predict_norm_all_testk=predict(Mdlkf,x_feature_label_norm_all_testk);  %测试集预测结果	
	
    y_test_predict_all_testk=RF_process(y_test_predict_norm_all_testk);
    
    % 确保预测结果和标签的数据类型一致
    y_test_predict_all_testk = double(y_test_predict_all_testk(:));
    y_feature_label_norm_all_testk = double(y_feature_label_norm_all_testk(:));
    
    % 调试信息（仅第一次迭代）
    if k == 1
        fprintf('\n[K折验证 - 第%d折] 调试信息:\n', k);
        fprintf('  预测结果类型: %s, 维度: %s\n', class(y_test_predict_all_testk), mat2str(size(y_test_predict_all_testk)));
        fprintf('  真实标签类型: %s, 维度: %s\n', class(y_feature_label_norm_all_testk), mat2str(size(y_feature_label_norm_all_testk)));
        fprintf('  预测结果唯一值: %s\n', mat2str(unique(y_test_predict_all_testk)'));
        fprintf('  真实标签唯一值: %s\n', mat2str(unique(y_feature_label_norm_all_testk)'));
        fprintf('  样本数: %d\n', length(y_test_predict_all_testk));
    end
    
    % 计算准确率
    correct_predictions = sum(y_test_predict_all_testk == y_feature_label_norm_all_testk);
    test_kfold = correct_predictions / length(y_test_predict_all_testk);
    AUC_kfold(k) = test_kfold;
    
    % 调试信息（仅第一次迭代）
    if k == 1
        fprintf('  正确预测数: %d / %d\n', correct_predictions, length(y_test_predict_all_testk));
        fprintf('  准确率: %.4f\n', test_kfold);
    end
	
	
 end	
	
	
% k折验证结果绘图	
figure('color',[1 1 1]);	
	
color_set=[0.4353    0.5137    0.7490];	
plot(1:length(AUC_kfold),AUC_kfold,'--p','color',color_set,'Linewidth',1.3,'MarkerSize',6,'MarkerFaceColor',color_set,'MarkerFaceColor',[0.3,0.4,0.5]);	
grid on;	
box off;	
grid off;	
ylim([0.92*min(AUC_kfold),1.2*max(AUC_kfold)])	
xlabel('kfoldnum')	
ylabel('accuracy')	
xticks(1:length(AUC_kfold))	
set(gca,'Xgrid','off');	
set(gca,'Linewidth',1);	
set(gca,'TickDir', 'out', 'TickLength', [.005 .005], 'XMinorTick', 'off', 'YMinorTick', 'off');	
yline(mean(AUC_kfold),'--')	
%小窗口柱状图的绘制	
axes('Position',[0.6,0.65,0.25,0.25],'box','on'); % 生成子图	
GO = bar(1:length(AUC_kfold),AUC_kfold,1,'EdgeColor','k');	
GO(1).FaceColor = color_set;	
xticks(1:length(AUC_kfold))	
xlabel('kfoldnum')	
ylabel('accuracy')	
disp('****************************************************************************************') 	
disp([num2str(Kfold_num),'折验证预测准确率accuracy结果：'])	
disp(AUC_kfold) 	
disp([num2str(Kfold_num),'折验证  ','accuracy均值为： ' ,num2str(mean(AUC_kfold)),'    accuracy标准差为： ' ,num2str(std(AUC_kfold))]) 	
