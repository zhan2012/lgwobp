%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取数据

%res = xlsread('diabetic_data.csv');
res = xlsread('fy3.csv');
%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.7;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 1;                        % 标志位为1，打开混淆矩阵（要求2018版本及以上）

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
end

%%  数据转置
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  得到训练集和测试样本个数
M = size(P_train, 2);
N = size(P_test , 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = ind2vec(T_train);
t_test  = ind2vec(T_test);

%%  网络参数设置
num_inputs = size(p_train, 1);         % 输入层维度
num_hidden = 6;                        % 隐藏层维度
num_output = num_class;                % 输出层维度
dim = (num_inputs + 1) * num_hidden + (num_hidden + 1) * num_output;    % 优化参数个数

%%  建立网络
net = newff(p_train, t_train, num_hidden);

%%  设置训练参数
net.trainParam.epochs     = 1000;      % 训练次数
net.trainParam.goal       = 1e-9;      % 目标误差
net.trainParam.lr         = 0.01;      % 学习率
net.trainParam.showWindow = 0;         % 关闭窗口

%%  优化算法参数设置
SearchAgents_no = 5;                   % 狼群数量
Max_iteration = 30;                    % 最大迭代次数
lb = -1.0 * ones(1, dim);              % 参数取值下界
ub =  1.0 * ones(1, dim);              % 参数取值上界

%%  优化算法初始化
Alpha_pos = zeros(1, dim);  % 初始化Alpha狼的位置
Alpha_score = inf;          % 初始化Alpha狼的目标函数值,将其更改为-inf以解决最大化问题

Beta_pos = zeros(1, dim);   % 初始化Beta狼的位置
Beta_score = inf;           % 初始化Beta狼的目标函数值 ,将其更改为-inf以解决最大化问题

Delta_pos = zeros(1, dim);  % 初始化Delta狼的位置
Delta_score = inf;          % 初始化Delta狼的目标函数值,将其更改为-inf以解决最大化问题

%%  初始化搜索狼群的位置
Positions = initialization(SearchAgents_no, dim, ub, lb);

%%  用于记录迭代曲线
Convergence_curve = zeros(1, Max_iteration);

%%  循环计数器
iter = 0;

%%  优化算法主循环
while iter < Max_iteration           % 对迭代次数循环
    for i = 1 : size(Positions, 1)   % 遍历每个狼

        % 返回超出搜索空间边界的搜索狼群
        % 若搜索位置超过了搜索空间，需要重新回到搜索空间
        Flag4ub = Positions(i, :) > ub;
        Flag4lb = Positions(i, :) < lb;

        % 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界
        % 若超出最小值，最回答最小值边界
        Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;                 

        % 计算适应度函数值
        X = reshape(Positions(i, :), 1, dim);
        fitness = (fitcal(X, num_hidden, p_train, t_train, net)); 

        % 更新 Alpha, Beta, Delta
        if fitness < Alpha_score           % 如果目标函数值小于Alpha狼的目标函数值
            Alpha_score = fitness;         % 则将Alpha狼的目标函数值更新为最优目标函数值
            Alpha_pos = Positions(i, :);   % 同时将Alpha狼的位置更新为最优位置
        end

        if fitness > Alpha_score && fitness < Beta_score   % 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
            Beta_score = fitness;                          % 则将Beta狼的目标函数值更新为最优目标函数值
            Beta_pos = Positions(i, :);                    % 同时更新Beta狼的位置
        end

        if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score  % 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
            Delta_score = fitness;                                                 % 则将Delta狼的目标函数值更新为最优目标函数值
            Delta_pos = Positions(i, :);                                           % 同时更新Delta狼的位置
        end

    end

    % 线性权重递减
    wa = 2 - iter * ((2) / Max_iteration);    

    % 更新搜索狼群的位置
    for i = 1 : size(Positions, 1)      % 遍历每个狼
        for j = 1 : size(Positions, 2)  % 遍历每个维度

            % 包围猎物，位置更新
            r1 = rand; % r1 is a random number in [0,1]
            r2 = rand; % r2 is a random number in [0,1]

            A1 = 2 * wa * r1 - wa;   % 计算系数A，Equation (3.3)
            C1 = 2 * r2;             % 计算系数C，Equation (3.4)

            % Alpha 位置更新
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));   % Equation (3.5)-part 1
            X1 = Alpha_pos(j) - A1 * D_alpha;                     % Equation (3.6)-part 1

            r1 = rand; % r1 is a random number in [0,1]
            r2 = rand; % r2 is a random number in [0,1]

            A2 = 2 * wa * r1 - wa;   % 计算系数A，Equation (3.3)
            C2 = 2 *r2;              % 计算系数C，Equation (3.4)

            % Beta 位置更新
            D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));    % Equation (3.5)-part 2
            X2 = Beta_pos(j) - A2 * D_beta;                      % Equation (3.6)-part 2       

            r1 = rand;  % r1 is a random number in [0,1]
            r2 = rand;  % r2 is a random number in [0,1]

            A3 = 2 *wa * r1 - wa;     % 计算系数A，Equation (3.3)
            C3 = 2 *r2;               % 计算系数C，Equation (3.4)

            % Delta 位置更新
            D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));   % Equation (3.5)-part 3
            X3 = Delta_pos(j) - A3 * D_delta;                     % Equation (3.5)-part 3

            % 位置更新
            Positions(i, j) = (X1 + X2 + X3) / 3;                 % Equation (3.7)

        end
    end

    % 更新迭代器
    iter = iter + 1;    
    Convergence_curve(iter) = Alpha_score;

end

%%  获取最优权值和偏置
w1 = Alpha_pos(1 : num_inputs * num_hidden);
B1 = Alpha_pos(num_inputs * num_hidden + 1 : num_inputs * num_hidden + num_hidden);
w2 = Alpha_pos(num_inputs * num_hidden + num_hidden + 1 : num_inputs * num_hidden ...
    + num_hidden + num_hidden * num_output);
B2 = Alpha_pos(num_inputs * num_hidden + num_hidden + num_hidden * num_output + 1 : ...
    num_inputs * num_hidden + num_hidden + num_hidden * num_output + num_output);

%%  网络赋值
net.Iw{1, 1} = reshape(w1, num_hidden, num_inputs);
net.Lw{2, 1} = reshape(w2, num_output, num_hidden);
net.b{1}     = reshape(B1, num_hidden, 1);
net.b{2}     = B2';

%%  打开训练窗口 
net.trainParam.showWindow = 1;        % 打开窗口

%%  网络训练
net = train(net, p_train, t_train);

%%  仿真预测
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  数据反归一化
T_sim1 = vec2ind(t_sim1);
T_sim2 = vec2ind(t_sim2);

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
grid

%%  适应度曲线
figure
plot(1 : length(Convergence_curve), Convergence_curve, 'LineWidth', 1.5);
title('适应度曲线', 'FontSize', 13);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度值', 'FontSize', 10);
xlim([1, length(Convergence_curve)])
grid on

%%  混淆矩阵
if flag_conusion == 1

    figure
    cm = confusionchart(T_train, T_sim1);
    cm.Title = 'Confusion Matrix for Train Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    
    figure
    cm = confusionchart(T_test, T_sim2);
    cm.Title = 'Confusion Matrix for Test Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end
%% ROC曲线绘制
Pro = 1;       %  绘制分类几的ROC曲线
% 训练集ROC
[x1,y1,~,auc1] = perfcurve(T_train,t_sim1(Pro,:),Pro);
figure
plot(x1,y1,'Color','#167C80','LineWidth',1.5);  
hold on
plot(x1,x1,'--','Color','k','linewidth',1.5)
xlabel('False Positive Rate','FontSize',14,'Fontname','Times New Roman');  
ylabel('True Positive Rate','FontSize',14,'Fontname','Times New Roman');  
title('ROC Curve for Train Data'); 
legend(['AUC=',num2str(auc1)])
set(gca,'Box','off','FontName','Times New Roman','FontSize',12,'LineWidth',1.5);
axis([-0.02,1,-0.02,1]) 

% 测试集ROC
[x2,y2,~,auc2] = perfcurve(T_test,t_sim2(Pro,:),Pro);
figure
plot(x2,y2,'Color','#167C80','LineWidth',1.5);  
hold on
plot(x2,x2,'--','Color','k','linewidth',1.5)
xlabel('False Positive Rate','FontSize',14,'Fontname','Times New Roman');  
ylabel('True Positive Rate','FontSize',14,'Fontname','Times New Roman');  
title('ROC Curve for Test Data'); 
legend(['AUC=',num2str(auc2)])
set(gca,'Box','off','FontName','Times New Roman','FontSize',12,'LineWidth',1.5);
axis([-0.02,1,-0.02,1]) 

save("NET.mat","net","ps_input",'x1','x2','y1','y2','auc1','auc2') % 保存训练好的模型和归一化规则
disp('-----------------------代码运行完毕--------------------------')

