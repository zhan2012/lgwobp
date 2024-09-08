%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��ȡ����

%res = xlsread('diabetic_data.csv');
res = xlsread('fy3.csv');
%%  ��������
num_class = length(unique(res(:, end)));  % �������Excel���һ�з����
num_res = size(res, 1);                   % ��������ÿһ�У���һ��������
num_size = 0.7;                           % ѵ����ռ���ݼ��ı���
res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�
flag_conusion = 1;                        % ��־λΪ1���򿪻�������Ҫ��2018�汾�����ϣ�

%%  ���ñ����洢����
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  �������ݼ�
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % ѭ��ȡ����ͬ��������
    mid_size = size(mid_res, 1);                    % �õ���ͬ�����������
    mid_tiran = round(num_size * mid_size);         % �õ�������ѵ����������

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % ѵ��������
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % ѵ�������

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % ���Լ�����
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % ���Լ����
end

%%  ����ת��
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  �õ�ѵ�����Ͳ�����������
M = size(P_train, 2);
N = size(P_test , 2);

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = ind2vec(T_train);
t_test  = ind2vec(T_test);

%%  �����������
num_inputs = size(p_train, 1);         % �����ά��
num_hidden = 6;                        % ���ز�ά��
num_output = num_class;                % �����ά��
dim = (num_inputs + 1) * num_hidden + (num_hidden + 1) * num_output;    % �Ż���������

%%  ��������
net = newff(p_train, t_train, num_hidden);

%%  ����ѵ������
net.trainParam.epochs     = 1000;      % ѵ������
net.trainParam.goal       = 1e-9;      % Ŀ�����
net.trainParam.lr         = 0.01;      % ѧϰ��
net.trainParam.showWindow = 0;         % �رմ���

%%  �Ż��㷨��������
SearchAgents_no = 5;                   % ��Ⱥ����
Max_iteration = 30;                    % ����������
lb = -1.0 * ones(1, dim);              % ����ȡֵ�½�
ub =  1.0 * ones(1, dim);              % ����ȡֵ�Ͻ�

%%  �Ż��㷨��ʼ��
Alpha_pos = zeros(1, dim);  % ��ʼ��Alpha�ǵ�λ��
Alpha_score = inf;          % ��ʼ��Alpha�ǵ�Ŀ�꺯��ֵ,�������Ϊ-inf�Խ���������

Beta_pos = zeros(1, dim);   % ��ʼ��Beta�ǵ�λ��
Beta_score = inf;           % ��ʼ��Beta�ǵ�Ŀ�꺯��ֵ ,�������Ϊ-inf�Խ���������

Delta_pos = zeros(1, dim);  % ��ʼ��Delta�ǵ�λ��
Delta_score = inf;          % ��ʼ��Delta�ǵ�Ŀ�꺯��ֵ,�������Ϊ-inf�Խ���������

%%  ��ʼ��������Ⱥ��λ��
Positions = initialization(SearchAgents_no, dim, ub, lb);

%%  ���ڼ�¼��������
Convergence_curve = zeros(1, Max_iteration);

%%  ѭ��������
iter = 0;

%%  �Ż��㷨��ѭ��
while iter < Max_iteration           % �Ե�������ѭ��
    for i = 1 : size(Positions, 1)   % ����ÿ����

        % ���س��������ռ�߽��������Ⱥ
        % ������λ�ó����������ռ䣬��Ҫ���»ص������ռ�
        Flag4ub = Positions(i, :) > ub;
        Flag4lb = Positions(i, :) < lb;

        % ���ǵ�λ�������ֵ����Сֵ֮�䣬��λ�ò���Ҫ���������������ֵ����ص����ֵ�߽�
        % ��������Сֵ����ش���Сֵ�߽�
        Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;                 

        % ������Ӧ�Ⱥ���ֵ
        X = reshape(Positions(i, :), 1, dim);
        fitness = (fitcal(X, num_hidden, p_train, t_train, net)); 

        % ���� Alpha, Beta, Delta
        if fitness < Alpha_score           % ���Ŀ�꺯��ֵС��Alpha�ǵ�Ŀ�꺯��ֵ
            Alpha_score = fitness;         % ��Alpha�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Alpha_pos = Positions(i, :);   % ͬʱ��Alpha�ǵ�λ�ø���Ϊ����λ��
        end

        if fitness > Alpha_score && fitness < Beta_score   % ���Ŀ�꺯��ֵ������Alpha�Ǻ�Beta�ǵ�Ŀ�꺯��ֵ֮��
            Beta_score = fitness;                          % ��Beta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Beta_pos = Positions(i, :);                    % ͬʱ����Beta�ǵ�λ��
        end

        if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score  % ���Ŀ�꺯��ֵ������Beta�Ǻ�Delta�ǵ�Ŀ�꺯��ֵ֮��
            Delta_score = fitness;                                                 % ��Delta�ǵ�Ŀ�꺯��ֵ����Ϊ����Ŀ�꺯��ֵ
            Delta_pos = Positions(i, :);                                           % ͬʱ����Delta�ǵ�λ��
        end

    end

    % ����Ȩ�صݼ�
    wa = 2 - iter * ((2) / Max_iteration);    

    % ����������Ⱥ��λ��
    for i = 1 : size(Positions, 1)      % ����ÿ����
        for j = 1 : size(Positions, 2)  % ����ÿ��ά��

            % ��Χ���λ�ø���
            r1 = rand; % r1 is a random number in [0,1]
            r2 = rand; % r2 is a random number in [0,1]

            A1 = 2 * wa * r1 - wa;   % ����ϵ��A��Equation (3.3)
            C1 = 2 * r2;             % ����ϵ��C��Equation (3.4)

            % Alpha λ�ø���
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));   % Equation (3.5)-part 1
            X1 = Alpha_pos(j) - A1 * D_alpha;                     % Equation (3.6)-part 1

            r1 = rand; % r1 is a random number in [0,1]
            r2 = rand; % r2 is a random number in [0,1]

            A2 = 2 * wa * r1 - wa;   % ����ϵ��A��Equation (3.3)
            C2 = 2 *r2;              % ����ϵ��C��Equation (3.4)

            % Beta λ�ø���
            D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));    % Equation (3.5)-part 2
            X2 = Beta_pos(j) - A2 * D_beta;                      % Equation (3.6)-part 2       

            r1 = rand;  % r1 is a random number in [0,1]
            r2 = rand;  % r2 is a random number in [0,1]

            A3 = 2 *wa * r1 - wa;     % ����ϵ��A��Equation (3.3)
            C3 = 2 *r2;               % ����ϵ��C��Equation (3.4)

            % Delta λ�ø���
            D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));   % Equation (3.5)-part 3
            X3 = Delta_pos(j) - A3 * D_delta;                     % Equation (3.5)-part 3

            % λ�ø���
            Positions(i, j) = (X1 + X2 + X3) / 3;                 % Equation (3.7)

        end
    end

    % ���µ�����
    iter = iter + 1;    
    Convergence_curve(iter) = Alpha_score;

end

%%  ��ȡ����Ȩֵ��ƫ��
w1 = Alpha_pos(1 : num_inputs * num_hidden);
B1 = Alpha_pos(num_inputs * num_hidden + 1 : num_inputs * num_hidden + num_hidden);
w2 = Alpha_pos(num_inputs * num_hidden + num_hidden + 1 : num_inputs * num_hidden ...
    + num_hidden + num_hidden * num_output);
B2 = Alpha_pos(num_inputs * num_hidden + num_hidden + num_hidden * num_output + 1 : ...
    num_inputs * num_hidden + num_hidden + num_hidden * num_output + num_output);

%%  ���縳ֵ
net.Iw{1, 1} = reshape(w1, num_hidden, num_inputs);
net.Lw{2, 1} = reshape(w2, num_output, num_hidden);
net.b{1}     = reshape(B1, num_hidden, 1);
net.b{2}     = B2';

%%  ��ѵ������ 
net.trainParam.showWindow = 1;        % �򿪴���

%%  ����ѵ��
net = train(net, p_train, t_train);

%%  ����Ԥ��
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  ���ݷ���һ��
T_sim1 = vec2ind(t_sim1);
T_sim2 = vec2ind(t_sim2);

%%  ��������
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['׼ȷ��=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['׼ȷ��=' num2str(error2) '%']};
title(string)
grid

%%  ��Ӧ������
figure
plot(1 : length(Convergence_curve), Convergence_curve, 'LineWidth', 1.5);
title('��Ӧ������', 'FontSize', 13);
xlabel('��������', 'FontSize', 10);
ylabel('��Ӧ��ֵ', 'FontSize', 10);
xlim([1, length(Convergence_curve)])
grid on

%%  ��������
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
%% ROC���߻���
Pro = 1;       %  ���Ʒ��༸��ROC����
% ѵ����ROC
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

% ���Լ�ROC
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

save("NET.mat","net","ps_input",'x1','x2','y1','y2','auc1','auc2') % ����ѵ���õ�ģ�ͺ͹�һ������
disp('-----------------------�����������--------------------------')

