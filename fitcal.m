function fitval = fitcal(pop, num_hidden, p_train, t_train, net)

%%  节点个数
inputnum  = size(p_train, 1);  % 输入层节点数
outputnum = size(t_train, 1);  % 输出层节点数

%%  提取权�?�和阈�??
w1 = pop(1 : inputnum * num_hidden);
B1 = pop(inputnum * num_hidden + 1 : inputnum * num_hidden + num_hidden);
w2 = pop(inputnum * num_hidden + num_hidden + 1 : inputnum * num_hidden + num_hidden + num_hidden * outputnum);
B2 = pop(inputnum * num_hidden + num_hidden + num_hidden * outputnum + 1 : inputnum * num_hidden + num_hidden + num_hidden * outputnum + outputnum);

%%  网络赋�??
net.Iw{1, 1} = reshape(w1, num_hidden, inputnum );
net.Lw{2, 1} = reshape(w2, outputnum, num_hidden);
net.b{1}     = reshape(B1, num_hidden, 1);
net.b{2}     = B2';

%%  网络训练
net = train(net, p_train, t_train);

%%  仿真测试
t_sim = sim(net, p_train);

%%  反归�?�?
T_sim  = vec2ind(t_sim);
T_train = vec2ind(t_train);

%%  适应度�??
fitval = 1 - sum(T_sim == T_train) / length(T_sim);

end