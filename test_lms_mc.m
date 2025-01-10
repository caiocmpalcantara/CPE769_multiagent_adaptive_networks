addpath("./Technique/")
% rng(8988466)
x = [1 1 1]';
u = [1 1 1];
N = 1000;
M = 300;
n = 1:N;
noise = randn(1,N,M);

d = u*x + noise;

%% Criando a técnica
% tec = Lms();
tec = Lms('H_ini', [0 0 0], 'mu', .2);
% tec = Lms('n_win', 50);
% tec = Lms('mu', 0.5);
% tec = Lms('epsilon', 0.1);

%% Monte-Carlo

buf_obs = zeros(tec.y_dim, tec.n_win);
buf_st = zeros(tec.x_dim, tec.n_win);
y_hat = zeros(tec.y_dim, N, M);
u_hat = zeros(tec.y_dim, tec.x_dim, N, M);
for m = 1:M
    for i = 1:N
        buf_obs = update_buffer(buf_obs, d(1,i,m));
        buf_st = update_buffer(buf_st, x);
        y_hat(:,i,m) = tec.apply(buf_obs, buf_st);
        u_hat(:,:,i,m) = tec.get_H();
    end
end

%% Figures
figure(1)
plot(n,d(:,:,9),'b')
hold on
plot(n,y_hat(:,:,9),'--r')
xlabel('Time')
ylabel('Observation')
title('A realization')
grid on
hold off
%% The error
e1 = (y_hat-u*x).^2;
e1m = mean(e1,3);
e2 = zeros(1,N);
for m = 1:M
    for i = 1:N
        e2(i,m) = norm(u_hat(:,:,i,m)-u);
    end    
end
e2m = mean(e2,2);

figure(2)
plot(n,10*log10(e1m),'b')
title('Test LMS: Error.')
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-30 0])
grid on

figure(3)
plot(n,10*log10(e2m),'r')
title('Test LMS: Error.')
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-30 0])
grid on

%% functions
function buff = update_buffer(buffer, new_sample)
    buff = buffer;
    buff(:, 2:end) = buff(:, 1:end-1);
    buff(:, 1) = new_sample;
end