addpath("./Technique/")
rng(8988467)
% x = [1 1 1]';
u = [.1 .3 .7];
N = 1000;
M = 300;
x = 0.5*randn(1,N+2) + 1;
noise = 0.2*randn(1,N,M);
d = zeros(1,N,M);
y = zeros(1,N);

for n = 1:N
    y(n) = u*x(n:n+2)';
    d(1,n,:) = y(n) + noise(1,n,:);
end


%% Criando a técnica
% tec = Wiener();
tec = Wiener('H_ini', [0 0 0], 'mu', .2);
% tec = Wiener('n_win', 50);
% tec = Wiener('mu', 0.5);
% tec = Wiener('epsilon', 0.1);

%% Monte-Carlo

buf_obs = zeros(tec.y_dim, tec.n_win);
buf_st = zeros(tec.x_dim, tec.n_win);
y_hat = zeros(tec.y_dim, N, M);
u_hat = zeros(tec.y_dim, tec.x_dim, N, M);
for m = 1:M
    for i = 1:N
        buf_obs = update_buffer(buf_obs, d(1,i,m));
        buf_st = update_buffer(buf_st, x(i:i+2));
        y_hat(:,i,m) = tec.apply(buf_obs, buf_st);
        u_hat(:,:,i,m) = tec.get_H();
    end
end

%% Figures
figure(10)
plot(1:N,d(:,:,9),'b')
hold on
plot(1:N,y_hat(:,:,9),'--r')
xlabel('Time')
ylabel('Observation')
title('A realization')
grid on
hold off
%% The error
e1 = (y_hat-y).^2;
e1m = mean(e1,3);
e2 = zeros(1,N,M);
for m = 1:M
    for i = 1:N
        e2(1,i,m) = norm(u_hat(:,:,i,m)-u);
    end    
end
e2m = mean(e2,3);

figure(20)
plot(1:N,10*log10(e1m),'b')
title('MSE: Wiener.')
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-30 0])
grid on

figure(30)
plot(1:N,20*log10(e2m),'r')
title('MSD: Wiener.')
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