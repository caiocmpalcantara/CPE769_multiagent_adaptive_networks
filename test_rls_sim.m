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
tec1 = Rls('x_dim', 3, 'y_dim', 1, 'H_ini', [0 0 0], 'lambda', .85, 'delta', 1e-3);
buf_obs = zeros(tec1.y_dim, 5);
buf_st = zeros(tec1.x_dim, 5);
y_hat = zeros(tec1.y_dim, N, M);
u_hat = zeros(tec1.y_dim, tec1.x_dim, N, M);

% MC
for m = 1:M
    for i = 1:N
        buf_obs = update_buffer(buf_obs, d(1,i,m));
        buf_st = update_buffer(buf_st, x(i:i+2));
        y_hat(:,i,m) = tec1.apply(buf_obs, buf_st);
        u_hat(:,:,i,m) = tec1.get_H();
    end
end
% %% Compare
% nlms = dsp.LMSFilter(2,'Method', 'Normalized LMS','StepSize', 1);
% [y_hat2,erro,u_hat2] = nlms(x',d');
% 
% plot(n, y_hat2, '--k')
% 
% %%
% [mmse,emse,meanW,mse,traceK] = msepred(nlms,x',d',m);
% [simmse,meanWsim,Wsim,traceKsim] = msesim(nlms,x',d',m);

%% Figures
figure(1)
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

figure(2)
plot(1:N,10*log10(e1m),'b')
title('MSE: RLS.')
ylabel('e[n]')
xlabel('n')
set(gca, 'YLim', [-30 0])
grid on

figure(3)
plot(1:N,20*log10(e2m),'r')
title('MSD: RLS.')
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