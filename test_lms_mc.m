addpath("./Technique/")

rng(8988466)
u = [.2 .6];
N = 300;
M = 600;
n = 1:N;

% x = linspace(1,1,N);
% x = linspace(3,3,N);
x = linspace(1,10,N);
% x = sin(n/10*pi);
% x = zeros(1,N);
% x(1:N/2) = linspace(1,10,N/2);
% x(N/2+1:end) = linspace(10,1,N/2);
d = zeros(1,N,M);
y = zeros(1,N);
noise = .2*randn(1,N,M);

% Dynamics
y(1) = u*[x(1) 0]';
for i = 1:N-1
    y(i+1) = u*[x(i+1) x(i)]';
end
for m = 1:M
    d(1,1,m) = y(1) + noise(1,1,m);
    for i = 1:N-1
        d(1,i+1,m) = y(i+1) + noise(1,i+1,m);
    end
end

%% Criando a técnica
tec1 = Lms('x_dim', 2, 'y_dim', 1, 'H_ini', [0 0], 'epsilon', 1e-5, 'mu', .3);
buf_obs = zeros(tec1.y_dim, 5);
buf_st = zeros(tec1.x_dim, 5);
y_hat = zeros(tec1.y_dim, N, M);
u_hat = zeros(tec1.y_dim, tec1.x_dim, N, M);

% MC
for m = 1:M
    for i = 1:N
        buf_obs = update_buffer(buf_obs, d(1,i,m));
        if i == 1
            buf_st = update_buffer(buf_st, [x(i) 0]');
        else
            buf_st = update_buffer(buf_st, [x(i) x(i-1)]');
        end
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
plot(n,d(1,:,m),'b')
xlabel('Time')
ylabel('Observation')
title('Testing LMS Class')
hold on
grid on
plot(n, y_hat(1,:,m), '--r')
hold off
%% The error
e1 = (y_hat(1,:,:)-d(1,:,:)).^2;
e1m = mean(e1,3);
e2 = zeros(1,N);
for m = 1:M
    for i = 1:N
        e2(i) = norm(u_hat(:,:,i,m)-u);
    end
end
e2m = mean(e2,4);
figure(2)
plot(n,10*log10(e1m),'b')
% hold on
% plot(n,10*log10(erro.^2),'r')
% hold off
title('Test LMS: Error.')
ylabel('e[n]')
xlabel('n')
grid on
figure(3)
plot(n,e2m,'r')
title('Test LMS: Error.')
ylabel('e[n]')
xlabel('n')
grid on

%% functions
function buff = update_buffer(buffer, new_sample)
    buff = buffer;
    buff(:, 2:end) = buff(:, 1:end-1);
    buff(:, 1) = new_sample;
end