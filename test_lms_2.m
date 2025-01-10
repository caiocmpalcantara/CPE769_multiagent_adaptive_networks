addpath("./Technique/")

rng(8988466)
u = [.2 .6];
N = 200;
n = 1:N;
% x = linspace(3,3,N);
x = linspace(1,10,N);
% x = sin(n/10*pi);
d = zeros(1,N);
y = zeros(1,N);
noise = 0.4*randn(1,N);

y(1) = u*[x(1) 0]';
d(1) = y(1) + noise(1);
for i = 1:N-1
    y(i+1) = u*[x(i+1) x(i)]';
    d(i+1) = y(i+1) + noise(i+1);
end
% d(2:end) = x(2:end);

%% Figures
figure(1)
plot(n,d,'b')
xlabel('Time')
ylabel('Observation')
title('Testing LMS Class')
hold on
grid on

%% Criando a técnica
tec1 = Lms2('x_dim', 2, 'y_dim', 1, 'H_ini', [0 0], 'epsilon', 1e-5, 'mu', 0.3);
buf_obs = zeros(tec1.y_dim, 5);
buf_st = zeros(tec1.x_dim, 5);
y_hat = zeros(tec1.y_dim, N);
u_hat = zeros(tec1.y_dim, tec1.x_dim, N);
for i = 1:N
    buf_obs = update_buffer(buf_obs, d(i));
    if i == 1
        buf_st = update_buffer(buf_st, [x(i) 0]');
    else
        buf_st = update_buffer(buf_st, [x(i) x(i-1)]');
    end
    y_hat(i) = tec1.apply(buf_obs, buf_st);
    u_hat(:,:,i) = tec1.get_H();
end

plot(n, y_hat, '--r')
%% Compare
% nlms = dsp.LMSFilter(2,'Method', 'Normalized LMS','StepSize', 1);
% [y_hat2,erro,u_hat2] = nlms(x',d');
% 
% plot(n, y_hat2, '--k')

%%
% [mmse,emse,meanW,mse,traceK] = msepred(nlms,x',d',m);
% [simmse,meanWsim,Wsim,traceKsim] = msesim(nlms,x',d',m);

%% The error
e1 = (y_hat-d).^2;
e2 = zeros(1,N);
for i = 1:N
    e2(i) = norm(u_hat(i)-u);
end
figure(2)
plot(n,10*log10(e1),'b')
% hold on
% plot(n,10*log10(erro.^2),'r')
% hold off
title('Test LMS: Error.')
ylabel('e[n]')
xlabel('n')
grid on
figure(3)
plot(n,e2,'r')
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