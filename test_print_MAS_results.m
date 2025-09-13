load('msd_rls_gaaf_N200_M100.mat')
rls_gaaf = final_msd_vec;
load('msd_kf_gaaf_N200_M100.mat')
kf_gaaf = final_msd_vec;
load('msd_kf_kftmd_N200_M100.mat')
kf_kftmd = final_msd_vec;
load('msd_kf_kfimd_N200_M100.mat')
kf_kfimd = final_msd_vec;
N = 200;
figure(1)
plot(1:N, 20*log10(rls_gaaf), 'b', 'LineWidth', 1.5);
hold on;
plot(1:N, 20*log10(kf_gaaf), 'r', 'LineWidth', 1.5);
plot(1:N, 20*log10(kf_kftmd), 'g', 'LineWidth', 1.5);
plot(1:N, 20*log10(kf_kfimd), 'k', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Final MSD [dB]');
title('Final MSD Comparison');
legend('RLS-GAAF', 'KF-GAAF', 'KF-KFTMD', 'KF-KFIMD', 'Location', 'best');
hold off;
grid on;