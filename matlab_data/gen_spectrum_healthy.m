clc; clear; close all;

data = readmatrix('compressed_virus_data.csv');

healthy_idx = find(data(:, 3) == 0);

healthy_data = data(healthy_idx, 4 : end);

avg_healthy_data = mean(healthy_data);
save('average_healthy_spectrum.mat', "avg_healthy_data");

figure();
plot(avg_healthy_data);
saveas(gcf, 'average_healthy_spectrum.png');
close();

