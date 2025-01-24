[X, Z] = meshgrid(0:0.5:30, 0:0.02:1); % 定义网格
Y = 1 - (0.75).^X; % 定义函数

figure;
surf(X, Z, Y, 'EdgeColor', 'none'); % 绘制表面图，隐藏网格线
colormap(parula); % 使用默认 Parula 色系
xlabel('Secret Length m', 'FontSize', 12); % X轴标签
zlabel('Detection \rho_2', 'FontSize', 12); % Z轴标签
set(gca, 'YTick', []); % 不显示Y轴刻度
set(gca, 'XTick', 0:5:30, 'XTickLabel', {'0', '5', '10', '15', '20', '25', '30'}); % 手动设置X轴刻度
set(gca, 'ZTick', 0:0.2:1); % 设置Z轴刻度
view(30, 45); % 调整视角
alpha(0.8); % 设置透明度
