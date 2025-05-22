%load data
file = readtable("filename");
lincome = file{:,5};
yeduc = file{:,3};
exper = file{:,1};
exper2 = file{:,2};

beta_ini = [2, 2, 2, 2];

% OLS
fun = @(beta) OLS(lincome, yeduc, exper, exper2, beta);
[beta_hat, fval] = fminsearch(fun,beta_ini);

n = numel(lincome);
k = numel(beta_ini) - 1;

y_hat = beta_hat(1) * ones(n,1) + beta_hat(2) * yeduc...
    + beta_hat(3) * exper + beta_hat(4) * exper2;

%残差
e_i = lincome - y_hat;
% histogram(e_i)

%残差２乗和
SSR = fval;
%総変動Y
SST = sum((lincome - mean(lincome)).^2);
%総変動yeduc
SST_yeduc = sum((yeduc - mean(yeduc)).^2);

%決定係数
R2 = 1 - SSR/SST;
%自由度調整済み決定係数
adj_R2 = 1 - (1 -R2)*(n -1)/(n - k - 1);

%=====標準誤差を行列演算で求める=====

%残差分散
s2 = SSR/(n - k);

% 説明変数行列 X を作成（切片項 + yeduc, exper, exper2）
X = [ ones(n,1), yeduc, exper, exper2 ];

%分散共分散行列
cov_e = s2 * inv(X' * X);

%分散共分散行列の対角成分
diag_cov_e = diag(cov_e);

%パラメターの標準誤差
SE = sqrt(diag_cov_e);

%=================================

%=====標準誤差を元の定義から求める=====

%誤差項の分散
s_hat2 = SSR/(n - k - 1);

% データ長(yeduc)
n_yeduc = numel(yeduc);

% 説明変数行列 Z を作成（切片項 + exper, exper2）
Z = [ ones(n_yeduc,1), exper, exper2 ];

% 回帰係数の最小二乗推定量
%    beta_edu_hat = (Z'Z)^{-1} Z'yeduc
beta_edu_hat = (Z' * Z) \ (Z' * yeduc);

% 残差ベクトル
resid_edu = yeduc - Z * beta_edu_hat;

% SSR(残差2乗和)
SSR_edu = resid_edu' * resid_edu;

% 部分決定係数(yeduc)
R2_edu = 1 - SSR_edu / SST_yeduc;

% パラメター(yeduc)の最小２乗推定量の分散
sigma_beta2_hat = s_hat2 / (SST_yeduc * (1 - R2_edu));

% パラメター(yeduc)の標準誤差
se_beta_hat = sqrt(sigma_beta2_hat);

%=================================
