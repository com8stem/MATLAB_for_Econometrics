%load data
file = readtable("filename");
lincome = file{:,5};
yeduc = file{:,3};
exper = file{:,1};
exper2 = file{:,2};

n = numel(lincome);
k = 4 - 1;

%=====最小二乗法によるパラメータ推定=====
X = [ ones(n,1), yeduc, exper, exper2 ];
beta_hat = (X' * X) \ (X' * lincome);

y_hat = beta_hat(1) * ones(n,1) + beta_hat(2) * yeduc...
    + beta_hat(3) * exper + beta_hat(4) * exper2;

%残差
e_i = lincome - y_hat;

%残差２乗和
SSR = e_i' * e_i;

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
SE = sqrt(diag_cov_e)

%=====bootstrapによる標準誤差の推定=====

%ブートストラップの回数
B = 200;

beta_hat_bs = zeros(B, k + 1);

rng('shuffle');
for i = 1:B
    sample_rate = rand(n, 1);
    sample_index = sample_rate * n;
    sample_index = ceil(sample_index);

    lincome_bs = lincome(sample_index, :);
    yeduc_bs = yeduc(sample_index, :);
    exper_bs = exper(sample_index, :);
    exper2_bs = exper2(sample_index, :);

    X_bs = [ones(n, 1), yeduc_bs, exper_bs, exper2_bs];
    beta_hat_bs(i, :) = (X_bs' * X_bs) \ (X_bs' * lincome_bs);
end

%ブートストラップで求めた標準誤差

beta_hat_bs_std = std(beta_hat_bs)
%=================================
