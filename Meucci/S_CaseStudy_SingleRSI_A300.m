clear all
% Case Study 1 
% In Case Study 1 we consider only one risk driver (the 10 year rate) and only one view that the expected 
% value of the 10-year rate will be the actual value minus 50 basis points
% at t^view = 1 year from today.

%load the Calibrated Parameters
load 'CalibratedParameters.mat'
mu = mu(1);
theta = theta(1,1);
sig2 = sig2(1,1);
mu_LT = theta\mu;

%load the simulated path for the 10y Government rate (this is equal to its shadow rate)
load 'path_daily.mat'

start_time = 501;
adjust_period = 10; % 90 periods
end_time = 501 + adjust_period*50;    % 1800 days

RSI_1 = rsindex(A300_path,10);
A300_allPath = A300_path;

b_series = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set the trading
tau = 1/252;                        % trading frequency (daily)
T_Hor = adjust_period / 252;%years             % effective future portfolio horizon   
tau_ = length([0:tau:T_Hor]);       % effective number of future tradings at any point in time
n_ = 1;                             % number of risk drivers 
n = 0;

for time = start_time:adjust_period:end_time
    time
    if exist("Posterior", "var")
        mu = mu_x10y * theta;
        %sig2 = sig2_1;
    end

    if exist('Posterior', 'var')
        b_series = [b_series b_MI_Bellman_post'];
    end
    x0 = A300_allPath(time);
    A300_path = A300_allPath(time:time+adjust_period - 1);
    maxRSI = max(RSI_1(time - 10:time - 1));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %set the view at time 0
    t_view0 = adjust_period / 252;%years           % time of the view at time 0
    if maxRSI >= 70
        mu_x10y = x0 * (1 - 0.15);
    elseif maxRSI <= 30
        mu_x10y = x0 * (1 + 0.12);
    else
        mu_x10y = x0;
    end
    t = [0:tau:T_Hor t_view0]';         % monitoring times
    t_ = length(t);                     % number of monitoring times

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %set the parameters for optimization
    gamma = 10^-2;                      % risk aversion parameter    
    eta = 2;                            % weight of the market impact of transaction
    lambda = log(2)/20;                 % discount (half life 20*tau)
    if exist('Posterior', 'var')
        b_legacy = b_MI_Bellman_post(end);                       % legacy portfolio
    else
        b_legacy = 0
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %compute the prior at time 0
    Prior0 = MVOU_Prior(t, x0, theta, sig2, mu);
    %matrix of market impact
    c2 = Prior0.cov(n_+1:2*n_,n_+1:2*n_);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Prior and Posterior Optimal exposure with Market Impcat of transaction.
    % SOLVING THE BELLMAN EQUATION
    
    [b_MI_Bellman_prior, b_MI_Bellman_post] = BellmanEq_CS1(eta, gamma, lambda, tau, theta, mu, sig2, c2, b_legacy, A300_path, t_view0, mu_x10y);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %CALCULUS OF VARIATION
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %initialize variables
    
    %prior
    b_NoMI_prior = NaN(tau_-1,n_);      %optimal 1-period myopic prior solution 
    b_MI_Vc_prior = NaN(tau_-1,n_);     %optimal prior solution with MI (Calculus of Variation)
    b_legacy_prior = b_legacy;
    b_legacy_prior_LR = b_legacy;
    b_legacy_prior_xt = b_legacy;
    
    %posterior 
    b_NoMI_post = NaN(tau_-1,n_);       %optimal 1-period myopic posterior solution
    b_NoMI_LongTerm = NaN(tau_-1,n_);   %Decomposition of the optimal solution in absence of market impact
    b_NoMI_viewMean = NaN(tau_-1,n_);   %Decomposition of the optimal solution in absence of market impact
    b_MI_Vc_post = NaN(tau_-1,n_);      %optimal posterior solution with MI (Calculus of Variation)
    b_legacy_post = b_legacy;
    b_legacy_post_LR = b_legacy;
    b_legacy_post_xt = b_legacy;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:tau_-1
        t_roll = [[0:tau:T_Hor] t_view0 - tau*(i-1)];
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%The Prior
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %compute the prior
        Prior = MVOU_Prior(t_roll, A300_path(i,:)', theta, sig2, mu);    
        Mean_prior = reshape(Prior.mean(1:t_*n_),n_,t_)';        
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% No Market impact myopic 1 period solution
        ER_prior = Mean_prior(2,1:n_)-Mean_prior(1,1:n_);
        sig2_1 = Prior.cov(n_+1:2*n_,n_+1:2*n_);
        b_NoMI_prior(i,1:n_) = (gamma*sig2_1)\ER_prior(1:n_)';    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Market impact: calculus of variation
        ER = diff(Mean_prior(1:end-1));
        l_t = - exp(-lambda*[0:1:length(ER)-1]').*diff(Mean_prior(1:end-1));
        l_t(1) = l_t(1) - eta*c2*b_legacy_prior;
        q_t = QuadraticMat_Vc(lambda,gamma,eta,Prior.cov,c2,length(ER),n_);
        b_MI_Vc_prior_tmp = (2*q_t)\l_t;
        b_MI_Vc_prior(i,1:n_) = b_MI_Vc_prior_tmp(1:n_);
        b_legacy_prior = b_MI_Vc_prior(i,1:n_);
             
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%The Posterior
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %set the rolling view 
        [T,N] = meshgrid(t_roll,1:n_);
        clear v_mu v_tmp mu_view;
        N_Meanviews = 1;%Number of views on expectations
        v_tmp = zeros(N_Meanviews,n_,t_);
        v_tmp(1,1,end) = 1;
        mu_view = mu_x10y; 
        v_mu = v_tmp(:,:);
        views = struct('N_Meanviews', N_Meanviews,'N_Covviews',[],'dimension',N(:),'monitoring_time',T(:),'v_mu',v_mu,'v_sig',NaN,'mu_view',mu_view,'sig2_view',[]);                    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % compute the posterior moments
        Posterior = MVOU_Posterior(Prior, views);   
        Mean_post = reshape(Posterior.mean(1:t_*n_),n_,t_)';     
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% No Market impact myopic 1 period solution
        ER_post = Mean_post(2,1:n_)-Mean_post(1,1:n_);
        sig2_1 = Posterior.cov(n_+1:2*n_,n_+1:2*n_);
        b_NoMI_post(i,1:n_) = (gamma*sig2_1)\ER_post(1:n_)';    
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Market impact: calculus of variation
        ER = diff(Mean_post(1:end-1));
        l_t = - exp(-lambda*[0:1:length(ER)-1]').*ER;
        l_t(1) = l_t(1) - eta*c2*b_legacy_post;
        q_t = QuadraticMat_Vc(lambda,gamma,eta,Posterior.cov,c2,length(ER),n_);
        b_MI_Vc_post_tmp = (2*q_t)\l_t;
        b_MI_Vc_post(i,1:n_) = b_MI_Vc_post_tmp(1:n_);
        b_legacy_post = b_MI_Vc_post(i,1:n_);
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% LongTerm and viewMean contributions of the optimal solution with views in absence of market impact of transactions 
        b_NoMI_LongTerm(i,1:n_) = 2*theta/(gamma*sig2)/(1+exp(-theta*tau))*(1-(1+exp(theta*tau))/(1+exp(theta*(t_roll(end)))))*(mu/theta-A300_path(i,1));
        b_NoMI_viewMean(i,1:n_) =  2*theta/(gamma*sig2)*exp(theta*tau)/(exp(theta*(t_roll(end)))-exp(-theta*t_roll(end)))*(mu_view-A300_path(i,1));
    end
end

b_series = [b_series b_MI_Bellman_post'];
t_series = [1:length(b_series)];
yyaxis left
ylabel('Exposure') 
line(t_series, b_series)
yyaxis right
ylabel('A300') 
line(t_series, A300_allPath(start_time:end_time + adjust_period - 1))
set(gca, 'XMinorGrid','on');
grid on