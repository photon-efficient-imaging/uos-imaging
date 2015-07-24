function [x_hat,i] = COSAMP_UOS(y,A,dels,ite_max)

% v - residual
% a - intermediate solution

[n] = size(A,2);

a = zeros(n,1);
v = y;
errs = zeros(n,1);
for i=1:ite_max
    % compute proxy solution
    a_prev = a;
    x_proxy = A'*v;
    [vals_proxy,inds_proxy] = sort(x_proxy(1:(n-1)),'descend');
    supp_now = inds_proxy(1);
    supp_prev = find(a);
    supp_total = [union(union(supp_now,supp_prev),n)]';
    % compute least-squares on subsignal
    b0 = pinv(A(:,supp_total))*y;
    b = zeros(n,1); b(supp_total) = b0;
    % project solution onto union-of-subspace
    [prune_val,prune_ind] = max(b(1:(n-1)));
    a = zeros(n,1); a(prune_ind) = prune_val; a(n) = b(n);
    % threshold non-negativity
    a(a<0) = 0;
    diff_norm = norm(a_prev-a)^2;
    % compute residual for next ite
    v = y - A*a; 
    % check convergence
    if(diff_norm<dels)
       break; 
    end
end
x_hat = a;

end

