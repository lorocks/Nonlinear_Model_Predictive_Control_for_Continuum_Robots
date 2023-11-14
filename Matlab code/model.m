function [x0, u0] = model(Ts, x0, u, f)
x0 = full(x0 + (Ts*f(x0,u(1,:)')));
u0 = [u(2:size(u,1),:);u(size(u,1),:)];
end