clear all
close all
clc

% Import CasAdi
addpath('D:/Casadi')
import casadi.*

Ts = 0.1; % Sample time
T = 40; % Simulation time

% Create required references
time = zeros(1, T/Ts);
x_ref = zeros(1, T/Ts + 1);
y_ref = zeros(1, T/Ts + 1);
z_ref = zeros(1, T/Ts);


for i = 0:1:T/Ts
    time(i+1) = i*Ts;

    x_ref(i+1) = cos(0.25*i*Ts);
    y_ref(i+1) = sin(0.25*i*Ts);
    z_ref(i+1) = 2 + 0.2*i*Ts;
end


N = 20; % prediction horizon

% State symbols
x = MX.sym('x');
y = MX.sym('y');
z = MX.sym('z');
s = MX.sym('s');
t = MX.sym('t');
p = MX.sym('p');

% Control action symbols
s_ = MX.sym('s_');
t_ = MX.sym('t_');
p_ = MX.sym('p_');

% Initialise states and control inputs
states = [x; y; z; s; t; p]; 
controls = [s_; t_; p_];

state_length = length(states);
input_length = length(controls);

% B matrix
B = [cos(p)*(cos(t) - 1)*s_/t + s*cos(p)*(1-cos(t)-(t*sin(t)))*t_/(t*t) + -s*(cos(t)-1)*sin(p)*p_/t;
         sin(p)*(cos(t)-1)*s_/t + s*sin(p)*(1-cos(t)-(t*sin(t)))*t_/(t*t) + s*(cos(t)-1)*cos(p)*p_/t;
         sin(t)*s_/t + ((s*t*cos(t)) - sin(t))*t_/(t*t) + 0;
         s_;
         t_;
         p_];

f = Function('f',{states,controls},{B}); % State space equation
X = SX.sym('X',state_length,(N+1)); % Symbolic states over the prediction horizon
U = SX.sym('U',input_length,N); % Symbolic control actions over the prediction horizon
X_initial = SX.sym('X0', state_length);
X_reference = SX.sym('Xr', N*(state_length+input_length));
%P = SX.sym('P',state_length + N*(state_length+input_length));


J = 0; % Cost function
g = [];  % Constraints equations

% Weight matrices for state error
Q = zeros(state_length,state_length); 
Q(1,1) = 10; 
Q(2,2) = 10; 
Q(3,3) = 1; 

% Weighing matrices for control increment
R = zeros(input_length,input_length); 
R(1,1) = 0.1; 
R(2,2) = 0.1; 
R(3,3) = 0.1; 

P = [X_initial; X_reference]; % Optimisation parameters

g = [g;X(:,1)-P(1:state_length)]; % initial condition constraints
for k = 1:N
    % Compute symbolic cost function over prediction horizon
    J = J+( P((state_length+input_length)*k -2:(state_length+input_length)*k +3) - X(:,k))'*Q*( P((state_length+input_length)*k -2:(state_length+input_length)*k +3) - X(:,k)) + ...
              ( U(:,k) - P((state_length+input_length)*k +4:(state_length+input_length)*k +6))'*R*( U(:,k) - P((state_length+input_length)*k +4:(state_length+input_length)*k +6));

    next_state = X(:,k+1); % Symbolic next state
    next_plant_state = X(:,k) + (Ts*f(X(:,k), U(:,k))); % Next state found using discretised plant model
    g = [g;next_state - next_plant_state]; % Adding constraints for next states
end

% Reshape optimisation variables for input into optimiser
opti_variables = [reshape(X,state_length*(N+1),1);reshape(U,input_length*N,1)];

% Define optimisation problem
opti_problem = struct;
opti_problem.f = J;
opti_problem.x = opti_variables;
opti_problem.g = g;
opti_problem.p = P;

% Define optimisation options
options = struct;
options.ipopt.max_iter = 2000;
options.ipopt.print_level =0;%0,3
options.print_time = 0;
options.ipopt.acceptable_tol =1e-8;
options.ipopt.acceptable_obj_change_tol = 1e-6;

% Create solver
solver = nlpsol('solver', 'ipopt', opti_problem, options);

% Define constraints
constraints = struct;

constraints.lbg(1:state_length*(N+1)) = 0;
constraints.ubg(1:state_length*(N+1)) = 0;

constraints.lbx(1:state_length:state_length*(N+1),1) = -4; %state x lower bound constraint
constraints.ubx(1:state_length:state_length*(N+1),1) = 4; %state x upper bound constraint
constraints.lbx(2:state_length:state_length*(N+1),1) = -4; %state y lower bound constraint
constraints.ubx(2:state_length:state_length*(N+1),1) = 4; %state y upper bound constraint
constraints.lbx(3:state_length:state_length*(N+1),1) = -4; %state z lower bound constraint
constraints.ubx(3:state_length:state_length*(N+1),1) = 4; %state z upper bound constraint
constraints.lbx(4:state_length:state_length*(N+1),1) = 0; %state s lower bound constraint
constraints.ubx(4:state_length:state_length*(N+1),1) = 10; %state s upper bound constraint
constraints.lbx(5:state_length:state_length*(N+1),1) = -pi; %state t lower bound constraint
constraints.ubx(5:state_length:state_length*(N+1),1) = pi; %state t upper bound constraint
constraints.lbx(6:state_length:state_length*(N+1),1) = -pi; %state p lower bound constraint
constraints.ubx(6:state_length:state_length*(N+1),1) = pi; %state p upper bound constraint

constraints.lbx(state_length*(N+1)+1:input_length:state_length*(N+1)+input_length*N,1) = 0; %s_ lower bound constraint
constraints.ubx(state_length*(N+1)+1:input_length:state_length*(N+1)+input_length*N,1) = 0.1; %s_ upper bound constraint
constraints.lbx(state_length*(N+1)+2:input_length:state_length*(N+1)+input_length*N,1) = -pi/10; %t_ lower bound constraint
constraints.ubx(state_length*(N+1)+2:input_length:state_length*(N+1)+input_length*N,1) = pi/10; %t_ upper bound constraint
constraints.lbx(state_length*(N+1)+3:input_length:state_length*(N+1)+input_length*N,1) = -pi/10; %p_ lower bound constraint
constraints.ubx(state_length*(N+1)+3:input_length:state_length*(N+1)+input_length*N,1) = pi/10; %p_ upper bound constraint


% Simulation
x0 = [0 ; 0 ; 0.9; 0.9; 0.0001; 0];% Initial start point


x_states(:,1) = x0; % Holds all states

% Initialise variables for optimizer
u0 = zeros(N,input_length);
X0 = repmat(x0,1,N+1)';

% Start MPC
loop_num = 0;

u_cl = zeros(T/Ts + 1, 3); % Holds all optimal control actions
u_cl(1,:)=[0.05, 0, 0];

while(loop_num < T / Ts)
    constraints.p(1:6) = x0;
    for k = 1:N
        constraints.p((state_length+input_length)*k -2:(state_length+input_length)*k +3) = [x_ref(loop_num+1), y_ref(loop_num+1), z_ref(loop_num+1), 1, 1, 1];
        constraints.p((state_length+input_length)*k +4:(state_length+input_length)*k +6) = u_cl(loop_num+1,:);
    end

    constraints.x0  = [reshape(X0',state_length*(N+1),1);reshape(u0',input_length*N,1)];
    solution = solver('x0', constraints.x0, 'lbx', constraints.lbx, 'ubx', constraints.ubx, 'lbg', constraints.lbg, 'ubg', constraints.ubg,'p',constraints.p');
    
    u = reshape(full(solution.x(state_length*(N+1)+1:end))',input_length,N)'; % Get control actions from solver

    % Apply the control to model for new state
    [x0, u0] = model(Ts, x0, u,f);

    x_states(:,loop_num+2) = x0;

    u_cl(loop_num+2,:) = u(1,:);
    X0 = reshape(full(solution.x(1:state_length*(N+1)))',state_length,N+1)';

    X0 = [X0(2:end,:);X0(end,:)];
    loop_num = loop_num + 1;

end

figure

xlabel("x (m)")
ylabel("y (m)")
hold on
plot(x_ref, y_ref, '--')
hold on
plot(x_states(1,:), x_states(2,:), ':')
hold on

figure
subplot(411)
xlabel("Time")
ylabel("x (m)")
hold on
plot(time, x_ref)
hold on
plot(time, x_states(1,:))
grid on

subplot(412)
xlabel("Time")
ylabel("y (m)")
hold on
plot(time, y_ref)
hold on
plot(time, x_states(2,:))
grid on

subplot(413)
xlabel("Time")
ylabel("z (m)")
hold on
plot(time, z_ref)
hold on
plot(time, x_states(3,:))
grid on

subplot(414)
xlabel("Time")
ylabel("u")
hold on
plot(time, u_cl(:,1), '-') % s straight line
hold on
plot(time, u_cl(:,2), '--') % t dashed line
hold on
plot(time, u_cl(:,3), '-.') % p odotted line
hold on
grid on
